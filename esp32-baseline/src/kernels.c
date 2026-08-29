/*
 * kernels.c - implementation of the low-level kernels.
 */
#define TM_PROFILE 1
#include "kernels.h"
#pragma GCC optimize ("O3")

#include <math.h>
#include <string.h>

/* ================= fp32 GEMM =================
 * C[M,N] = A[M,K]*W[N,K]^T + bias. Row-major A, W stored as N*K rows
 * (torch Linear weight layout). fp32 accumulate in index order (matches
 * torch CPU reference closely enough for the EXACT path's own gate).
 */
void tm_gemm_f32(const float* A, const float* W, const float* bias,
                 float* C, int M, int K, int N, int rowStride) {
    for (int i = 0; i < M; i++) {
        const float* arow = A + (size_t)i * K;
        for (int j = 0; j < N; j++) {
            const float* wrow = W + (size_t)j * K;
            float acc = 0.0f;
            for (int k = 0; k < K; k++) acc += arow[k] * wrow[k];
            C[(size_t)i * rowStride + j] = acc + (bias ? bias[j] : 0.0f);
        }
    }
}

/* ================= Q15 x Q12 fixed-point GEMM =================
 * Activation A quantized to Q15:  a16 = lrintf(A * sa), sa = 32767/max|A|.
 * Weight rows pre-quantized Q12 (scale 2047/maxrow|W|, stored w_scale
 * = max|W|/2047). Output dequant: C = acc * (1/sa) * w_scale + bias.
 * int16*int16 products are exact in int32; the K-term accumulation bound
 * (~ 6e-9 * K) keeps sums far below INT32 saturation for this model.
 */
/* Shared Q15 activation scratch (32 KB).  Used by the FAST projection
 * GEMMs.  Sequential use: the QKV path quantizes g_buf1 ONCE per layer and
 * reuses this buffer across all heads; oproj/f1/f2 re-use it after. */
static int16_t a16[TM_S * TM_D];

/* Accessor so model.c can share the same scratch (no duplicate 32 KB). */
int16_t* tm_gemm_a16(void) { return a16; }

void tm_gemm_q12(const float* A, const int16_t* Wq, float w_scale,
                 const float* bias, float* C,
                 int M, int K, int N, int rowStride) {
    /* find activation scale + quantize A into a static scratch */
    float amax = 0.0f;
    for (int i = 0; i < M * K; i++) {
        float av = A[i] < 0.0f ? -A[i] : A[i];
        if (av > amax) amax = av;
    }
    if (amax == 0.0f) amax = 1.0f;          /* all-zero slice guard */
    float sa = TM_QACT_MAX / amax;
    float sa_inv = 1.0f / sa;
    tm_gemm_quantA_into(A, M * K, a16, sa);
    tm_gemm_core4(a16, sa_inv, Wq, w_scale, bias, C, M, K, N, rowStride);
}


#ifdef TM_PROFILE
#include <stdio.h>
#if defined(__riscv)
#include "esp_timer.h"
#define KB_NOW() ((uint32_t)esp_timer_get_time())
#else
#include <time.h>
static __inline__ uint32_t tm_kb_now_host(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint32_t)((uint64_t)ts.tv_sec * 1000000u + ts.tv_nsec / 1000u);
}
#define KB_NOW() tm_kb_now_host()
#endif
/* kernels-internal microsecond instrumentation (independent of model.c slots) */
static uint32_t k_t0[4];
static uint64_t k_acc[4]; static uint32_t k_n[4];
#define KPB(i) do { k_t0[(i)] = KB_NOW(); } while (0)
#define KPE(i) do { uint32_t d = KB_NOW() - k_t0[(i)]; \
                    k_acc[(i)] += d; k_n[(i)]++; } while (0)
void tm_kbench_clear(void) { for (int i=0;i<4;i++){k_acc[i]=0;k_n[i]=0;} }
void tm_kbench_dump(void) {
    extern void tm_prof_emit(const char* s);
    char line[128];
    for (int i=0;i<4 && k_n[i];i++){
        (void)snprintf(line,sizeof line,"KB%d n=%u avg_us=%.1f tot_ms=%.1f\n",
            i, k_n[i], (double)k_acc[i]/(double)k_n[i],
            (double)k_acc[i]/1000.0);
        tm_prof_emit(line);
    }
}
#else
#define KPB(i) do {} while (0)
#define KPE(i) do {} while (0)
#endif


/* Quantize A into a caller buffer (Q15 fast-round, no libm). sa precomputed. */
void tm_gemm_quantA_into(const float* A, int n, int16_t* out, float sa) {
    for (int i = 0; i < n; i++) {
        float v = A[i] * sa;
        int q;
        if (v >= 0.0f) {
            q = (int)(v + 0.5f);
            if (q > TM_QACT_MAX) q = (int)TM_QACT_MAX;
        } else {
            q = (int)(v - 0.5f);
            if (q < -TM_QACT_MAX) q = -(int)TM_QACT_MAX;
        }
        out[i] = (int16_t)q;
    }
}

/* Amax scan of A; returns the Q15 scale sa = TM_QACT_MAX/amax. */
float tm_gemm_amax(const float* A, int n) {
    float amax = 0.0f;
    for (int i = 0; i < n; i++) {
        float av = A[i] < 0.0f ? -A[i] : A[i];
        if (av > amax) amax = av;
    }
    if (amax == 0.0f) amax = 1.0f;
    return TM_QACT_MAX / amax;
}

/* Quant-accumulate core (CMSIS arm_mat_mult_fast_q15 style): 4-way unrolled
 * inner loop with 4 independent int32 accumulators.  int16*int16 exact in
 * int32; each accumulator sees K/4 terms so far below saturation here. */
void tm_gemm_core(const int16_t* Aq, float sa_inv, const int16_t* Wq,
                  float w_scale, const float* bias, float* C,
                  int M, int K, int N, int rowStride) {
    const float g = sa_inv * w_scale;
    for (int i = 0; i < M; i++) {
        const int16_t* arow = Aq + (size_t)i * K;
        for (int j = 0; j < N; j++) {
            const int16_t* wrow = Wq + (size_t)j * K;
            int32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
            int k = 0;
            for (; k + 4 <= K; k += 4) {
                int32_t a0 = arow[k], a1 = arow[k+1], a2 = arow[k+2], a3 = arow[k+3];
                int32_t b0 = wrow[k], b1 = wrow[k+1], b2 = wrow[k+2], b3 = wrow[k+3];
                acc0 += a0 * b0;
                acc1 += a1 * b1;
                acc2 += a2 * b2;
                acc3 += a3 * b3;
            }
            for (; k < K; k++)
                acc0 += (int32_t)arow[k] * (int32_t)wrow[k];
            C[(size_t)i * rowStride + j] =
                (float)(acc0 + acc1 + acc2 + acc3) * g + (bias ? bias[j] : 0.0f);
        }
    }
}

/* Tiled quant-accumulate core v2: 2-row i-tile reuses each weight row for
 * two i's (halves flash XIP reads of the 32 KB weight blocks), 4-way j-unroll
 * into register accumulators.  Inner k body does 8 MACs/iteration.
 * K=TM_D=128, M=TM_S=128; IBLK=2, JBLK=4 (8 accs fit RV32 registers). */
void tm_gemm_core2(const int16_t* Aq, float sa_inv, const int16_t* Wq,
                   float w_scale, const float* bias, float* C,
                   int M, int K, int N, int rowStride) {
    const float g = sa_inv * w_scale;
    enum { IBLK = 2, JBLK = 4 };
    for (int it = 0; it < M; it += IBLK) {
        const int16_t* a0r = Aq + (size_t)it * K;
        const int16_t* a1r = Aq + (size_t)(it + 1) * K;
        for (int jb = 0; jb < N; jb += JBLK) {
            const int16_t* wr = Wq + (size_t)jb * K;
            int32_t c00 = 0, c01 = 0, c02 = 0, c03 = 0;
            int32_t c10 = 0, c11 = 0, c12 = 0, c13 = 0;
            for (int k = 0; k < K; k++) {
                int32_t a0 = a0r[k], a1 = a1r[k];
                int32_t b0 = wr[k], b1 = wr[K + k], b2 = wr[2*K + k], b3 = wr[3*K + k];
                c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3;
                c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3;
            }
            float* cr0 = C + (size_t)it * rowStride + jb;
            float* cr1 = C + (size_t)(it + 1) * rowStride + jb;
            cr0[0] = (float)c00 * g + bias[jb];     cr0[1] = (float)c01 * g + bias[jb+1];
            cr0[2] = (float)c02 * g + bias[jb+2];   cr0[3] = (float)c03 * g + bias[jb+3];
            cr1[0] = (float)c10 * g + bias[jb];     cr1[1] = (float)c11 * g + bias[jb+1];
            cr1[2] = (float)c12 * g + bias[jb+2];   cr1[3] = (float)c13 * g + bias[jb+3];
        }
    }
}

/* Tiled core v3: 4-row i-tile halves weight bytes fetched from flash vs v2
 * (weights in registers are reused across 4 output rows within a k step) and
 * 2-wide j-unroll keeps 8 accumulator registers.  N and M multiples of 4/2. */

/* Fused per-head GEMM that quantizes to int16 Q15 directly (no fp32 staging).
 * Pass 1: 4-row i-tiled core3 gemm -> int32 accumulator scratch (per head),
 *          tracking the GLOBAL real-value amax (fp32 fold: acc*g + bias) so the
 *          dequant scale matches quant_head exactly (single per-buffer scale).
 * Pass 2: fixed-point quantize, ~12 cyc/element vs ~250 for soft-float q15:
 *          q15 = (acc*GX + BX[d]) >> 15,  GX = g*sa*2^15, BX = bias*sa*2^15,
 *          computed in int64 to stay overflow-safe (acc can approach 2^31).
 * Returns dequant scale amax/32767 (same contract as quant_head). */
float tm_gemm_head_q15(const int16_t* Aq, float sa_inv, const int16_t* Wq,
                       float w_scale, const float* bias,
                       int32_t* acc, int16_t* dst, int K) {
    KPB(0);
    const float g = sa_inv * w_scale;
    const int S = TM_S, HD = TM_HD;
    /* Pass 1: j-outer GEMM (sequential W columns, flash-cache friendly like
     * core4) straight into the int32 accumulator scratch.  Tracks per-column
     * min/max of the int32 accs so amax is EXACT but requires only 2*HD fp32
     * evals per gemm (c*g+b monotone in c, g>0) instead of one per output. */
    int32_t cmin[TM_HD], cmax[TM_HD];
    for (int d = 0; d < HD; d++) { cmin[d] = 0x7fffffff; cmax[d] = 0x80000000; }
    enum { IBLK = 8 };
    for (int j = 0; j < HD; j++) {
        const int16_t* wr = Wq + (size_t)j * K;
        for (int it = 0; it < S; it += IBLK) {
            const int16_t* a0 = Aq + (size_t)(it+0) * K;
            const int16_t* a1 = Aq + (size_t)(it+1) * K;
            const int16_t* a2 = Aq + (size_t)(it+2) * K;
            const int16_t* a3 = Aq + (size_t)(it+3) * K;
            const int16_t* a4 = Aq + (size_t)(it+4) * K;
            const int16_t* a5 = Aq + (size_t)(it+5) * K;
            const int16_t* a6 = Aq + (size_t)(it+6) * K;
            const int16_t* a7 = Aq + (size_t)(it+7) * K;
            int32_t c0=0,c1=0,c2=0,c3=0,c4=0,c5=0,c6=0,c7=0;
            for (int k = 0; k < K; k++) {
                int32_t b = wr[k];
                c0 += (int32_t)a0[k] * b;
                c1 += (int32_t)a1[k] * b;
                c2 += (int32_t)a2[k] * b;
                c3 += (int32_t)a3[k] * b;
                c4 += (int32_t)a4[k] * b;
                c5 += (int32_t)a5[k] * b;
                c6 += (int32_t)a6[k] * b;
                c7 += (int32_t)a7[k] * b;
            }
            int32_t mn = c0 < c1 ? c0 : c1, mx = c0 > c1 ? c0 : c1;
            mn = mn < c2 ? mn : c2; mx = mx > c2 ? mx : c2;
            mn = mn < c3 ? mn : c3; mx = mx > c3 ? mx : c3;
            mn = mn < c4 ? mn : c4; mx = mx > c4 ? mx : c4;
            mn = mn < c5 ? mn : c5; mx = mx > c5 ? mx : c5;
            mn = mn < c6 ? mn : c6; mx = mx > c6 ? mx : c6;
            mn = mn < c7 ? mn : c7; mx = mx > c7 ? mx : c7;
            if (mn < cmin[j]) cmin[j] = mn;
            if (mx > cmax[j]) cmax[j] = mx;
            int32_t* r0 = acc + (size_t)it * HD + j;
            r0[0*HD] = c0; r0[1*HD] = c1; r0[2*HD] = c2; r0[3*HD] = c3;
            r0[4*HD] = c4; r0[5*HD] = c5; r0[6*HD] = c6; r0[7*HD] = c7;
        }
    }
    /* exact amax from per-column extremes (matches the old per-output fp32
     * eval exactly: same expressions, only evaluated at the extreme c's) */
    KPB(3);
    float amax = 0.0f;
    for (int d = 0; d < HD; d++) {
        float v1 = (float)cmax[d] * g + bias[d];
        float v2 = (float)cmin[d] * g + bias[d];
        if (v1 < 0.0f) v1 = -v1;
        if (v2 < 0.0f) v2 = -v2;
        if (v1 > amax) amax = v1;
        if (v2 > amax) amax = v2;
    }
    KPE(3);
    float sa = TM_QACT_MAX / (amax > 0.0f ? amax : 1.0f);
    KPE(0);
    /* Fixed-point pass in Q30 (int64): coefficient rounding error * acc must
     * stay below 0.5 q15 unit; a Q15 coefficient loses precision whenever acc
     * is large (acc can approach 2^31), so Q30 + int64 is required. */
    KPB(1);
const float gsa = g * sa;
    const int64_t GX = (int64_t)(gsa * 1073741824.0f);          /* *2^30 */
    int64_t BX[TM_HD];
    for (int d = 0; d < HD; d++)
        BX[d] = (int64_t)(bias[d] * sa * 1073741824.0f);
    for (int i = 0; i < S; i++) {
        const int32_t* ra = acc + (size_t)i * HD;
        int16_t* dd = dst + (size_t)i * HD;
        for (int d = 0; d < HD; d++) {
            int64_t t = (int64_t)ra[d] * GX + BX[d];
            int64_t tt = (t >= 0) ? (t + 536870912) : (-t + 536870912); /* +2^29 */
            int64_t qq = (t >= 0) ? tt : -tt;
            int32_t q = (int32_t)(qq >> 30);
            if (q > TM_QACT_MAX) q = (int)TM_QACT_MAX;
            else if (q < (int)-TM_QACT_MAX) q = -(int)TM_QACT_MAX;
            dd[d] = (int16_t)q;
        }
    }
    KPE(1);
    return amax / TM_QACT_MAX;
}


/* core4: j-outer, 8-row i-tile, 8 int32 register accs.  Each flash weight
 * column is read once per 8 rows -> M*N*K*2/8 bytes of flash XIP (half of
 * core3), at the cost of re-reading A (SRAM) per output column.  Best for
 * the N=K=128 gemms (oproj/FFN1/FFN2) where the weight operand is
 * flash-bound. */
void tm_gemm_core4(const int16_t* Aq, float sa_inv, const int16_t* Wq,
                   float w_scale, const float* bias, float* C,
                   int M, int K, int N, int rowStride) {
    const float g = sa_inv * w_scale;
    enum { IBLK = 8 };
KPB(2);
    for (int j = 0; j < N; j++) {
        const int16_t* wr = Wq + (size_t)j * K;
        for (int it = 0; it < M; it += IBLK) {
            const int16_t* a0 = Aq + (size_t)(it+0) * K;
            const int16_t* a1 = Aq + (size_t)(it+1) * K;
            const int16_t* a2 = Aq + (size_t)(it+2) * K;
            const int16_t* a3 = Aq + (size_t)(it+3) * K;
            const int16_t* a4 = Aq + (size_t)(it+4) * K;
            const int16_t* a5 = Aq + (size_t)(it+5) * K;
            const int16_t* a6 = Aq + (size_t)(it+6) * K;
            const int16_t* a7 = Aq + (size_t)(it+7) * K;
            int32_t c0=0,c1=0,c2=0,c3=0,c4=0,c5=0,c6=0,c7=0;
            for (int k = 0; k < K; k++) {
                int32_t b = wr[k];
                c0 += (int32_t)a0[k] * b;
                c1 += (int32_t)a1[k] * b;
                c2 += (int32_t)a2[k] * b;
                c3 += (int32_t)a3[k] * b;
                c4 += (int32_t)a4[k] * b;
                c5 += (int32_t)a5[k] * b;
                c6 += (int32_t)a6[k] * b;
                c7 += (int32_t)a7[k] * b;
            }
            float bj = bias ? bias[j] : 0.0f;
            float* c0r = C + (size_t)it * rowStride + j;
            c0r[0]           = (float)c0 * g + bj;
            c0r[1*rowStride] = (float)c1 * g + bj;
            c0r[2*rowStride] = (float)c2 * g + bj;
            c0r[3*rowStride] = (float)c3 * g + bj;
            c0r[4*rowStride] = (float)c4 * g + bj;
            c0r[5*rowStride] = (float)c5 * g + bj;
            c0r[6*rowStride] = (float)c6 * g + bj;
            c0r[7*rowStride] = (float)c7 * g + bj;
        }
    }
    KPE(2);
}

void tm_gemm_core4_v2(const int16_t* Aq, float sa_inv, const int16_t* Wq,
                     float w_scale, const float* bias, float* C,
                     int M, int K, int N, int rowStride) {
    /* j-tile-2 (2 output cols) x 8-row x K-pair: 16 int32 accs.  Each A q15
     * load is reused across 2 weight cols (halves A-side lhu per MAC) and the
     * 32 independent mul/add chains with all 20 loads hoisted ahead of the
     * muls hide load->use latency on the 4-stage in-order RV32 core. */
    const float g = sa_inv * w_scale;
    for (int j = 0; j + 1 < N; j += 2) {
        const int16_t* w0 = Wq + (size_t)j * K;
        const int16_t* w1 = Wq + (size_t)(j + 1) * K;
        for (int it = 0; it < M; it += 8) {
            const int16_t* a0 = Aq + (size_t)(it + 0) * K;
            const int16_t* a1 = Aq + (size_t)(it + 1) * K;
            const int16_t* a2 = Aq + (size_t)(it + 2) * K;
            const int16_t* a3 = Aq + (size_t)(it + 3) * K;
            const int16_t* a4 = Aq + (size_t)(it + 4) * K;
            const int16_t* a5 = Aq + (size_t)(it + 5) * K;
            const int16_t* a6 = Aq + (size_t)(it + 6) * K;
            const int16_t* a7 = Aq + (size_t)(it + 7) * K;
            int32_t c00=0,c10=0,c20=0,c30=0,c40=0,c50=0,c60=0,c70=0;
            int32_t c01=0,c11=0,c21=0,c31=0,c41=0,c51=0,c61=0,c71=0;
            int k = 0;
            for (; k + 1 < K; k += 2) {
                int32_t b0 = w0[k],   b1 = w1[k];
                int32_t b0n = w0[k+1], b1n = w1[k+1];
                int32_t v00=a0[k],   v10=a1[k],   v20=a2[k],   v30=a3[k];
                int32_t v01=a0[k+1], v11=a1[k+1], v21=a2[k+1], v31=a3[k+1];
                int32_t v40=a4[k],   v50=a5[k],   v60=a6[k],   v70=a7[k];
                int32_t v41=a4[k+1], v51=a5[k+1], v61=a6[k+1], v71=a7[k+1];
                c00 += v00*b0; c10 += v10*b0; c20 += v20*b0; c30 += v30*b0;
                c40 += v40*b0; c50 += v50*b0; c60 += v60*b0; c70 += v70*b0;
                c01 += v00*b1; c11 += v10*b1; c21 += v20*b1; c31 += v30*b1;
                c41 += v40*b1; c51 += v50*b1; c61 += v60*b1; c71 += v70*b1;
                c00 += v01*b0n; c10 += v11*b0n; c20 += v21*b0n; c30 += v31*b0n;
                c40 += v41*b0n; c50 += v51*b0n; c60 += v61*b0n; c70 += v71*b0n;
                c01 += v01*b1n; c11 += v11*b1n; c21 += v21*b1n; c31 += v31*b1n;
                c41 += v41*b1n; c51 += v51*b1n; c61 += v61*b1n; c71 += v71*b1n;
            }
            for (; k < K; k++) {
                int32_t b0 = w0[k], b1 = w1[k];
                int32_t v00=a0[k], v10=a1[k], v20=a2[k], v30=a3[k];
                int32_t v40=a4[k], v50=a5[k], v60=a6[k], v70=a7[k];
                c00 += v00*b0; c10 += v10*b0; c20 += v20*b0; c30 += v30*b0;
                c40 += v40*b0; c50 += v50*b0; c60 += v60*b0; c70 += v70*b0;
                c01 += v00*b1; c11 += v10*b1; c21 += v20*b1; c31 += v30*b1;
                c41 += v40*b1; c51 += v50*b1; c61 += v60*b1; c71 += v70*b1;
            }
            float bj  = bias ? bias[j] : 0.0f;
            float bj1 = bias ? bias[j+1] : 0.0f;
            float* c0r = C + (size_t)it * rowStride + j;
            c0r[0]              = (float)c00 * g + bj;
            c0r[1*rowStride]    = (float)c10 * g + bj;
            c0r[2*rowStride]    = (float)c20 * g + bj;
            c0r[3*rowStride]    = (float)c30 * g + bj;
            c0r[4*rowStride]    = (float)c40 * g + bj;
            c0r[5*rowStride]    = (float)c50 * g + bj;
            c0r[6*rowStride]    = (float)c60 * g + bj;
            c0r[7*rowStride]    = (float)c70 * g + bj;
            c0r[1]              = (float)c01 * g + bj1;
            c0r[1+1*rowStride]  = (float)c11 * g + bj1;
            c0r[1+2*rowStride]  = (float)c21 * g + bj1;
            c0r[1+3*rowStride]  = (float)c31 * g + bj1;
            c0r[1+4*rowStride]  = (float)c41 * g + bj1;
            c0r[1+5*rowStride]  = (float)c51 * g + bj1;
            c0r[1+6*rowStride]  = (float)c61 * g + bj1;
            c0r[1+7*rowStride]  = (float)c71 * g + bj1;
        }
    }
    if (N & 1) {  /* odd-N tail: single column */
        int j = N - 1;
        const int16_t* wr = Wq + (size_t)j * K;
        for (int it = 0; it < M; it += 8) {
            const int16_t* a0 = Aq + (size_t)(it+0) * K;
            const int16_t* a1 = Aq + (size_t)(it+1) * K;
            const int16_t* a2 = Aq + (size_t)(it+2) * K;
            const int16_t* a3 = Aq + (size_t)(it+3) * K;
            const int16_t* a4 = Aq + (size_t)(it+4) * K;
            const int16_t* a5 = Aq + (size_t)(it+5) * K;
            const int16_t* a6 = Aq + (size_t)(it+6) * K;
            const int16_t* a7 = Aq + (size_t)(it+7) * K;
            int32_t c0=0,c1=0,c2=0,c3=0,c4=0,c5=0,c6=0,c7=0;
            for (int k = 0; k < K; k++) {
                int32_t b = wr[k];
                c0 += (int32_t)a0[k]*b; c1 += (int32_t)a1[k]*b; c2 += (int32_t)a2[k]*b; c3 += (int32_t)a3[k]*b;
                c4 += (int32_t)a4[k]*b; c5 += (int32_t)a5[k]*b; c6 += (int32_t)a6[k]*b; c7 += (int32_t)a7[k]*b;
            }
            float bj = bias ? bias[j] : 0.0f;
            float* c0r = C + (size_t)it * rowStride + j;
            c0r[0]           = (float)c0 * g + bj;
            c0r[1*rowStride] = (float)c1 * g + bj;
            c0r[2*rowStride] = (float)c2 * g + bj;
            c0r[3*rowStride] = (float)c3 * g + bj;
            c0r[4*rowStride] = (float)c4 * g + bj;
            c0r[5*rowStride] = (float)c5 * g + bj;
            c0r[6*rowStride] = (float)c6 * g + bj;
            c0r[7*rowStride] = (float)c7 * g + bj;
        }
    }
}

/* accumulate variant: C += Aq*Wq*g (+bias when bias!=NULL). Used by the
 * per-head oproj split so attention can emit Q15 ctx without a requant pass. */
void tm_gemm_core4_acc(const int16_t* Aq, float sa_inv, const int16_t* Wq,
                       float w_scale, const float* bias, float* C,
                       int M, int K, int N, int rowStride, int wStride,
                       int first) {
    const float g = sa_inv * w_scale;
    enum { IBLK = 8 };
    for (int j = 0; j < N; j++) {
        const int16_t* wr = Wq + (size_t)j * wStride;
        for (int it = 0; it < M; it += IBLK) {
            const int16_t* a0 = Aq + (size_t)(it+0) * K;
            const int16_t* a1 = Aq + (size_t)(it+1) * K;
            const int16_t* a2 = Aq + (size_t)(it+2) * K;
            const int16_t* a3 = Aq + (size_t)(it+3) * K;
            const int16_t* a4 = Aq + (size_t)(it+4) * K;
            const int16_t* a5 = Aq + (size_t)(it+5) * K;
            const int16_t* a6 = Aq + (size_t)(it+6) * K;
            const int16_t* a7 = Aq + (size_t)(it+7) * K;
            int32_t c0=0,c1=0,c2=0,c3=0,c4=0,c5=0,c6=0,c7=0;
            for (int k = 0; k < K; k++) {
                int32_t b = wr[k];
                c0 += (int32_t)a0[k] * b;
                c1 += (int32_t)a1[k] * b;
                c2 += (int32_t)a2[k] * b;
                c3 += (int32_t)a3[k] * b;
                c4 += (int32_t)a4[k] * b;
                c5 += (int32_t)a5[k] * b;
                c6 += (int32_t)a6[k] * b;
                c7 += (int32_t)a7[k] * b;
            }
            float bj = bias ? bias[j] : 0.0f;
            float* c0r = C + (size_t)it * rowStride + j;
            if (first) {
                c0r[0]           = (float)c0 * g + bj;
                c0r[1*rowStride] = (float)c1 * g + bj;
                c0r[2*rowStride] = (float)c2 * g + bj;
                c0r[3*rowStride] = (float)c3 * g + bj;
                c0r[4*rowStride] = (float)c4 * g + bj;
                c0r[5*rowStride] = (float)c5 * g + bj;
                c0r[6*rowStride] = (float)c6 * g + bj;
                c0r[7*rowStride] = (float)c7 * g + bj;
            } else {
                c0r[0]           += (float)c0 * g;
                c0r[1*rowStride] += (float)c1 * g;
                c0r[2*rowStride] += (float)c2 * g;
                c0r[3*rowStride] += (float)c3 * g;
                c0r[4*rowStride] += (float)c4 * g;
                c0r[5*rowStride] += (float)c5 * g;
                c0r[6*rowStride] += (float)c6 * g;
                c0r[7*rowStride] += (float)c7 * g;
            }
        }
    }
}

void tm_gemm_core3(const int16_t* Aq, float sa_inv, const int16_t* Wq,
                   float w_scale, const float* bias, float* C,
                   int M, int K, int N, int rowStride) {
    const float g = sa_inv * w_scale;
    enum { IBLK = 4, JBLK = 2 };
    for (int it = 0; it < M; it += IBLK) {
        const int16_t* a0r = Aq + (size_t)it * K;
        const int16_t* a1r = Aq + (size_t)(it+1) * K;
        const int16_t* a2r = Aq + (size_t)(it+2) * K;
        const int16_t* a3r = Aq + (size_t)(it+3) * K;
        for (int jb = 0; jb < N; jb += JBLK) {
            const int16_t* wr = Wq + (size_t)jb * K;
            int32_t c00=0,c01=0, c10=0,c11=0, c20=0,c21=0, c30=0,c31=0;
            for (int k = 0; k < K; k++) {
                int32_t a0 = a0r[k], a1 = a1r[k], a2 = a2r[k], a3 = a3r[k];
                int32_t b0 = wr[k], b1 = wr[K + k];
                c00 += a0*b0; c01 += a0*b1;
                c10 += a1*b0; c11 += a1*b1;
                c20 += a2*b0; c21 += a2*b1;
                c30 += a3*b0; c31 += a3*b1;
            }
            float* c0r = C + (size_t)it * rowStride + jb;
            float* c1r = c0r + rowStride;
            float* c2r = c1r + rowStride;
            float* c3r = c2r + rowStride;
            c0r[0]=(float)c00*g+bias[jb]; c0r[1]=(float)c01*g+bias[jb+1];
            c1r[0]=(float)c10*g+bias[jb]; c1r[1]=(float)c11*g+bias[jb+1];
            c2r[0]=(float)c20*g+bias[jb]; c2r[1]=(float)c21*g+bias[jb+1];
            c3r[0]=(float)c30*g+bias[jb]; c3r[1]=(float)c31*g+bias[jb+1];
        }
    }
}

/* ================= LayerNorm (two-pass fp32) ================= */
void tm_layernorm(const float* in, const float* gamma, const float* beta,
                  float* out, int S, int D) {
    for (int i = 0; i < S; i++) {
        const float* src = in + (size_t)i * D;
        float* dst = out + (size_t)i * D;
        float mean = 0.0f;
        for (int k = 0; k < D; k++) mean += src[k];
        mean /= (float)D;
        float var = 0.0f;
        for (int k = 0; k < D; k++) {
            float t = src[k] - mean;
            var += t * t;
        }
        var /= (float)D;
        float rstd = 1.0f / sqrtf(var + TM_LN_EPS);
        for (int k = 0; k < D; k++)
            dst[k] = (src[k] - mean) * rstd * gamma[k] + beta[k];
    }
}

/* ---- fused LayerNorm -> Q15 (FAST path) ----
 * Computes LN stats once, derives a SAFE per-buffer amax from a tight bound
 * (max over k of Bmax*|gamma_k| + |beta_k| with Bmax = max_row rstd*(max|x| +
 * |mean|); measured 1.01x median / 1.12x worst vs true max), then emits the
 * Q15 activations in ONE normalize+round pass.  Replaces the fp32 LN
 * normalize + separate amax scan + separate quantize in the qkv path.
 * Returns the activation quant scale sa = 32767/amax (== tm_gemm_amax
 * contract).  output_q replaces the fp32 out buffer.
 */
float tm_bn_q15(const float* in, const float* gamma, const float* beta,
                int16_t* out_q, int S, int D) {
    /* Pass 1: per-row sum, sumsq, max|x| (one fused scan). */
    float sums[TM_S], sqs[TM_S], mxx[TM_S];
    float bmax = 0.0f;
    {
        float sum0 = 0.f, sq0 = 0.f, mx0 = 0.f;
        for (int i = 0; i < S; i++) {
            const float* src = in + (size_t)i * D;
            sum0 = 0.f; sq0 = 0.f; mx0 = 0.f;
            for (int k = 0; k < D; k++) {
                float t = src[k];
                sum0 += t; sq0 += t * t;
                float a = t < 0.f ? -t : t;
                if (a > mx0) mx0 = a;
            }
            sums[i] = sum0; sqs[i] = sq0; mxx[i] = mx0;
        }
    }
    float mean_r[TM_S], rstd_r[TM_S];
    for (int i = 0; i < S; i++) {
        float mean = sums[i] / (float)D;
        float sq = sqs[i] / (float)D;
        float vr = sq - mean * mean;
        if (vr < 0.f) vr = 0.f;
        float rstd = 1.0f / sqrtf(vr + TM_LN_EPS);
        mean_r[i] = mean; rstd_r[i] = rstd;
        float br = rstd * (mxx[i] + (mean < 0.f ? -mean : mean));
        if (br > bmax) bmax = br;
    }
    /* safe per-buffer amax bound (median 1.01x, worst ~1.12x of true max) */
    float amb = 0.0f;
    for (int k = 0; k < D; k++) {
        float g = gamma[k] < 0.f ? -gamma[k] : gamma[k];
        float b = beta[k]  < 0.f ? -beta[k]  : beta[k];
        float v = bmax * g + b;
        if (v > amb) amb = v;
    }
    if (!(amb > 0.f)) amb = 1.f;
    /* 0.9999 safety => |q| <= 32767 always => no per-element clamp */
    float sa = TM_QACT_MAX / amb * 0.9999f;
    float AG[TM_D], AB[TM_D];
    for (int k = 0; k < D; k++) { AG[k] = sa * gamma[k]; AB[k] = sa * beta[k]; }
    /* Pass 2 (normalize+quant fused): out_q = round(A*x + B), A/B per row. */
    for (int i = 0; i < S; i++) {
        const float* src = in + (size_t)i * D;
        int16_t* dst = out_q + (size_t)i * D;
        float mean = mean_r[i], rstd = rstd_r[i];
        float bA[TM_D], bB[TM_D];
        for (int k = 0; k < D; k++) {
            float A = AG[k] * rstd;
            bA[k] = A; bB[k] = AB[k] - mean * A;
        }
        for (int k = 0; k < D; k++) {
            float qf = bA[k] * src[k] + bB[k];
            int32_t q = (qf >= 0.f) ? (int32_t)(qf + 0.5f) : (int32_t)(qf - 0.5f);
            dst[k] = (int16_t)q;
        }
    }
    return sa; /* the scale actually used on the a16 values */
}

/* Integer LayerNorm -> Q15 (single output quant, int stats + fixed-point
 * normalize).  Only per-row math (mean/var/rstd/bound) touches fp32; the
 * per-element hot path is 1 fcvt + integer adds/macs + 2 int64->shift->int32
 * rounding steps.  Returns the Q15 scale applied to out_q (same contract as
 * tm_bn_q15).  Numerics laid out so the output amplitude matches tm_bn_q15:
 *   out_q = round( ((x_q-mean_q) * Fk_r15 >> 15) * rstd_m15 >> 15 + Bk ) */
float tm_bn_q15_int(const float* in, const float* gamma, const float* beta,
                    int16_t* out_q, int S, int D) {
    const int32_t Q = 32767;
    /* input scale (exact amax -> Q15) */
    float amax = 0.0f;
    const int32_t N = S * D;
    for (int32_t i = 0; i < N; i++) {
        float t = in[i]; t = t < 0.0f ? -t : t;
        if (t > amax) amax = t;
    }
    if (!(amax > 0.0f)) amax = 1.0f;
    const float sxi = 1.0f / ((float)Q / amax);   /* = 1/sx */
    const float sxi2 = sxi * sxi;
    /* pass 1: quantize x once, accumulate integer row stats */
    int32_t row = 0;
    static int32_t s1_[TM_S]; static int64_t s2_[TM_S]; static int16_t mx_[TM_S];
    float bmax = 0.0f;
    for (int32_t i = 0; i < S; i++) {
        int32_t s1 = 0; int64_t s2 = 0; int32_t mx = 0;
        const float* src = in + (size_t)i * D;
        int16_t* dst = out_q + (size_t)i * D;
        for (int32_t k = 0; k < D; k++) {
            float qf = src[k] * ((float)Q / amax);
            int32_t q = (qf >= 0.0f) ? (int32_t)(qf + 0.5f) : (int32_t)(qf - 0.5f);
            if (q > Q) q = Q; else if (q < -Q) q = -Q;
            int32_t a = q < 0 ? -q : q;
            if (a > mx) mx = a;
            s1 += q;
            s2 += (int64_t)q * (int64_t)q;
            dst[k] = (int16_t)q;
        }
        s1_[i] = s1; s2_[i] = s2; mx_[i] = (int16_t)mx;
        row++;
    }
    /* per row: mean/var/rstd (fp32, S rows only) + amax bound */
    float mean_r[TM_S], rstd_r[TM_S]; int32_t meanq_[TM_S]; int32_t rstd15_[TM_S];
    for (int32_t i = 0; i < S; i++) {
        int32_t mq = s1_[i] / D;                       /* D=128 -> exact-ish */
        int64_t vq = (s2_[i] / D) - (int64_t)mq * (int64_t)mq;
        if (vq < 0) vq = 0;
        float mean = (float)mq * sxi;                  /* real mean */
        float vr = (float)vq * sxi2;                   /* real variance */
        float rstd = 1.0f / sqrtf(vr + TM_LN_EPS);
        meanq_[i] = mq;  mean_r[i] = mean; rstd_r[i] = rstd;
        rstd15_[i] = (int32_t)(rstd * 32768.0f + 0.5f);
        float br = rstd * ((float)mx_[i] * sxi + (mean < 0.0f ? -mean : mean));
        if (br > bmax) bmax = br;
    }
    float amb = 0.0f;
    for (int32_t k = 0; k < D; k++) {
        float g = gamma[k] < 0.0f ? -gamma[k] : gamma[k];
        float b = beta[k]  < 0.0f ? -beta[k]  : beta[k];
        float v = bmax * g + b;
        if (v > amb) amb = v;
    }
    if (!(amb > 0.0f)) amb = 1.0f;
    float sa = TM_QACT_MAX / amb * 0.9999f;
    static int32_t Fk_[TM_D]; static int32_t Bk_[TM_D];
    for (int32_t k = 0; k < D; k++) {
        float F = sa * gamma[k] * sxi;
        Fk_[k] = (int32_t)(F * 32768.0f + (F < 0.0f ? -0.5f : 0.5f));
        Bk_[k] = (int32_t)(sa * beta[k] + (sa * beta[k] < 0.0f ? -0.5f : 0.5f));
    }
    /* pass 2: fixed-point normalize in place (int hot loop) */
    const int32_t R = 1 << 14;   /* round constant for >>15 */
    for (int32_t i = 0; i < S; i++) {
        int16_t* p = out_q + (size_t)i * D;
        int32_t mq = meanq_[i], r15 = rstd15_[i];
        for (int32_t k = 0; k < D; k++) {
            int32_t d = (int32_t)p[k] - mq;
            int64_t p64 = (int64_t)d * Fk_[k];
            int32_t t = (int32_t)((p64 + (p64 < 0 ? -R : R)) >> 15);
            int64_t q64 = (int64_t)t * r15;
            int32_t u = (int32_t)((q64 + (q64 < 0 ? -R : R)) >> 15);
            int32_t q = u + Bk_[k];
            if (q > Q) q = Q; else if (q < -Q) q = -Q;
            p[k] = (int16_t)q;
        }
    }
    return sa;
}

/* ================= GELU (deg-11 poly erf) =================
 * Coeffs are a weighted LSQ minimax of erf on [0,4] in t = x*0.5 - 1
 * (fp32-rounded; validated vs scipy.special.erf: max abs err 4.6e-5).
 * NOTE: Horner must run highest-power -> constant (coef[0] = constant term).
 */
static const float TM_ERF_C[] = {
     0.995298564f,   /* t^0  */
     0.041422032f,   /* t^1  */
    -0.163504452f,   /* t^2  */
     0.383033544f,   /* t^3  */
    -0.573859155f,   /* t^4  */
     0.443728685f,   /* t^5  */
     0.131832853f,   /* t^6  */
    -0.531797051f,   /* t^7  */
     0.201801717f,   /* t^8  */
     0.178449884f,   /* t^9  */
    -0.091607437f,   /* t^10 */
    -0.014828608f    /* t^11 */
};
static float tm_erf_poly(float x) {
    /* |x| < 4 needed */
    float ax = fabsf(x);
    if (ax >= 4.0f) return x < 0.0f ? -1.0f : 1.0f;
    float t = 0.5f * ax - 1.0f;
    float p = TM_ERF_C[11];
    for (int k = 10; k >= 0; k--) p = p * t + TM_ERF_C[k];
    return x < 0.0f ? -p : p;
}
void tm_gelu_inplace(float* x, int n) {
    static const float inv_sqrt2 = 0.7071067690849304f;
    for (int i = 0; i < n; i++) {
        float u = x[i] * inv_sqrt2;
        x[i] = 0.5f * x[i] * (1.0f + tm_erf_poly(u));
    }
}

/* ================= fused Q15 GELU via per-scale runtime LUT =================
 * GELU(x) needs x in REAL units (erf(x/sqrt2)), which a fixed LUT on the
 * Q15 index cannot know.  Instead we rebuild a tiny 513-entry LUT once per
 * GEMM call (i.e. per layer) from that layer's activation max: for entry k,
 * u = -1 + 2k/512 = x/amax, so gelu/amax = 0.5 u (1 + erf(u*amax/sqrt2)).
 * Build cost ~513 soft-float evals (well under a ms) and the LUT is exact
 * for the actual scale; applies in pure integer (7-bit linear interp, max
 * err <0.1 Q15 units).  Output stays on the input Q15 grid (|gelu|<=|x|).
 */
static int16_t g_gelu_lut[513];
void tm_gelu_q15_lut(int16_t* x, int n, float amax) {
    const float inv_sqrt2 = 0.70710677f;
    float amr2 = amax * inv_sqrt2;
    for (int k = 0; k <= 512; k++) {
        float u = -1.0f + 2.0f * (float)k * (1.0f / 512.0f);
        float p = tm_erf_poly(u * amr2);
        float g = 0.5f * u * (1.0f + p);          /* gelu / amax */
        g_gelu_lut[k] = (int16_t)(g * 32767.0f + (g >= 0.0f ? 0.5f : -0.5f));
    }
    for (int i = 0; i < n; i++) {
        int32_t p = (int32_t)x[i] + 32768;        /* 0..65535 */
        int32_t idx = p >> 7;                     /* 0..511 */
        int32_t off = p & 127;
        x[i] = (int16_t)((int32_t)g_gelu_lut[idx] +
                         ((((int32_t)g_gelu_lut[idx+1] - (int32_t)g_gelu_lut[idx]) * off + 64) >> 7));
    }
}

/* ================= elementwise add ================= */
void tm_add_inplace(const float* x, float* y, int n) {
    for (int i = 0; i < n; i++) y[i] += x[i];
}

/* ================= fast exp (valid y<=0) =================
 * e^y = 2^(n+f):  y*log2(e) = n+f, n = floor, f in [0,1).
 * 2^f via 5-term Taylor of exp(f*ln2), 2^n by bit-insert.
 */
float tm_exp_fast(float y) {
    if (y < TM_EXP_FAST_MIN) return 0.0f;
    static const float LOG2E = 1.4426950f;
    static const float C1 = 0.69314718f, C2 = 0.24022651f,
                       C3 = 0.055504110f, C4 = 0.0096181287f,
                       C5 = 0.0013333558f;
    float t = y * LOG2E;             /* <= 0 */
    int n = (int)t;                  /* trunc toward zero == ceil */
    if (t < (float)n) n -= 1;        /* exact floor */
    float f = t - (float)n;          /* [0,1) */
    float p = C5;
    p = p * f + C4;
    p = p * f + C3;
    p = p * f + C2;
    p = p * f + C1;
    p = p * f + 1.0f;                /* 2^f */
    /* 2^n via IEEE bit-insert (n in [-57,0] -> normal range) */
    union { uint32_t u; float f; } bi;
    bi.u = (uint32_t)(n + 127) << 23;
    return p * bi.f;
}

float tm_exp_f32(float y) { return expf(y); }
