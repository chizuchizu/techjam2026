/*
 * kernels.c - implementation of the low-level kernels.
 */
#define TM_PROFILE 1
#include "kernels.h"
#pragma GCC optimize ("O3")
#ifndef TM_IRAM
#define TM_IRAM
#endif

#include <math.h>
#include <string.h>
#include "gelu_tab_2049.h"

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
/* The default H=1 build aliases these workspaces onto model.c buffers to meet
 * its tight SRAM limit. The tiled WiFi build has no model.c buffers, so it
 * uses reduced standalone workspaces sized to TM_A16_ROWS. */
#ifdef TM_TILED_FORWARD
#ifndef TM_A16_ROWS
#define TM_A16_ROWS TM_S
#endif
static int16_t a16[TM_A16_ROWS * TM_D];
static int32_t tiled_s1[TM_S];
static int64_t tiled_s2[TM_S];
static int16_t tiled_mx[TM_S];
static int32_t tiled_mx15[TM_S];
static int32_t tiled_sh[TM_S];
static int32_t tiled_fk[TM_D];
static int32_t tiled_bk[TM_D];
#define s1_ tiled_s1
#define s2_ tiled_s2
#define mx_ tiled_mx
#define mx15_ tiled_mx15
#define sh_ tiled_sh
#define Fk_ tiled_fk
#define Bk_ tiled_bk
#else
extern int16_t g_qh[TM_S * TM_HD];
#define a16 g_qh

/* Fused-LayerNorm scratch overlays the head of model.c's g_buf2, which is
 * provably free during every norm call (bn/layernorm run between the GEMM
 * phases that own g_buf2).  Kernels keep their own .bss otherwise. */
extern float g_buf2[TM_S * TM_D];
#define s1_   ((int32_t *)(g_buf2))
#define s2_   ((int64_t *)((int32_t *)(g_buf2) + TM_S))
#define mx_   ((int16_t *)((int32_t *)(g_buf2) + 3 * TM_S))
#define mx15_ ((int32_t *)((int32_t *)(g_buf2) + 4 * TM_S))
#define sh_   ((int32_t *)((int32_t *)(g_buf2) + 5 * TM_S))
#define Fk_   ((int32_t *)((int32_t *)(g_buf2) + 6 * TM_S + TM_D))
#define Bk_   ((int32_t *)((int32_t *)(g_buf2) + 6 * TM_S + 2 * TM_D))
#endif

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
#include "esp_cpu.h"
#define KB_NOW() ((uint32_t)esp_timer_get_time())
#else
#include <time.h>
static __inline__ uint32_t tm_kb_now_host(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint32_t)((uint64_t)ts.tv_sec * 1000000u + ts.tv_nsec / 1000u);
}
#define KB_NOW() tm_kb_now_host()
#endif

/* RISC-V cycle counter to separate CPU-bound vs flash-XIP stalls
 * (esp_cpu_get_cycle_count reads the 'cycle' CSR; official ESP-IDF API). */
static __inline__ uint64_t tm_cyc_now(void) {
#if defined(__riscv)
    return (uint64_t)esp_cpu_get_ccount();
#else
    return 0;
#endif
}
static uint64_t g_c4_cyc = 0; static uint32_t g_c4_n = 0;
static uint64_t g_c5_cyc = 0; static uint32_t g_c5_n = 0;

#ifdef TM_RANGE
static double r_min=1e30f, r_max=-1e30f; static long r_n=0;
#define RTR(x) do { double v=(double)(x); if (v<r_min) r_min=v; if (v>r_max) r_max=v; r_n++; } while(0)
static double h_gx_min=1e30f, h_gx_max=-1e30f, h_bx_min=1e30f, h_bx_max=-1e30f;
static double h_bq_min=1e30f, h_bq_max=-1e30f, h_c_min=1e30f, h_c_max=-1e30f;
static double h_c5g_min=1e30f, h_c5g_max=-1e30f, h_c5bg_min=1e30f, h_c5bg_max=-1e30f;
#endif

/* kernels-internal microsecond instrumentation (independent of model.c slots) */
static uint32_t k_t0[4];
static uint64_t k_acc[4]; static uint32_t k_n[4];
#define KPB(i) do { k_t0[(i)] = KB_NOW(); } while (0)
#define KPE(i) do { uint32_t d = KB_NOW() - k_t0[(i)]; \
                    k_acc[(i)] += d; k_n[(i)]++; } while (0)
void tm_kbench_clear(void) { for (int i=0;i<4;i++){k_acc[i]=0;k_n[i]=0;} g_c4_cyc=0; g_c4_n=0; g_c5_cyc=0; g_c5_n=0; }
void tm_kbench_dump(void) {
    extern void tm_prof_emit(const char* s);
    char line[128];
    for (int i=0;i<4 && k_n[i];i++){
        (void)snprintf(line,sizeof line,"KB%d n=%u avg_us=%.1f tot_ms=%.1f\n",
            i, k_n[i], (double)k_acc[i]/(double)k_n[i],
            (double)k_acc[i]/1000.0);
        tm_prof_emit(line);
    }
    if (g_c4_n) {
        (void)snprintf(line,sizeof line,"C4CYC n=%u avg_cyc=%.1f avg_us=%.1f\n",
            g_c4_n, (double)g_c4_cyc/(double)g_c4_n,
            (double)g_c4_cyc/(double)g_c4_n/160.0);
        tm_prof_emit(line);
    }
    if (g_c5_n) {
        (void)snprintf(line,sizeof line,"C5CYC n=%u avg_cyc=%.1f avg_us=%.1f\n",
            g_c5_n, (double)g_c5_cyc/(double)g_c5_n,
            (double)g_c5_cyc/(double)g_c5_n/160.0);
        tm_prof_emit(line);
    }
}
#else
#define KPB(i) do {} while (0)
#define KPE(i) do {} while (0)
#endif


/* Quantize A into a caller buffer (Q15 fast-round, no libm). sa precomputed. */
/* Amax scan of A; returns the Q15 scale sa = TM_QACT_MAX/amax. */
/* integer f32 -> q15 with runtime scale sa = sc_m * 2^sc_e (exact 48-bit
 * product, single rounding at the shift): no soft-float per element. */
static inline int32_t tm_f2q15(uint32_t b, int32_t sc_m, int sc_e, int32_t Q) {
    if ((b & 0x7f800000u) == 0) return 0;                 /* zero / subnormal */
    if ((b & 0x7f800000u) == 0x7f800000u)                 /* inf/nan guard */
        return (b >> 31) ? -Q : Q;
    int sh = (int)((b >> 23) & 0xffu) - 150 + sc_e;
    uint64_t P = (uint64_t)((b & 0x7fffffu) | 0x800000u) * (uint64_t)(uint32_t)sc_m;
    int64_t pr;
    if (sh >= 0) { pr = (int64_t)(P << sh); }
    else { int rs = -sh; pr = (int64_t)((P + (1ULL << (rs - 1))) >> rs); }
    if (pr > Q) pr = Q;
    int32_t q = (int32_t)pr;
    return (b >> 31) ? -q : q;
}
/* split float scale sa into (sc_m, sc_e) with sc = sc_m * 2^sc_e, sc_m in [2^23,2^24) */
static inline void tm_split_scale(float sa, int32_t* sc_m, int* sc_e) {
    uint32_t sbb; memcpy(&sbb, &sa, 4);
    int e = (int)((sbb >> 23) & 0xffu);
    *sc_e = e - 150;
    *sc_m = (int32_t)((sbb & 0x7fffffu) | 0x800000u);
    if (e == 0) { *sc_m = 0; *sc_e = 0; }
}

float tm_gemm_amax(const float* A, int n) {
    const uint32_t* ai = (const uint32_t *)(const void *)A;
    uint32_t am = 0;
    for (int i = 0; i < n; i++) {
        uint32_t u = ai[i] & 0x7fffffffu;
        if (u > am) am = u;
    }
    if (am == 0) am = 0x3f800000u;               /* amax = 1.0f */
    return (TM_QACT_MAX / *((const float*)&am));
}

void tm_gemm_quantA_into(const float* A, int n, int16_t* out, float sa) {
    int32_t sc_m; int sc_e;
    tm_split_scale(sa, &sc_m, &sc_e);
    const uint32_t* ai = (const uint32_t *)(const void *)A;
    for (int i = 0; i < n; i++) {
        out[i] = (int16_t)tm_f2q15(ai[i], sc_m, sc_e, (int32_t)TM_QACT_MAX);
    }
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
    return tm_gemm_head_q15_m(Aq, sa_inv, Wq, w_scale, bias, acc, dst,
                              TM_S, K);
}

/* Row-count-parameterised variant used by the sequential tile schedule. */
float tm_gemm_head_q15_m(const int16_t* Aq, float sa_inv, const int16_t* Wq,
                         float w_scale, const float* bias,
                         int32_t* acc, int16_t* dst, int M, int K) {
    KPB(0);
    const float g = sa_inv * w_scale;
    const int S = M, HD = TM_HD;
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
#if defined(__riscv)
            if ((K & 1) == 0 && K >= 4 && K >= 128) {  /* asm strides assume K==128 */
                /* 2-pair (16-MAC) unroll with software-pipelined flash-XIP
                 * weight prefetch.  a4/a5 hold w[k], w[k+1]; the next pair
                 * w[k+2], w[k+3] is prefetched at the END of each iteration
                 * (a4/a5 are dead by then), giving each weight load ~36
                 * instructions of lead time so flash-XIP latency is fully
                 * hidden.  The main loop runs pairs k = 0..K-4 only; the
                 * final pair (k = K-2, K-1) is a straight-line tail block
                 * using the already-prefetched weights.  No reads past the
                 * weight column; same accumulation order/data types as the
                 * C fallback -> bit-exact. */
                const int16_t* w = wr + 2;
                const int16_t* whend = wr + K;
                const int16_t* rb = a0;
                int32_t w0 = (int32_t)wr[0], w1 = (int32_t)wr[1];
                __asm__ __volatile__(
                    ".p2align 4\n"
                    "mv  a4, %[w0]\n"
                    "mv  a5, %[w1]\n"
                    "1:\n"
                    /* block 1: 8 rows x w[k] */
                    "lh  t0, 0(%[rb])\n"
                    "lh  t1, 256(%[rb])\n"
                    "lh  t2, 512(%[rb])\n"
                    "lh  t3, 768(%[rb])\n"
                    "lh  t4, 1024(%[rb])\n"
                    "lh  t5, 1280(%[rb])\n"
                    "lh  t6, 1536(%[rb])\n"
                    "lh  a0, 1792(%[rb])\n"
                    "mul  t0, t0, a4\n"
                    "mul  t1, t1, a4\n"
                    "mul  t2, t2, a4\n"
                    "mul  t3, t3, a4\n"
                    "mul  t4, t4, a4\n"
                    "mul  t5, t5, a4\n"
                    "mul  t6, t6, a4\n"
                    "mul  a0, a0, a4\n"
                    "add  %[c0], %[c0], t0\n"
                    "add  %[c1], %[c1], t1\n"
                    "add  %[c2], %[c2], t2\n"
                    "add  %[c3], %[c3], t3\n"
                    "add  %[c4], %[c4], t4\n"
                    "add  %[c5], %[c5], t5\n"
                    "add  %[c6], %[c6], t6\n"
                    "add  %[c7], %[c7], a0\n"
                    /* prefetch w[k+2] (a4 free after block-1 muls) */
                    "lh  a4, 0(%[wpt])\n"
                    /* block 2: 8 rows x w[k+1] at +2 */
                    "lh  t0, 2(%[rb])\n"
                    "lh  t1, 258(%[rb])\n"
                    "lh  t2, 514(%[rb])\n"
                    "lh  t3, 770(%[rb])\n"
                    "lh  t4, 1026(%[rb])\n"
                    "lh  t5, 1282(%[rb])\n"
                    "lh  t6, 1538(%[rb])\n"
                    "lh  a0, 1794(%[rb])\n"
                    "mul  t0, t0, a5\n"
                    "mul  t1, t1, a5\n"
                    "mul  t2, t2, a5\n"
                    "mul  t3, t3, a5\n"
                    "mul  t4, t4, a5\n"
                    "mul  t5, t5, a5\n"
                    "mul  t6, t6, a5\n"
                    "mul  a0, a0, a5\n"
                    "add  %[c0], %[c0], t0\n"
                    "add  %[c1], %[c1], t1\n"
                    "add  %[c2], %[c2], t2\n"
                    "add  %[c3], %[c3], t3\n"
                    "add  %[c4], %[c4], t4\n"
                    "add  %[c5], %[c5], t5\n"
                    "add  %[c6], %[c6], t6\n"
                    "add  %[c7], %[c7], a0\n"
                    /* prefetch w[k+3] (a5 free) + advance pointers */
                    "lh  a5, 2(%[wpt])\n"
                    "addi %[wpt], %[wpt], 4\n"
                    "addi %[rb], %[rb], 4\n"
                    "bne  %[wpt], %[wend], 1b\n"
                    /* tail: last pair k = K-2, K-1 (a4/a5 already hold it) */
                    "lh  t0, 0(%[rb])\n"
                    "lh  t1, 256(%[rb])\n"
                    "lh  t2, 512(%[rb])\n"
                    "lh  t3, 768(%[rb])\n"
                    "lh  t4, 1024(%[rb])\n"
                    "lh  t5, 1280(%[rb])\n"
                    "lh  t6, 1536(%[rb])\n"
                    "lh  a0, 1792(%[rb])\n"
                    "mul  t0, t0, a4\n"
                    "mul  t1, t1, a4\n"
                    "mul  t2, t2, a4\n"
                    "mul  t3, t3, a4\n"
                    "mul  t4, t4, a4\n"
                    "mul  t5, t5, a4\n"
                    "mul  t6, t6, a4\n"
                    "mul  a0, a0, a4\n"
                    "add  %[c0], %[c0], t0\n"
                    "add  %[c1], %[c1], t1\n"
                    "add  %[c2], %[c2], t2\n"
                    "add  %[c3], %[c3], t3\n"
                    "add  %[c4], %[c4], t4\n"
                    "add  %[c5], %[c5], t5\n"
                    "add  %[c6], %[c6], t6\n"
                    "add  %[c7], %[c7], a0\n"
                    "lh  t0, 2(%[rb])\n"
                    "lh  t1, 258(%[rb])\n"
                    "lh  t2, 514(%[rb])\n"
                    "lh  t3, 770(%[rb])\n"
                    "lh  t4, 1026(%[rb])\n"
                    "lh  t5, 1282(%[rb])\n"
                    "lh  t6, 1538(%[rb])\n"
                    "lh  a0, 1794(%[rb])\n"
                    "mul  t0, t0, a5\n"
                    "mul  t1, t1, a5\n"
                    "mul  t2, t2, a5\n"
                    "mul  t3, t3, a5\n"
                    "mul  t4, t4, a5\n"
                    "mul  t5, t5, a5\n"
                    "mul  t6, t6, a5\n"
                    "mul  a0, a0, a5\n"
                    "add  %[c0], %[c0], t0\n"
                    "add  %[c1], %[c1], t1\n"
                    "add  %[c2], %[c2], t2\n"
                    "add  %[c3], %[c3], t3\n"
                    "add  %[c4], %[c4], t4\n"
                    "add  %[c5], %[c5], t5\n"
                    "add  %[c6], %[c6], t6\n"
                    "add  %[c7], %[c7], a0\n"
                    : [c0] "+r"(c0), [c1] "+r"(c1), [c2] "+r"(c2),
                      [c3] "+r"(c3), [c4] "+r"(c4), [c5] "+r"(c5),
                      [c6] "+r"(c6), [c7] "+r"(c7),
                      [wpt] "+r"(w), [rb] "+r"(rb)
                    : [w0] "r"(w0), [w1] "r"(w1), [wend] "r"(whend)
                    : "a0", "a4", "a5", "t0", "t1", "t2", "t3", "t4", "t5", "t6");
            } else
#endif
            {
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
    /* Bias fold in Q15-input units: BD[d] = round(bias[d]/g).  BX[d] =
     * bias[d]*sa*2^30 grows to ~1.6e12 (int32 overflow) so fold the bias into
     * the multiplier; fold error |BD*g - bias| <= g/2 lands far below the
     * >>30 output bin (<= GX/2 = ~5e4 / 2^30 = 5e-5 q15 units).  GX = g*sa*2^30
     * stays in [7.3e4, 1.04e5] on real QKV head scales. */
    KPB(1);
    const float gsa = g * sa;
    const int32_t GX = (int32_t)(gsa * 1073741824.0f);
    const float ginv = (g != 0.0f) ? (1.0f / g) : 0.0f;
    int32_t BD[TM_HD];
    for (int d = 0; d < HD; d++)
        BD[d] = (int32_t)(bias[d] * ginv + (bias[d] >= 0.0f ? 0.5f : -0.5f));
    for (int i = 0; i < S; i++) {
        const int32_t* ra = acc + (size_t)i * HD;
        int16_t* dd = dst + (size_t)i * HD;
        for (int d = 0; d < HD; d++) {
#if defined(__riscv)
            int32_t r = ra[d];
#ifdef TM_RANGE
            if (r < h_c_min) h_c_min = r; if (r > h_c_max) h_c_max = r;
#endif
            int32_t w = r + BD[d];
            int32_t q, lo, hi, sn, c2, lo2;
            __asm__ volatile(
                "mul  %[lo], %[w], %[gx]\n\t"   /* lo = w*GX (low 32) */
                "mulh %[hi], %[w], %[gx]\n\t"   /* hi = w*GX (high 32) */
                "srai %[sn], %[hi], 31\n\t"     /* sign of t (64-bit) */
                "slli %[c2], %[sn], 29\n\t"     /* +sign*2^29 low */
                "add  %[lo], %[lo], %[c2]\n\t"
                "sltu %[c2], %[lo], %[c2]\n\t"  /* carry into hi */
                "add  %[hi], %[hi], %[c2]\n\t"
                "slli %[sn], %[hi], 2\n\t"      /* (hi:lo) >> 30 arithmetic low32 */
                "srl  %[lo2], %[lo], 30\n\t"
                "or   %[q], %[sn], %[lo2]\n\t"
                "li   %[sn], 32767\n\t"
                "blt  %[sn], %[q], 2f\n\t"
                "li   %[sn], -32767\n\t"
                "blt  %[q], %[sn], 2f\n\t"
                "j    3f\n\t"
                "2:   mv   %[q], %[sn]\n\t"
                "3:\n\t"
                : [q] "=&r"(q), [lo] "=&r"(lo), [hi] "=&r"(hi),
                  [sn] "=&r"(sn), [c2] "=&r"(c2), [lo2] "=&r"(lo2)
                : [w] "r"(w), [gx] "r"(GX)
                : "memory");
            dd[d] = (int16_t)q;
#else
            int32_t r = ra[d];
#ifdef TM_RANGE
            if (r < h_c_min) h_c_min = r; if (r > h_c_max) h_c_max = r;
#endif
            int64_t t = ((int64_t)r + (int64_t)BD[d]) * (int64_t)GX;
            int64_t tt = (t >= 0) ? (t + 536870912) : (-t + 536870912);
            int64_t qq = (t >= 0) ? tt : -tt;
            int32_t q = (int32_t)(qq >> 30);
            if (q > TM_QACT_MAX) q = (int)TM_QACT_MAX;
            else if (q < (int)-TM_QACT_MAX) q = -(int)TM_QACT_MAX;
            dd[d] = (int16_t)q;
#endif
        }
    }
    KPE(1);
    return amax / TM_QACT_MAX;
}



#ifdef TM_RANGE
void tm_range_dump(void) {
    fprintf(stderr, "RNG n=%ld\n", r_n);
    fprintf(stderr, "RNG GX=[%.1f,%.1f] BX=[%.1f,%.1f] (head_q15)\n", h_gx_min,h_gx_max,h_bx_min,h_bx_max);
    fprintf(stderr, "RNG bq=[%.1f,%.1f] r=[%.1f,%.1f] (c5q15/head)\n", h_bq_min,h_bq_max,h_c_min,h_c_max);
    fprintf(stderr, "RNG c5g=[%.6f,%.6f] BQ=[%.1f,%.1f] (core5)\n", h_c5g_min,h_c5g_max,h_c5bg_min,h_c5bg_max);
}
#endif

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
    uint64_t _c4s = tm_cyc_now();
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
    g_c4_cyc += (uint64_t)(tm_cyc_now() - _c4s); g_c4_n++;
}


/* core5: oproj/FFN1/FFN2 GEMM - j-tile-2 x IBLK=4 x K-pair, 8 int32 accs and
 * ~20 live registers (vs core4_v2's 36+ which spill on the in-order RV32I).
 * Bit-exact vs core4_v2 (identical product order); fp32 epilogue. */
void tm_gemm_core5(const int16_t* Aq, float sa_inv, const int16_t* Wq,
                   float w_scale, const float* bias, float* C,
                   int M, int K, int N, int rowStride) {
    uint64_t _c5s = tm_cyc_now();
    const float g = sa_inv * w_scale;
    int32_t BQ[1024] __attribute__((aligned(4)));
/* ranges via file-scope h_c5* */
    const float ginv5 = (g != 0.0f) ? (1.0f / g) : 0.0f;
    for (int jj = 0; jj < N && jj < 1024; jj++) {
        float bv = bias ? bias[jj] : 0.0f;
        float bg = bv * ginv5;
        BQ[jj] = (int32_t)(bg + (bg >= 0.0f ? 0.5f : -0.5f));
#ifdef TM_RANGE
        if (g < h_c5g_min) h_c5g_min = g; if (g > h_c5g_max) h_c5g_max = g;
        if (bg < h_c5bg_min) h_c5bg_min = bg; if (bg > h_c5bg_max) h_c5bg_max = bg;
#endif
    }
    for (int j = 0; j + 1 < N; j += 2) {
        const int16_t* w0 = Wq + (size_t)j * K;
        const int16_t* w1 = Wq + (size_t)(j + 1) * K;
        const float bj  = bias ? bias[j]   : 0.0f;
        const float bj1 = bias ? bias[j+1] : 0.0f;
        for (int it = 0; it < M; it += 4) {
            const int16_t* a0 = Aq + (size_t)(it + 0) * K;
            const int16_t* a1 = Aq + (size_t)(it + 1) * K;
            const int16_t* a2 = Aq + (size_t)(it + 2) * K;
            const int16_t* a3 = Aq + (size_t)(it + 3) * K;
            int32_t c00=0,c10=0,c20=0,c30=0, c01=0,c11=0,c21=0,c31=0;
            int k = 0;
            for (; k + 1 < K; k += 2) {
                int32_t b0 = w0[k], b1 = w1[k], b0n = w0[k+1], b1n = w1[k+1];
                int32_t v00=a0[k],   v10=a1[k],   v20=a2[k],   v30=a3[k];
                int32_t v01=a0[k+1], v11=a1[k+1], v21=a2[k+1], v31=a3[k+1];
                c00 += v00*b0; c10 += v10*b0; c20 += v20*b0; c30 += v30*b0;
                c01 += v00*b1; c11 += v10*b1; c21 += v20*b1; c31 += v30*b1;
                c00 += v01*b0n; c10 += v11*b0n; c20 += v21*b0n; c30 += v31*b0n;
                c01 += v01*b1n; c11 += v11*b1n; c21 += v21*b1n; c31 += v31*b1n;
            }
            for (; k < K; k++) {
                int32_t b0 = w0[k], b1 = w1[k];
                int32_t v00=a0[k], v10=a1[k], v20=a2[k], v30=a3[k];
                c00 += v00*b0; c10 += v10*b0; c20 += v20*b0; c30 += v30*b0;
                c01 += v00*b1; c11 += v10*b1; c21 += v20*b1; c31 += v30*b1;
            }


            float* c0r = C + (size_t)it * rowStride + j;
            const int32_t bq0 = BQ[j], bq1 = BQ[j+1];
            c0r[0]              = (float)(c00 + bq0) * g;
            c0r[1]              = (float)(c01 + bq1) * g;
            c0r[rowStride]      = (float)(c10 + bq0) * g;
            c0r[rowStride+1]    = (float)(c11 + bq1) * g;
            c0r[2*rowStride]    = (float)(c20 + bq0) * g;
            c0r[2*rowStride+1]  = (float)(c21 + bq1) * g;
            c0r[3*rowStride]    = (float)(c30 + bq0) * g;
            c0r[3*rowStride+1]  = (float)(c31 + bq1) * g;
        }
    }
    if (N & 1) {
        int j = N - 1;
        const int16_t* wr = Wq + (size_t)j * K;
        const float bj = bias ? bias[j] : 0.0f;
        for (int it = 0; it < M; it += 4) {
            const int16_t* a0 = Aq + (size_t)(it+0) * K;
            const int16_t* a1 = Aq + (size_t)(it+1) * K;
            const int16_t* a2 = Aq + (size_t)(it+2) * K;
            const int16_t* a3 = Aq + (size_t)(it+3) * K;
            int32_t c0=0,c1=0,c2=0,c3=0;
            for (int k = 0; k < K; k++) {
                int32_t b = wr[k];
                c0 += (int32_t)a0[k]*b; c1 += (int32_t)a1[k]*b;
                c2 += (int32_t)a2[k]*b; c3 += (int32_t)a3[k]*b;
            }
            float* c0r = C + (size_t)it * rowStride + j;
            const int32_t bq = BQ[j];
            c0r[0] = (float)(c0 + bq) * g;
            c0r[rowStride]   = (float)(c1 + bq) * g;
            c0r[2*rowStride] = (float)(c2 + bq) * g;
            c0r[3*rowStride] = (float)(c3 + bq) * g;
        }
    }
    g_c5_cyc += (uint64_t)(tm_cyc_now() - _c5s); g_c5_n++;
}

/* core5 variant writing Q15 directly (fixed-point epilogue):
 * out_q15 = round((acc + bq_j) * QMAX / amax), bq_j = round(bias[j]/g),
 * amax = max|acc + bq_j| (int, bias folded).  Stored in place into Out[]
 * as int32 whose low half holds the q15 (caller uses (int16_t*)Out).
 * Returns sa2 = QMAX / (amax*g) -- the inverse of the float output
 * magnitude, matching tm_gemm_amax's convention (caller uses QMAX/sa2 as
 * the true max for gelu and 1/sa2 as the q15 input scale for FFN2). */
float tm_gemm_core5_q15(const int16_t* Aq, float sa_inv, const int16_t* Wq,
                        float w_scale, const float* bias, int32_t* scratch,
                        int16_t* Out, int M, int K, int N, int rowStride) {
    const float g = sa_inv * w_scale;
    const float ginv = (g != 0.0f) ? (1.0f / g) : 0.0f;
    int32_t amax = 1;
    for (int j = 0; j + 1 < N; j += 2) {
        const int16_t* w0 = Wq + (size_t)j * K;
        const int16_t* w1 = Wq + (size_t)(j + 1) * K;
        int32_t bq0 = 0, bq1 = 0;
        if (bias) {
            bq0 = (int32_t)(bias[j]   * ginv + (bias[j]   >= 0.0f ? 0.5f : -0.5f));
            bq1 = (int32_t)(bias[j+1] * ginv + (bias[j+1] >= 0.0f ? 0.5f : -0.5f));
#ifdef TM_RANGE
        { double v0=bq0,v1=bq1; if(v0<h_bq_min)h_bq_min=v0; if(v0>h_bq_max)h_bq_max=v0; if(v1<h_bq_min)h_bq_min=v1; if(v1>h_bq_max)h_bq_max=v1; }
#endif
        }
        for (int it = 0; it < M; it += 4) {
            const int16_t* a0 = Aq + (size_t)(it + 0) * K;
            const int16_t* a1 = Aq + (size_t)(it + 1) * K;
            const int16_t* a2 = Aq + (size_t)(it + 2) * K;
            const int16_t* a3 = Aq + (size_t)(it + 3) * K;
            int32_t c00=0,c10=0,c20=0,c30=0, c01=0,c11=0,c21=0,c31=0;
#if defined(__riscv)
            if ((K & 1) == 0 && K >= 128) {  /* asm assumes K==128 row strides */
                const int16_t* w_ = w0;
                const int16_t* whend_ = w0 + K;
                const int16_t* rb_ = a0;
                __asm__ __volatile__(
                    ".p2align 4\n"
                    "1:\n"
                    "lh  a5, 0(%[wpt])\n"
                    "lh  a6, 256(%[wpt])\n"
                    "lh  t0, 0(%[rb])\n"
                    "lh  t1, 256(%[rb])\n"
                    "lh  t2, 512(%[rb])\n"
                    "lh  t3, 768(%[rb])\n"
                    "mul  t4, t0, a5\n"
                    "mul  t5, t1, a5\n"
                    "mul  t6, t2, a5\n"
                    "mul  a0, t3, a5\n"
                    "mul  a1, t0, a6\n"
                    "mul  a2, t1, a6\n"
                    "mul  a3, t2, a6\n"
                    "mul  a4, t3, a6\n"
                    "add  %[c0], %[c0], t4\n"
                    "add  %[c1], %[c1], t5\n"
                    "add  %[c2], %[c2], t6\n"
                    "add  %[c3], %[c3], a0\n"
                    "add  %[c4], %[c4], a1\n"
                    "add  %[c5], %[c5], a2\n"
                    "add  %[c6], %[c6], a3\n"
                    "add  %[c7], %[c7], a4\n"
                    "addi %[wpt], %[wpt], 2\n"
                    "addi %[rb], %[rb], 2\n"
                    "bne  %[wpt], %[wend], 1b\n"
                    : [c0] "+r"(c00), [c1] "+r"(c10), [c2] "+r"(c20), [c3] "+r"(c30),
                      [c4] "+r"(c01), [c5] "+r"(c11), [c6] "+r"(c21), [c7] "+r"(c31),
                      [wpt] "+r"(w_), [rb] "+r"(rb_)
                    : [wend] "r"(whend_)
                    : "a0", "a1", "a2", "a3", "a4", "a5", "a6",
                      "t0", "t1", "t2", "t3", "t4", "t5", "t6");
            } else
#endif
            {
                int k = 0;
                for (; k + 1 < K; k += 2) {
                    int32_t b0 = w0[k], b1 = w1[k], b0n = w0[k+1], b1n = w1[k+1];
                    int32_t v00=a0[k],   v10=a1[k],   v20=a2[k],   v30=a3[k];
                    int32_t v01=a0[k+1], v11=a1[k+1], v21=a2[k+1], v31=a3[k+1];
                    c00 += v00*b0; c10 += v10*b0; c20 += v20*b0; c30 += v30*b0;
                    c01 += v00*b1; c11 += v10*b1; c21 += v20*b1; c31 += v30*b1;
                    c00 += v01*b0n; c10 += v11*b0n; c20 += v21*b0n; c30 += v31*b0n;
                    c01 += v01*b1n; c11 += v11*b1n; c21 += v21*b1n; c31 += v31*b1n;
                }
                for (; k < K; k++) {
                    int32_t b0 = w0[k], b1 = w1[k];
                    int32_t v00=a0[k], v10=a1[k], v20=a2[k], v30=a3[k];
                    c00 += v00*b0; c10 += v10*b0; c20 += v20*b0; c30 += v30*b0;
                    c01 += v00*b1; c11 += v10*b1; c21 += v20*b1; c31 += v30*b1;
                }
            }
            int32_t v00 = c00 + bq0, v10 = c10 + bq0, v20 = c20 + bq0, v30 = c30 + bq0;
            int32_t v01 = c01 + bq1, v11 = c11 + bq1, v21 = c21 + bq1, v31 = c31 + bq1;
            int32_t* orow = scratch + (size_t)it * rowStride + j;
            orow[0] = v00; orow[1] = v01;
            orow[rowStride]   = v10; orow[rowStride+1]   = v11;
            orow[2*rowStride] = v20; orow[2*rowStride+1] = v21;
            orow[3*rowStride] = v30; orow[3*rowStride+1] = v31;
            int32_t a;
            a = v00<0?-v00:v00; if (a>amax) amax=a;
            a = v10<0?-v10:v10; if (a>amax) amax=a;
            a = v20<0?-v20:v20; if (a>amax) amax=a;
            a = v30<0?-v30:v30; if (a>amax) amax=a;
            a = v01<0?-v01:v01; if (a>amax) amax=a;
            a = v11<0?-v11:v11; if (a>amax) amax=a;
            a = v21<0?-v21:v21; if (a>amax) amax=a;
            a = v31<0?-v31:v31; if (a>amax) amax=a;
        }
    }
    if (N & 1) {
        int j = N - 1;
        const int16_t* wr = Wq + (size_t)j * K;
        int32_t bq0 = 0;
        if (bias) bq0 = (int32_t)(bias[j] * ginv + (bias[j] >= 0.0f ? 0.5f : -0.5f));
        for (int it = 0; it < M; it += 4) {
            const int16_t* a0 = Aq + (size_t)(it+0) * K;
            const int16_t* a1 = Aq + (size_t)(it+1) * K;
            const int16_t* a2 = Aq + (size_t)(it+2) * K;
            const int16_t* a3 = Aq + (size_t)(it+3) * K;
            int32_t c0=0,c1=0,c2=0,c3=0;
            for (int k = 0; k < K; k++) {
                int32_t b = wr[k];
                c0 += (int32_t)a0[k]*b; c1 += (int32_t)a1[k]*b;
                c2 += (int32_t)a2[k]*b; c3 += (int32_t)a3[k]*b;
            }
            int32_t v0 = c0 + bq0, v1 = c1 + bq0, v2 = c2 + bq0, v3 = c3 + bq0;
            int32_t* orow = scratch + (size_t)it * rowStride + j;
            orow[0] = v0; orow[rowStride] = v1;
            orow[2*rowStride] = v2; orow[3*rowStride] = v3;
            int32_t aa;
            aa = v0<0?-v0:v0; if (aa>amax) amax=aa;
            aa = v1<0?-v1:v1; if (aa>amax) amax=aa;
            aa = v2<0?-v2:v2; if (aa>amax) amax=aa;
            aa = v3<0?-v3:v3; if (aa>amax) amax=aa;
        }
    }
    /* pass 2: fixed-point convert -> q15 = round(x * QMAX / amax).
     * sa: drop low bits so divisor fits 20 bits (max precision with int64). */
    int sa = 0; int64_t amr = amax;
    while (amr > ((int64_t)1 << 20)) { amr >>= 1; sa++; }
    const int64_t SC = (int64_t)(0.5 + (double)TM_QACT_MAX * (double)((int64_t)1 << 20) / (double)amr);
    const int32_t R2 = 1 << 19;
    const int32_t QQ = (int32_t)TM_QACT_MAX;
    int n = M * N;
    for (int i = 0; i < n; i++) {
        int64_t x = scratch[i] >> sa;       /* arithmetic shift */
        int64_t q = (x * SC + R2) >> 20;
        if (q > QQ) q = QQ;
        if (q < -QQ) q = -QQ;
        Out[i] = (int16_t)q;
    }
    return TM_QACT_MAX / ((float)amax * g);
}

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
    /* ---- integer amax scan: for non-negative finite floats the IEEE bit
     * pattern lexicographic order == float order, so |x| via one AND and an
     * integer compare; exact float amax is decoded from the max bits. ---- */
    const uint32_t* srci = (const uint32_t *)(const void *)in;
    const int32_t N = S * D;
    uint32_t am = 0;
    for (int32_t i = 0; i < N; i++) {
        uint32_t u = srci[i] & 0x7fffffffu;
        if (u > am) am = u;
    }
    if (am == 0) am = 0x3f800000u;               /* amax = 1.0f */
    float amax;
    { uint32_t a32 = am; memcpy(&amax, &a32, 4); }
    /* split sc = Q/amax into exact 24-bit mantissa * 2^exp  (sc_m, sc_e):
     * x*sc = (x_mant * sc_m) * 2^(x_exp-150+sc_e), a 48-bit exact product,
     * rounded once at the shift -- no soft-float muls/fcvt per element. */
    float sc = ((float)Q) / amax;
    uint32_t sbb; memcpy(&sbb, &sc, 4);
    int sc_e = (int)((sbb >> 23) & 0xffu) - 150;      /* sc = sc_m * 2^sc_e, sc_m in [2^23,2^24) */
    int32_t sc_m = (int32_t)((sbb & 0x7fffffu) | 0x800000u);
    if (((sbb >> 23) & 0xffu) == 0) { sc_m = 0; sc_e = 0; }  /* subnormal sc: x*sc rounds to 0 */

    const float sxi = 1.0f / sc;
    const float sxi2 = sxi * sxi;
    /* pass 1: quantize x once (integer bit-trick conversion), accumulate
     * integer row stats. */
        float bmax = 0.0f;
    for (int32_t i = 0; i < S; i++) {
        int32_t s1 = 0; int64_t s2 = 0; int32_t mx = 0;
        const uint32_t* rowbits = srci + (size_t)i * D;
        int16_t* dst = out_q + (size_t)i * D;
        for (int32_t k = 0; k < D; k++) {
            uint32_t b = rowbits[k];
            int32_t q;
            if ((b & 0x7f800000u) == 0) {            /* zero / subnormal */
                q = 0;
            } else if ((b & 0x7f800000u) == 0x7f800000u) { /* inf/nan guard */
                q = (b >> 31) ? -Q : Q;
            } else {
                int sh = (int)((b >> 23) & 0xffu) - 150 + sc_e;
                uint64_t P = (uint64_t)((b & 0x7fffffu) | 0x800000u) * (uint64_t)(uint32_t)sc_m;
                int64_t pr;
                if (sh >= 0) {
                    pr = (int64_t)(P << sh);
                } else {
                    int rs = -sh;
                    pr = (int64_t)((P + (1ULL << (rs - 1))) >> rs);
                }
                if (pr > Q) pr = Q;
                q = ((int32_t)pr);
                if (b >> 31) q = -q;
            }
            int32_t a = q < 0 ? -q : q;
            if (a > mx) mx = a;
            s1 += q;
            s2 += (int64_t)q * (int64_t)q;
            dst[k] = (int16_t)q;
        }
        s1_[i] = s1; s2_[i] = s2; mx_[i] = (int16_t)mx;
    }    /* per row: mean/var/rstd (fp32, S rows only) + amax bound */
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

/* ================= R1: integer-residual kernels (FAST) =================
 * The FAST layer residual ("g_x") is carried as INT32 at a fixed scale sx
 * (real value = x * sx; LSB = span/2^31, i.e. numerically exact for our
 * magnitudes).  This removes, per element: the oproj/f2 float epilogues
 * (268 instr -> ~10 fixed-point), the fp32 res1/res2 passes (271
 * instr/elem each), and the float amax-scan + float->q15 quantize the
 * per-layer norms used to do (the norm now stats the int32 residual
 * directly). EXACT path and tm_layernorm are untouched.
 */

/* quantize float*in -> int32 residual at scale sx (single-rounding bit trick). */
void tm_quant_res_i32(const float* in, int32_t* out_q, int n, float sx) {
    const float sc = (sx > 0.0f) ? (1.0f / sx) : 0.0f;
    uint32_t sbb; memcpy(&sbb, &sc, 4);
    int sc_e = (int)((sbb >> 23) & 0xffu) - 150;
    int32_t sc_m = (int32_t)((sbb & 0x7fffffu) | 0x800000u);
    if (((sbb >> 23) & 0xffu) == 0) { sc_m = 0; sc_e = 0; }
    const uint32_t* bits = (const uint32_t*)(const void*)in;
    for (int i = 0; i < n; i++) {
        uint32_t b = bits[i];
        int64_t q;
        if ((b & 0x7f800000u) == 0) {
            q = 0;
        } else if ((b & 0x7f800000u) == 0x7f800000u) {
            q = (b >> 31) ? INT32_MIN : INT32_MAX;
        } else {
            int sh = (int)((b >> 23) & 0xffu) - 150 + sc_e;
            uint64_t P = (uint64_t)((b & 0x7fffffu) | 0x800000u) * (uint64_t)(uint32_t)sc_m;
            if (sh >= 0) {
                q = (int64_t)(P << sh);
            } else {
                int rs = -sh;
                q = (int64_t)((P + (1ULL << (rs - 1))) >> rs);
            }
            if (b >> 31) q = -q;
        }
        if (q > INT32_MAX) q = INT32_MAX; else if (q < INT32_MIN) q = INT32_MIN;
        out_q[i] = (int32_t)q;
    }
}

/* core5 GEMM whose epilogue FUSES into the int32 residual xq at scale sx:
 *   xq[i*rowStride + j] += (acc + bq_j) * g / sx
 * where g = sa_inv * w_scale (same per-call factor as tm_gemm_core5) and
 * bq_j = round(bias[j]/g).  The fixed multiply uses r = g/sx as an exact
 * 24-bit-mantissa pair (r_m, re), so each output costs one 32x32->64 mult
 * + shifts (mulh on RV32) instead of two soft-float libcalls. */
void tm_gemm_core5_resid(const int16_t* Aq, float sa_inv, const int16_t* Wq,
                         float w_scale, const float* bias, int32_t* xq,
                         float sx, int M, int K, int N, int rowStride) {
    const float g = sa_inv * w_scale;
    float r = (g != 0.0f && sx > 0.0f) ? (g / sx) : 0.0f;
    if (r < 0.0f) r = 0.0f;
    uint32_t rb; memcpy(&rb, &r, 4);
    int re = (int)((rb >> 23) & 0xffu) - 150;
    int32_t r_m = (int32_t)((rb & 0x7fffffu) | 0x800000u);
    if (((rb >> 23) & 0xffu) == 0 || !(r > 0.0f)) { r_m = 0; re = 0; }
    int32_t BQ[1024] __attribute__((aligned(4)));
    const float ginv = (g != 0.0f) ? (1.0f / g) : 0.0f;
    for (int jj = 0; jj < N && jj < 1024; jj++) {
        float bg = bias ? bias[jj] * ginv : 0.0f;
        BQ[jj] = (int32_t)(bg + (bg >= 0.0f ? 0.5f : -0.5f));
    }
    for (int j = 0; j + 1 < N; j += 2) {
        const int16_t* w0 = Wq + (size_t)j * K;
        const int16_t* w1 = Wq + (size_t)(j + 1) * K;
        for (int it = 0; it < M; it += 4) {
            const int16_t* a0 = Aq + (size_t)(it + 0) * K;
            const int16_t* a1 = Aq + (size_t)(it + 1) * K;
            const int16_t* a2 = Aq + (size_t)(it + 2) * K;
            const int16_t* a3 = Aq + (size_t)(it + 3) * K;
            int32_t c00=0,c10=0,c20=0,c30=0, c01=0,c11=0,c21=0,c31=0;
#if defined(__riscv)
            if (((K) & 1) == 0 && (K) >= 4 && (K) >= 128) {  /* asm assumes K==128 */
                const int16_t* wpt  = w0 + 2;
                const int16_t* rhend = a0 + (K - 2);
                const int16_t* rb_   = a0;
                int32_t w00 = (int32_t)w0[0], w10 = (int32_t)w1[0];
                int32_t w01 = (int32_t)w0[1], w11 = (int32_t)w1[1];
                __asm__ __volatile__(
".p2align 4\n"
"mv  a4, %[w00]\n"
"mv  a5, %[w10]\n"
"mv  a6, %[w01]\n"
"mv  a7, %[w11]\n"
"1:\n"
/* loads: 4 rows at k */
"lh  t0, 0(%[rb])\n"
"lh  t1, 256(%[rb])\n"
"lh  t2, 512(%[rb])\n"
"lh  t3, 768(%[rb])\n"
/* col0 x w0[k] -> cc0..cc30 */
"mul  t4, t0, a4\n"
"mul  t5, t1, a4\n"
"mul  t6, t2, a4\n"
"mul  a0, t3, a4\n"
"add  %[cc0], %[cc0], t4\n"
"add  %[cc10], %[cc10], t5\n"
"add  %[cc20], %[cc20], t6\n"
"add  %[cc30], %[cc30], a0\n"
/* col1 x w1[k] (reuse same act values) */
"mul  t4, t0, a5\n"
"mul  t5, t1, a5\n"
"mul  t6, t2, a5\n"
"mul  a0, t3, a5\n"
"add  %[cc1], %[cc1], t4\n"
"add  %[cc11], %[cc11], t5\n"
"add  %[cc21], %[cc21], t6\n"
"add  %[cc31], %[cc31], a0\n"
/* loads: 4 rows at k+1 */
"lh  t0, 2(%[rb])\n"
"lh  t1, 258(%[rb])\n"
"lh  t2, 514(%[rb])\n"
"lh  t3, 770(%[rb])\n"
/* col0 x w0[k+1] */
"mul  t4, t0, a6\n"
"mul  t5, t1, a6\n"
"mul  t6, t2, a6\n"
"mul  a0, t3, a6\n"
"add  %[cc0], %[cc0], t4\n"
"add  %[cc10], %[cc10], t5\n"
"add  %[cc20], %[cc20], t6\n"
"add  %[cc30], %[cc30], a0\n"
/* col1 x w1[k+1] */
"mul  t4, t0, a7\n"
"mul  t5, t1, a7\n"
"mul  t6, t2, a7\n"
"mul  a0, t3, a7\n"
"add  %[cc1], %[cc1], t4\n"
"add  %[cc11], %[cc11], t5\n"
"add  %[cc21], %[cc21], t6\n"
"add  %[cc31], %[cc31], a0\n"
/* prefetch w0[k+2],w1[k+2] -> a4,a5 ; w0[k+3],w1[k+3] -> a6,a7 */
"lh  a4, 0(%[wpt])\n"
"lh  a5, 256(%[wpt])\n"
"lh  a6, 2(%[wpt])\n"
"lh  a7, 258(%[wpt])\n"
"addi %[wpt], %[wpt], 4\n"
"addi %[rb], %[rb], 4\n"
"bne %[rb], %[rhend], 1b\n"
/* tail: final pair (k = K-2, K-1) already in a4..a7 */
"lh  t0, 0(%[rb])\n"
"lh  t1, 256(%[rb])\n"
"lh  t2, 512(%[rb])\n"
"lh  t3, 768(%[rb])\n"
"mul  t4, t0, a4\n"
"mul  t5, t1, a4\n"
"mul  t6, t2, a4\n"
"mul  a0, t3, a4\n"
"add  %[cc0], %[cc0], t4\n"
"add  %[cc10], %[cc10], t5\n"
"add  %[cc20], %[cc20], t6\n"
"add  %[cc30], %[cc30], a0\n"
"mul  t4, t0, a5\n"
"mul  t5, t1, a5\n"
"mul  t6, t2, a5\n"
"mul  a0, t3, a5\n"
"add  %[cc1], %[cc1], t4\n"
"add  %[cc11], %[cc11], t5\n"
"add  %[cc21], %[cc21], t6\n"
"add  %[cc31], %[cc31], a0\n"
"lh  t0, 2(%[rb])\n"
"lh  t1, 258(%[rb])\n"
"lh  t2, 514(%[rb])\n"
"lh  t3, 770(%[rb])\n"
"mul  t4, t0, a6\n"
"mul  t5, t1, a6\n"
"mul  t6, t2, a6\n"
"mul  a0, t3, a6\n"
"add  %[cc0], %[cc0], t4\n"
"add  %[cc10], %[cc10], t5\n"
"add  %[cc20], %[cc20], t6\n"
"add  %[cc30], %[cc30], a0\n"
"mul  t4, t0, a7\n"
"mul  t5, t1, a7\n"
"mul  t6, t2, a7\n"
"mul  a0, t3, a7\n"
"add  %[cc1], %[cc1], t4\n"
"add  %[cc11], %[cc11], t5\n"
"add  %[cc21], %[cc21], t6\n"
"add  %[cc31], %[cc31], a0\n"
        : [cc0] "+r"(c00), [cc10] "+r"(c10), [cc20] "+r"(c20), [cc30] "+r"(c30),
          [cc1] "+r"(c01), [cc11] "+r"(c11), [cc21] "+r"(c21), [cc31] "+r"(c31),
          [wpt] "+r"(wpt), [rb] "+r"(rb_)
        : [w00] "r"(w00), [w10] "r"(w10), [w01] "r"(w01), [w11] "r"(w11),
          [rhend] "r"(rhend)
        : "a0", "a4", "a5", "a6", "a7", "t0", "t1", "t2", "t3", "t4", "t5", "t6");
            } else
#endif
            {
                int k = 0;
                for (; k + 1 < K; k += 2) {
                    int32_t b0 = w0[k], b1 = w1[k], b0n = w0[k+1], b1n = w1[k+1];
                    int32_t v00=a0[k],   v10=a1[k],   v20=a2[k],   v30=a3[k];
                    int32_t v01=a0[k+1], v11=a1[k+1], v21=a2[k+1], v31=a3[k+1];
                    c00 += v00*b0; c10 += v10*b0; c20 += v20*b0; c30 += v30*b0;
                    c01 += v00*b1; c11 += v10*b1; c21 += v20*b1; c31 += v30*b1;
                    c00 += v01*b0n; c10 += v11*b0n; c20 += v21*b0n; c30 += v31*b0n;
                    c01 += v01*b1n; c11 += v11*b1n; c21 += v21*b1n; c31 += v31*b1n;
                }
                for (; k < K; k++) {
                    int32_t b0 = w0[k], b1 = w1[k];
                    int32_t v00=a0[k], v10=a1[k], v20=a2[k], v30=a3[k];
                    c00 += v00*b0; c10 += v10*b0; c20 += v20*b0; c30 += v30*b0;
                    c01 += v00*b1; c11 += v10*b1; c21 += v20*b1; c31 += v30*b1;
                }
            }            const int32_t bq0 = BQ[j], bq1 = BQ[j+1];
            int32_t* xr = xq + (size_t)it * rowStride + j;
#define R1EMIT(idx, cv, bq) do { \
                int64_t P = (int64_t)(cv + bq) * (int64_t)r_m; \
                int32_t cb; \
                if (re >= 0) cb = (int32_t)(P << re); \
                else { int rs = -re; cb = (int32_t)((P + (P < 0 ? -(1LL << (rs - 1)) : (1LL << (rs - 1)))) >> rs); } \
                xr[idx] += cb; } while (0)
            R1EMIT(0,            c00, bq0);
            R1EMIT(1,            c01, bq1);
            R1EMIT(rowStride,    c10, bq0);
            R1EMIT(rowStride + 1, c11, bq1);
            R1EMIT(2*rowStride,  c20, bq0);
            R1EMIT(2*rowStride + 1, c21, bq1);
            R1EMIT(3*rowStride,  c30, bq0);
            R1EMIT(3*rowStride + 1, c31, bq1);
#undef R1EMIT
        }
    }
    if (N & 1) {
        int j = N - 1;
        const int16_t* wr = Wq + (size_t)j * K;
        for (int it = 0; it < M; it += 4) {
            const int16_t* a0 = Aq + (size_t)(it+0) * K;
            const int16_t* a1 = Aq + (size_t)(it+1) * K;
            const int16_t* a2 = Aq + (size_t)(it+2) * K;
            const int16_t* a3 = Aq + (size_t)(it+3) * K;
            int32_t c0=0,c1=0,c2=0,c3=0;
            for (int k = 0; k < K; k++) {
                int32_t b = wr[k];
                c0 += (int32_t)a0[k]*b; c1 += (int32_t)a1[k]*b;
                c2 += (int32_t)a2[k]*b; c3 += (int32_t)a3[k]*b;
            }
            const int32_t bq = BQ[j];
            int32_t* xr = xq + (size_t)it * rowStride + j;
#define R1EMIT(cv) do { \
                int64_t P = (int64_t)(cv + bq) * (int64_t)r_m; \
                int32_t cb; \
                if (re >= 0) cb = (int32_t)(P << re); \
                else { int rs = -re; cb = (int32_t)((P + (P < 0 ? -(1LL << (rs - 1)) : (1LL << (rs - 1)))) >> rs); } \
                xr[0] += cb; } while (0)
            R1EMIT(c0);
            xr += rowStride; R1EMIT(c1);
            xr += rowStride; R1EMIT(c2);
            xr += rowStride; R1EMIT(c3);
#undef R1EMIT
        }
    }
}

/* fused LN on an ALREADY-INT32 residual (scale sx) -> a16 Q15 activations.
 * Same as tm_bn_q15_int but the input is int32, so pass 1 is pure integer
 * row stats (no float amax scan, no float->q15 quantize) and sxi = sx. */

float tm_bn_q15_res(const int32_t* in_q, float sx, const float* gamma,
                    const float* beta, int16_t* out_q, int S, int D) {
    const int32_t Q = 32767;
        float bmax = 0.0f;
    for (int32_t i = 0; i < S; i++) {
        const int32_t* src = in_q + (size_t)i * D;
        int16_t* dst = out_q + (size_t)i * D;
        int64_t amax = 0;
        for (int32_t k = 0; k < D; k++) {
            int32_t q = src[k];
            int64_t a = q < 0 ? -(int64_t)q : (int64_t)q;
            if (a > amax) amax = a;
        }
        /* local integer rescale: p15 = qx>>sh with p15 amax in [16384,32767].
         * Recreates the baseline's adaptive q15 quantization range but with
         * integer ops (no float amax scan, no float->q15 quantize). */
        int32_t sh = 0;
        if (amax > 0) {
            uint32_t b = (uint32_t)(amax > 32767 ? amax : 1);
            /* bit_width(amax) - 15, clamped to [0, 30] */
            int bw = 32 - __builtin_clz((uint32_t)amax);
            sh = bw - 15;
            if (sh < 0) sh = 0;
            if (sh > 30) sh = 30;
        }
        int32_t s1 = 0; int64_t s2 = 0; int32_t mx15 = 0;
        for (int32_t k = 0; k < D; k++) {
            int32_t p = (int32_t)src[k] >> sh;
            if (sh == 0) p = (int32_t)src[k];
            int32_t a = p < 0 ? -p : p;
            if (a > mx15) mx15 = a;
            s1 += p;
            s2 += (int64_t)p * (int64_t)p;
            dst[k] = (int16_t)p;
        }
        s1_[i] = s1; s2_[i] = s2; mx15_[i] = mx15; sh_[i] = sh;
    }
    float mean_r[TM_S], rstd_r[TM_S]; int32_t meanq_[TM_S]; int32_t rstd15_[TM_S];
    for (int32_t i = 0; i < S; i++) {
        int32_t mq = s1_[i] / D;
        int64_t vq = (s2_[i] / D) - (int64_t)mq * (int64_t)mq;
        if (vq < 0) vq = 0;
        float sxi = (float)(1LL << sh_[i]) * sx;     /* value per p15 LSB */
        float mean = (float)mq * sxi;
        float vr = (float)vq * sxi * sxi;
        float rstd = 1.0f / sqrtf(vr + TM_LN_EPS);
        meanq_[i] = mq;  mean_r[i] = mean; rstd_r[i] = rstd;
        rstd15_[i] = (int32_t)(rstd * 32768.0f + 0.5f);
        float br = rstd * ((float)mx15_[i] * sxi + (mean < 0.0f ? -mean : mean));
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
    for (int32_t k = 0; k < D; k++) {
        float F = sa * gamma[k];                     /* per-LSB factor (sxi applied later per row) */
        Fk_[k] = (int32_t)(F * 32768.0f + (F < 0.0f ? -0.5f : 0.5f));
        Bk_[k] = (int32_t)(sa * beta[k] + (sa * beta[k] < 0.0f ? -0.5f : 0.5f));
    }
    const int32_t R = 1 << 14;
    for (int32_t i = 0; i < S; i++) {
        int16_t* p = out_q + (size_t)i * D;
        int32_t mq = meanq_[i];
        /* fold sxi*rstd into one per-row 30-bit factor (keeps pass 2 to two
         * int mults/element like baseline):  t = d*sa*gamma ;  u = t*PR;  */
        float sxi = (float)(1LL << sh_[i]) * sx;
        float PR = sxi * rstd_r[i] * 1073741824.0f;         /* *2^30 */
        int32_t PR15 = (int32_t)(PR + (PR < 0.0f ? -0.5f : 0.5f));
        for (int32_t k = 0; k < D; k++) {
            int32_t d = (int32_t)p[k] - mq;
            int64_t p64 = (int64_t)d * Fk_[k];             /* d*sa*gamma*2^15 */
            int32_t t = (int32_t)((p64 + (p64 < 0 ? -R : R)) >> 15);
            int64_t q64 = (int64_t)t * PR15;               /* /2^30 -> *2^-15 net */
            int32_t u = (int32_t)((q64 + (q64 < 0 ? -R : R)) >> 30);
            int32_t q = u + Bk_[k];
            if (q > Q) q = Q; else if (q < -Q) q = -Q;
            p[k] = (int16_t)q;
        }
    }
    return sa;
}
/* final LayerNorm with float output from an int32 residual (scale sx):
 * yout = LN(x*sx) with gamma/beta. pass 1 int stats; output fp32. */

void tm_ln_final_res(const int32_t* in_q, float sx, const float* gamma,
                     const float* beta, float* out, int S, int D) {
    for (int i = 0; i < S; i++) {
        const int32_t* p = in_q + (size_t)i * D;
        float* dst = out + (size_t)i * D;
        int64_t amax = 0;
        for (int k = 0; k < D; k++) {
            int64_t a = p[k] < 0 ? -(int64_t)p[k] : (int64_t)p[k];
            if (a > amax) amax = a;
        }
        int32_t sh = 0;
        if (amax > 0) {
            int bw = 32 - __builtin_clz((uint32_t)amax);
            sh = bw - 15; if (sh < 0) sh = 0; if (sh > 30) sh = 30;
        }
        int32_t s1 = 0; int64_t s2 = 0;
        for (int k = 0; k < D; k++) {
            int32_t q = (int32_t)p[k] >> sh;
            s1 += q; s2 += (int64_t)q * (int64_t)q;
        }
        int32_t mq = s1 / D;
        int64_t vq = (s2 / D) - (int64_t)mq * (int64_t)mq;
        if (vq < 0) vq = 0;
        float sxi = (float)(1LL << sh) * sx;
        float mean = (float)mq * sxi;
        float vr = (float)vq * sxi * sxi;
        float rstd = 1.0f / sqrtf(vr + TM_LN_EPS);
        float gsx = sxi * rstd;
        for (int k = 0; k < D; k++) {
            dst[k] = (float)((int32_t)(p[k] >> sh) - mq) * gsx * gamma[k] + beta[k];
        }
    }
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
/* ================= GELU (fixed-table, ROCK 20) =================
 * out = 0.5*x*(1+erf(x/(sa2*sqrt2))),  sa2 = TM_QACT_MAX/amax.
 * erf via FIXED 2049-entry table over the argument z>=0 (gelu_tab_2049.h)
 * + per-layer integer index scale.  index(x)=|x|/sqrt2/sa2/ZSTEP;
 * q = round(csc*2^S).  Replaces the per-layer soft-float LUT rebuild
 * (~40-55 ms/fwd) with ~7 integer ops/element (4 KB flash .rodata, 0 RAM).
 */
void tm_gelu_q15_lut(int16_t* x, int n, float amax) {
    const float inv_sqrt2 = 0.70710677f;
    const float ZSTEP = 3.2f / 2048.0f;
    const float csc = (amax / (32767.0f * inv_sqrt2)) / ZSTEP;
    int S = 0;
    int32_t q = 1;
    if (csc < 2048.0f) {
        while (S < 20) {
            int32_t qtry = (int32_t)(csc * (float)(1 << S) + 0.5f);
            if ((int64_t)32767 * (int64_t)qtry < ((int64_t)1 << 30)) { q = qtry; S++; }
            else break;
        }
    }
    if (csc >= 2048.0f) {
        for (int i = 0; i < n; i++) x[i] = x[i] > 0 ? x[i] : (int16_t)0;
        return;
    }
    const int32_t half = 1 << (S - 1);
    const int32_t mask = (1 << S) - 1;
    for (int i = 0; i < n; i++) {
        int32_t v = x[i];
        if (v == 0) continue;
        int32_t ax = v < 0 ? -v : v;
        int64_t avq = (int64_t)ax * q;
        int32_t idx = (int32_t)(avq >> S);
        if (idx >= 2048) { x[i] = v > 0 ? v : (int16_t)0; continue; }
        int32_t e0 = g_erf_tab[idx], e1 = g_erf_tab[idx + 1];
        int32_t off = (int32_t)(avq & mask);
        int32_t ev = e0 + (((e1 - e0) * off + half) >> S);
        float g;
        if (v > 0) g = 0.5f * (float)ax * (32767.0f + (float)ev) / 32767.0f;
        else       g = 0.5f * (float)ax * ((float)ev - 32767.0f) / 32767.0f;
        x[i] = (int16_t)(g + (g >= 0.0f ? 0.5f : -0.5f));
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

/* ---- microbench: MUL latency / issue + flash-vs-IRAM GEMM loop pressure ---- */
void tm_microbench(char* out, size_t outsz) {
#if defined(__riscv)
    volatile int32_t x0=2,x1=3,x2=5,x3=7,x4=11,x5=13,x6=17,x7=19;
    uint32_t w0=2,w1=3,w2=5,w3=7,w4=11,w5=13,w6=17,w7=19;
    volatile uint64_t r;
    uint64_t t;
    /* dependent chain: MUL latency + add serialize */
    x0=2; for (int i=0;i<10000;i++) x0 = x0*w0; r = x0;
    t = tm_cyc_now();
    x0=2; for (int i=0;i<100000;i++) x0 = x0*w0; r = x0;
    uint64_t dep = tm_cyc_now()-t;
    /* 8 independent chains: issue-limited throughput */
    t = tm_cyc_now();
    for (int i=0;i<100000;i++){
        x0=x0*w0; x1=x1*w1; x2=x2*w2; x3=x3*w3;
        x4=x4*w4; x5=x5*w5; x6=x6*w6; x7=x7*w7;
    }
    r = (uint64_t)(x0+x1+x2+x3+x4+x5+x6+x7);
    uint64_t par = tm_cyc_now()-t;
    /* raw 4x2 GEMM k-loop from this location (flash or IRAM depending on env) */
    int32_t c0=0,c1=0,c2=0,c3=0,c4=0,c5=0,c6=0,c7=0;
    static int16_t ga[8][128], gw[2][128];
    t = tm_cyc_now();
    for (int k=0;k<128;k++){
        int16_t a0=ga[0][k],a1=ga[1][k],a2=ga[2][k],a3=ga[3][k];
        int16_t b0=gw[0][k], b1=gw[1][k];
        c0+=a0*b0; c1+=a1*b0; c2+=a2*b0; c3+=a3*b0;
        c4+=a0*b1; c5+=a1*b1; c6+=a2*b1; c7+=a3*b1;
    }
    uint64_t g1 = tm_cyc_now()-t;
    t = tm_cyc_now();
    for (int kk=0;kk<128;kk+=2){
        int16_t a0=ga[0][kk],a1=ga[1][kk],a2=ga[2][kk],a3=ga[3][kk];
        int16_t b0=gw[0][kk], b1=gw[1][kk];
        c0+=a0*b0; c1+=a1*b0; c2+=a2*b0; c3+=a3*b0;
        c4+=a0*b1; c5+=a1*b1; c6+=a2*b1; c7+=a3*b1;
        a0=ga[0][kk+1];a1=ga[1][kk+1];a2=ga[2][kk+1];a3=ga[3][kk+1];
        b0=gw[0][kk+1]; b1=gw[1][kk+1];
        c0+=a0*b0; c1+=a1*b0; c2+=a2*b0; c3+=a3*b0;
        c4+=a0*b1; c5+=a1*b1; c6+=a2*b1; c7+=a3*b1;
    }
    uint64_t g2 = tm_cyc_now()-t;
    r = (uint64_t)(c0+c1+c2+c3+c4+c5+c6+c7);
    (void)r;
    snprintf(out, outsz,
       "MB dep=%.2fcz par8=%.2fcz 4x2k1=%.2f cz/MAC 4x2k2=%.2f cz/MAC\n",
       (double)dep/100000.0, (double)par/(100000u*8u),
       (double)g1/128.0/8.0, (double)g2/128.0/8.0);
#else
    snprintf(out, outsz, "MB host\n");
#endif
}

void tm_kbench2(char* out, size_t outsz) {
#if defined(__riscv)
    enum { S=128, HD=32 };
    int16_t* qh = tm_gemm_a16();
    int16_t* kh = qh + S*HD;
    int16_t* vh = kh + S*HD;
    /* p15b and dummy4 live in the a16 tail (2048 int32 free after qh/kh/vh) */
    int32_t* p15b = (int32_t*)(qh + 3*S*HD);
    int32_t* dummy4 = p15b + S;
    uint32_t st=12345;
    for (int i=0;i<S*HD;i++){ st=st*1664525u+1013904223u; qh[i]=(int16_t)(st>>16); kh[i]=(int16_t)((st>>8)&0xffff); vh[i]=(int16_t)((st>>4)&0xffff); }
    for (int i=0;i<S;i++){ st=st*1664525u+1013904223u; p15b[i]=(int32_t)(st>>17); }
    uint32_t cyc0, cyc1;
    volatile int32_t sink=0;
    double qk_cur=0,qk_v2=0,pv_cur=0,pv_v2=0,c5_float=0,c5_int=0,c5_epi=0;

    cyc0 = tm_cyc_now();
    for (int rep=0; rep<4; rep++) {
      for (int i=0;i<S;i++){
        const int16_t* qi16=qh+(size_t)i*HD;
        int64_t maxs = INT64_MIN;
        for (int j=0;j<=i;j++){
          const int16_t* kj16=kh+(size_t)j*HD;
          int64_t dot=0; int d=0;
          for (; d+1<HD; d+=2)
            dot += (int32_t)(int16_t)qi16[d]*(int32_t)kj16[d]
                 + (int32_t)(int16_t)qi16[d+1]*(int32_t)kj16[d+1];
          for (; d<HD; d++) dot += (int32_t)(int16_t)qi16[d]*(int32_t)kj16[d];
          if (dot>maxs) maxs=dot;
        }
        sink += (int32_t)maxs;
      }
    }
    cyc1 = tm_cyc_now();
    qk_cur = (double)(cyc1-cyc0)/4.0;

    cyc0 = tm_cyc_now();
    for (int rep=0; rep<4; rep++) {
      for (int i=0;i<S;i++){
        const int16_t* qi16=qh+(size_t)i*HD;
        int32_t maxs32=INT32_MIN;
        for (int j=0;j<=i;j++){
          const int16_t* kj16=kh+(size_t)j*HD;
          int32_t L=0,H=0; int d=0;
          for (; d+1<HD; d+=2){
            int32_t p = (int32_t)qi16[d]*(int32_t)kj16[d]
                      + (int32_t)qi16[d+1]*(int32_t)kj16[d+1];
            L += p; if ((uint32_t)L < (uint32_t)p) H+=1;
          }
          (void)H;
          if (L>maxs32) maxs32=L;
        }
        sink += maxs32;
      }
    }
    cyc1 = tm_cyc_now();
    qk_v2 = (double)(cyc1-cyc0)/4.0;

    cyc0 = tm_cyc_now();
    for (int rep=0; rep<4; rep++) {
      for (int i=0;i<S;i++){
        int32_t a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0,a7=0;
        for (int db=0; db<HD; db+=8){
          for (int j=0;j<=i;j++){
            const int16_t* vj=vh+(size_t)j*HD;
            int32_t p=p15b[j];
            a0+=p*(int32_t)vj[db+0]; a1+=p*(int32_t)vj[db+1];
            a2+=p*(int32_t)vj[db+2]; a3+=p*(int32_t)vj[db+3];
            a4+=p*(int32_t)vj[db+4]; a5+=p*(int32_t)vj[db+5];
            a6+=p*(int32_t)vj[db+6]; a7+=p*(int32_t)vj[db+7];
          }
        }
        sink += a0+a1+a2+a3+a4+a5+a6+a7;
      }
    }
    cyc1 = tm_cyc_now();
    pv_cur = (double)(cyc1-cyc0)/4.0;

    cyc0 = tm_cyc_now();
    for (int rep=0; rep<4; rep++) {
      for (int i=0;i<S;i++){
        int32_t c0=0,c1=0,c2=0,c3=0;
        for (int db=0; db<HD; db+=4){
          for (int j=0;j<=i;j++){
            const int16_t* vj=vh+(size_t)j*HD;
            int32_t p=p15b[j];
            c0+=p*(int32_t)vj[db]; c1+=p*(int32_t)vj[db+1];
            c2+=p*(int32_t)vj[db+2]; c3+=p*(int32_t)vj[db+3];
          }
        }
        sink += c0+c1+c2+c3;
      }
    }
    cyc1 = tm_cyc_now();
    pv_v2 = (double)(cyc1-cyc0)/4.0;

    {
      enum { M=128, K=128, N=128, IBLK=4 };
      /* reuse qh/kh scratch for A5/W5; dummy circular store to limit RAM */
      int16_t* A5 = qh;
      int16_t* W5 = kh + S*HD;
      int32_t amx=12345;
      for (int i=0;i<M*K;i++){ amx=amx*1103515245+12345; A5[i]=(int16_t)(amx>>16); }
      for (int i=0;i<N*K;i++){ amx=amx*1103515245+12345; W5[i]=(int16_t)(amx>>17); }
      uint32_t t0,t1;
      t0=tm_cyc_now();
      for (int rep=0;rep<4;rep++){
        for (int j=0;j<N;j++){
          const int16_t* wr=W5+(size_t)j*K;
          for (int it=0; it<M; it+=IBLK){
            const int16_t* a0=A5+(size_t)(it+0)*K; const int16_t* a1=A5+(size_t)(it+1)*K;
            const int16_t* a2=A5+(size_t)(it+2)*K; const int16_t* a3=A5+(size_t)(it+3)*K;
            int32_t c0=0,c1=0,c2=0,c3=0;
            for (int k=0;k<K;k++){ int32_t b=wr[k];
              c0+=(int32_t)a0[k]*b; c1+=(int32_t)a1[k]*b; c2+=(int32_t)a2[k]*b; c3+=(int32_t)a3[k]*b; }
            float bj=(float)j;
            int32_t e0=(int32_t)((float)c0*0.25f+bj);
            sink += e0 ^ (int32_t)((float)c1*0.25f+bj) ^ (int32_t)((float)c2*0.25f+bj) ^ (int32_t)((float)c3*0.25f+bj);
          }
        }
      }
      t1=tm_cyc_now();
      c5_float=(double)(t1-t0)/4.0;
      t0=tm_cyc_now();
      for (int rep=0;rep<4;rep++){
        for (int j=0;j<N;j++){
          const int16_t* wr=W5+(size_t)j*K;
          for (int it=0; it<M; it+=IBLK){
            const int16_t* a0=A5+(size_t)(it+0)*K; const int16_t* a1=A5+(size_t)(it+1)*K;
            const int16_t* a2=A5+(size_t)(it+2)*K; const int16_t* a3=A5+(size_t)(it+3)*K;
            int32_t c0=0,c1=0,c2=0,c3=0;
            for (int k=0;k<K;k++){ int32_t b=wr[k];
              c0+=(int32_t)a0[k]*b; c1+=(int32_t)a1[k]*b; c2+=(int32_t)a2[k]*b; c3+=(int32_t)a3[k]*b; }
            int32_t dn=it*N+j; dummy4[dn&127]^=c0; dummy4[(dn+1)&127]^=c1; dummy4[(dn+2)&127]^=c2; dummy4[(dn+3)&127]^=c3;
          }
        }
      }
      t1=tm_cyc_now();
      c5_int=(double)(t1-t0)/4.0;
      t0=tm_cyc_now();
      for (int rep=0;rep<4;rep++){
        for (int it=0; it<M; it+=IBLK){
          for (int j=0;j<N;j++){
            int32_t c0=dummy4[(it*N+j)&127];
            sink += (int32_t)((float)c0*0.25f+(float)j) ^ (int32_t)((float)dummy4[(it*N+j+1)&127]*0.25f);
          }
        }
      }
      t1=tm_cyc_now();
      c5_epi=(double)(t1-t0)/4.0;
    }
    {
      enum { M=128, K=128, N=128, IBLK=4 };
      int16_t* A5 = qh;
      int16_t* W5 = kh + S*HD;
      uint32_t t0,t1;
      /* current core5 inner shape (j2 / i4 / K-pair) */
      t0=tm_cyc_now();
      for (int rep=0;rep<4;rep++){
        for (int j=0;j+1<N;j+=2){
          const int16_t* w0=W5+(size_t)j*K; const int16_t* w1=W5+(size_t)(j+1)*K;
          for (int it=0;it<M;it+=4){
            const int16_t* a0=A5+(size_t)it*K;
            int32_t c00=0,c10=0,c20=0,c30=0,c01=0,c11=0,c21=0,c31=0;
            int k=0;
            for (;k+1<K;k+=2){
              int32_t b0=w0[k],b1=w1[k],b0n=w0[k+1],b1n=w1[k+1];
              int32_t v00=a0[k],v10=a0[K+k],v20=a0[2*K+k],v30=a0[3*K+k];
              int32_t v01=a0[k+1],v11=a0[K+k+1],v21=a0[2*K+k+1],v31=a0[3*K+k+1];
              c00+=v00*b0;c10+=v10*b0;c20+=v20*b0;c30+=v30*b0;
              c01+=v00*b1;c11+=v11*b1;c21+=v21*b1;c31+=v31*b1;
              c00+=v01*b0n;c10+=v11*b0n;c20+=v21*b0n;c30+=v31*b0n;
              c01+=v01*b1n;c11+=v11*b1n;c21+=v21*b1n;c31+=v31*b1n;
            }
            sink += c00+c10+c20+c30+c01+c11+c21+c31;
          }
        }
      }
      t1=tm_cyc_now();
      double c_c = (double)(t1-t0)/4.0;
      /* KB0-style 8x1 asm inner */
      t0=tm_cyc_now();
      for (int rep=0;rep<4;rep++){
        for (int j=0;j<N;j++){
          const int16_t* wr=W5+(size_t)j*K;
          for (int it=0;it<M;it+=8){
            int32_t c0=0,c1=0,c2=0,c3=0,c4=0,c5=0,c6=0,c7=0;
            const int16_t* rw=wr; const int16_t* rb=A5+(size_t)it*K;
            const int16_t* whend=wr+K;
            __asm__ __volatile__(
              ".p2align 4\n"
              "1:\n"
              "lh  a5, 0(%[wpt])\n"
              "lh  t0, 0(%[rb])\n"
              "lh  t1, 256(%[rb])\n"
              "lh  t2, 512(%[rb])\n"
              "lh  t3, 768(%[rb])\n"
              "lh  t4, 1024(%[rb])\n"
              "lh  t5, 1280(%[rb])\n"
              "lh  t6, 1536(%[rb])\n"
              "lh  a0, 1792(%[rb])\n"
              "addi %[wpt], %[wpt], 2\n"
              "addi %[rb], %[rb], 2\n"
              "mul  t0, t0, a5\n"
              "mul  t1, t1, a5\n"
              "mul  t2, t2, a5\n"
              "mul  t3, t3, a5\n"
              "mul  t4, t4, a5\n"
              "mul  t5, t5, a5\n"
              "mul  t6, t6, a5\n"
              "mul  a0, a0, a5\n"
              "add  %[c0], %[c0], t0\n"
              "add  %[c1], %[c1], t1\n"
              "add  %[c2], %[c2], t2\n"
              "add  %[c3], %[c3], t3\n"
              "add  %[c4], %[c4], t4\n"
              "add  %[c5], %[c5], t5\n"
              "add  %[c6], %[c6], t6\n"
              "add  %[c7], %[c7], a0\n"
              "bne  %[wpt], %[wend], 1b\n"
              : [c0] "+r"(c0),[c1] "+r"(c1),[c2] "+r"(c2),[c3] "+r"(c3),
                [c4] "+r"(c4),[c5] "+r"(c5),[c6] "+r"(c6),[c7] "+r"(c7),
                [wpt] "+r"(rw),[rb] "+r"(rb)
              : [wend] "r"(whend)
              : "a0","a5","t0","t1","t2","t3","t4","t5","t6");
            sink += c0+c1+c2+c3+c4+c5+c6+c7;
          }
        }
      }
      t1=tm_cyc_now();
      double c_asm = (double)(t1-t0)/4.0;
      snprintf(out,outsz,"K2 qk_cur=%.0f qk_v2=%.0f pv_cur=%.0f pv_v2=%.0f c5f=%.0f c5i=%.0f c5ep=%.0f Cinner=%.0f Asm8=%.0f sink=%d\n",
        qk_cur,qk_v2,pv_cur,pv_v2,c5_float,c5_int,c5_epi,c_c,c_asm,(int)sink);
    }
#else
    snprintf(out,outsz,"K2 host\n");
#endif
}
void tm_dbg_c5acc(char* out, size_t outsz) {
#if defined(__riscv)
    enum { K = 128 };
    int16_t* pb = tm_gemm_a16();
    int16_t* wb = pb + 4*K;
    uint32_t st = 999;
    for (int i = 0; i < 4*K; i++) { st = st*1664525u + 1013904223u; pb[i] = (int16_t)(st>>16); }
    for (int i = 0; i < 2*K; i++) { st = st*1664525u + 1013904223u; wb[i] = (int16_t)(st>>16); }
    const int16_t* a0 = pb; const int16_t* a1 = pb+K;
    const int16_t* a2 = pb+2*K; const int16_t* a3 = pb+3*K;
    const int16_t* w0 = wb; const int16_t* w1 = wb+K;
    int32_t cr[8] = {0};
    {
        int32_t c00=0,c10=0,c20=0,c30=0,c01=0,c11=0,c21=0,c31=0;
        for (int k = 0; k+1 < K; k += 2) {
            int32_t b0 = w0[k], b1 = w1[k], b0n = w0[k+1], b1n = w1[k+1];
            int32_t v00=a0[k], v10=a1[k], v20=a2[k], v30=a3[k];
            int32_t v01=a0[k+1], v11=a1[k+1], v21=a2[k+1], v31=a3[k+1];
            c00 += v00*b0; c10 += v10*b0; c20 += v20*b0; c30 += v30*b0;
            c01 += v00*b1; c11 += v10*b1; c21 += v20*b1; c31 += v30*b1;
            c00 += v01*b0n; c10 += v11*b0n; c20 += v21*b0n; c30 += v31*b0n;
            c01 += v01*b1n; c11 += v11*b1n; c21 += v21*b1n; c31 += v31*b1n;
        }
        cr[0]=c00; cr[1]=c10; cr[2]=c20; cr[3]=c30;
        cr[4]=c01; cr[5]=c11; cr[6]=c21; cr[7]=c31;
    }
    int32_t ca[8] = {0};
    {
        const int16_t* wpt  = w0 + 2;
        const int16_t* rhend = a0 + (K - 2);
        const int16_t* rb_   = a0;
        int32_t w00 = (int32_t)w0[0], w10 = (int32_t)w1[0];
        int32_t w01 = (int32_t)w0[1], w11 = (int32_t)w1[1];
        register int32_t cc0 = 0, cc10 = 0, cc20 = 0, cc30 = 0;
        register int32_t cc1 = 0, cc11 = 0, cc21 = 0, cc31 = 0;
        __asm__ __volatile__(
".p2align 4\n"
"mv  a4, %[w00]\n"
"mv  a5, %[w10]\n"
"mv  a6, %[w01]\n"
"mv  a7, %[w11]\n"
"1:\n"
/* loads: 4 rows at k */
"lh  t0, 0(%[rb])\n"
"lh  t1, 256(%[rb])\n"
"lh  t2, 512(%[rb])\n"
"lh  t3, 768(%[rb])\n"
/* col0 x w0[k] -> cc0..cc30 */
"mul  t4, t0, a4\n"
"mul  t5, t1, a4\n"
"mul  t6, t2, a4\n"
"mul  a0, t3, a4\n"
"add  %[cc0], %[cc0], t4\n"
"add  %[cc10], %[cc10], t5\n"
"add  %[cc20], %[cc20], t6\n"
"add  %[cc30], %[cc30], a0\n"
/* col1 x w1[k] (reuse same act values) */
"mul  t4, t0, a5\n"
"mul  t5, t1, a5\n"
"mul  t6, t2, a5\n"
"mul  a0, t3, a5\n"
"add  %[cc1], %[cc1], t4\n"
"add  %[cc11], %[cc11], t5\n"
"add  %[cc21], %[cc21], t6\n"
"add  %[cc31], %[cc31], a0\n"
/* loads: 4 rows at k+1 */
"lh  t0, 2(%[rb])\n"
"lh  t1, 258(%[rb])\n"
"lh  t2, 514(%[rb])\n"
"lh  t3, 770(%[rb])\n"
/* col0 x w0[k+1] */
"mul  t4, t0, a6\n"
"mul  t5, t1, a6\n"
"mul  t6, t2, a6\n"
"mul  a0, t3, a6\n"
"add  %[cc0], %[cc0], t4\n"
"add  %[cc10], %[cc10], t5\n"
"add  %[cc20], %[cc20], t6\n"
"add  %[cc30], %[cc30], a0\n"
/* col1 x w1[k+1] */
"mul  t4, t0, a7\n"
"mul  t5, t1, a7\n"
"mul  t6, t2, a7\n"
"mul  a0, t3, a7\n"
"add  %[cc1], %[cc1], t4\n"
"add  %[cc11], %[cc11], t5\n"
"add  %[cc21], %[cc21], t6\n"
"add  %[cc31], %[cc31], a0\n"
/* prefetch w0[k+2],w1[k+2] -> a4,a5 ; w0[k+3],w1[k+3] -> a6,a7 */
"lh  a4, 0(%[wpt])\n"
"lh  a5, 256(%[wpt])\n"
"lh  a6, 2(%[wpt])\n"
"lh  a7, 258(%[wpt])\n"
"addi %[wpt], %[wpt], 4\n"
"addi %[rb], %[rb], 4\n"
"bne %[rb], %[rhend], 1b\n"
/* tail: final pair (k = K-2, K-1) already in a4..a7 */
"lh  t0, 0(%[rb])\n"
"lh  t1, 256(%[rb])\n"
"lh  t2, 512(%[rb])\n"
"lh  t3, 768(%[rb])\n"
"mul  t4, t0, a4\n"
"mul  t5, t1, a4\n"
"mul  t6, t2, a4\n"
"mul  a0, t3, a4\n"
"add  %[cc0], %[cc0], t4\n"
"add  %[cc10], %[cc10], t5\n"
"add  %[cc20], %[cc20], t6\n"
"add  %[cc30], %[cc30], a0\n"
"mul  t4, t0, a5\n"
"mul  t5, t1, a5\n"
"mul  t6, t2, a5\n"
"mul  a0, t3, a5\n"
"add  %[cc1], %[cc1], t4\n"
"add  %[cc11], %[cc11], t5\n"
"add  %[cc21], %[cc21], t6\n"
"add  %[cc31], %[cc31], a0\n"
"lh  t0, 2(%[rb])\n"
"lh  t1, 258(%[rb])\n"
"lh  t2, 514(%[rb])\n"
"lh  t3, 770(%[rb])\n"
"mul  t4, t0, a6\n"
"mul  t5, t1, a6\n"
"mul  t6, t2, a6\n"
"mul  a0, t3, a6\n"
"add  %[cc0], %[cc0], t4\n"
"add  %[cc10], %[cc10], t5\n"
"add  %[cc20], %[cc20], t6\n"
"add  %[cc30], %[cc30], a0\n"
"mul  t4, t0, a7\n"
"mul  t5, t1, a7\n"
"mul  t6, t2, a7\n"
"mul  a0, t3, a7\n"
"add  %[cc1], %[cc1], t4\n"
"add  %[cc11], %[cc11], t5\n"
"add  %[cc21], %[cc21], t6\n"
"add  %[cc31], %[cc31], a0\n"
        : [cc0] "+r"(cc0), [cc10] "+r"(cc10), [cc20] "+r"(cc20), [cc30] "+r"(cc30),
          [cc1] "+r"(cc1), [cc11] "+r"(cc11), [cc21] "+r"(cc21), [cc31] "+r"(cc31),
          [wpt] "+r"(wpt), [rb] "+r"(rb_)
        : [w00] "r"(w00), [w10] "r"(w10), [w01] "r"(w01), [w11] "r"(w11),
          [rhend] "r"(rhend)
        : "a0", "a4", "a5", "a6", "a7", "t0", "t1", "t2", "t3", "t4", "t5", "t6");
        ca[0]=cc0; ca[1]=cc10; ca[2]=cc20; ca[3]=cc30;
        ca[4]=cc1; ca[5]=cc11; ca[6]=cc21; ca[7]=cc31;
    }
    int bad = 0, worst = 0;
    for (int q = 0; q < 8; q++) { int d = ca[q]-cr[q]; if (d<0) d=-d; if (d>worst) worst=d; if (d) bad++; }
    snprintf(out, outsz,
        "C5P asm=[%d %d %d %d | %d %d %d %d]\nC5P  C =[%d %d %d %d | %d %d %d %d] bad=%d worst=%d\n",
        ca[0],ca[1],ca[2],ca[3],ca[4],ca[5],ca[6],ca[7],
        cr[0],cr[1],cr[2],cr[3],cr[4],cr[5],cr[6],cr[7], bad, worst);
#else
    snprintf(out, outsz, "C5P host\n");
#endif
}
