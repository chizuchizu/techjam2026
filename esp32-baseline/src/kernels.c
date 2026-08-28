/*
 * kernels.c - implementation of the low-level kernels.
 */
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
    tm_gemm_core(a16, sa_inv, Wq, w_scale, bias, C, M, K, N, rowStride);
}

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
    const float g = sa_inv * w_scale;
    const int S = TM_S, HD = TM_HD;
    float amax = 0.0f;
    enum { IBLK = 4, JBLK = 2 };
    for (int it = 0; it < S; it += IBLK) {
        const int16_t* a0r = Aq + (size_t)it * K;
        const int16_t* a1r = Aq + (size_t)(it+1) * K;
        const int16_t* a2r = Aq + (size_t)(it+2) * K;
        const int16_t* a3r = Aq + (size_t)(it+3) * K;
        for (int jb = 0; jb < HD; jb += JBLK) {
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
            int32_t* r0 = acc + (size_t)it * HD + jb;
            int32_t* r1 = r0 + HD;
            int32_t* r2 = r1 + HD;
            int32_t* r3 = r2 + HD;
            r0[0]=c00; r0[1]=c01;
            r1[0]=c10; r1[1]=c11;
            r2[0]=c20; r2[1]=c21;
            r3[0]=c30; r3[1]=c31;
            float v;
            v = (float)c00*g + bias[jb];   if (v<0) v=-v;   if (v>amax) amax=v;
            v = (float)c01*g + bias[jb+1]; if (v<0) v=-v;   if (v>amax) amax=v;
            v = (float)c10*g + bias[jb];   if (v<0) v=-v;   if (v>amax) amax=v;
            v = (float)c11*g + bias[jb+1]; if (v<0) v=-v;   if (v>amax) amax=v;
            v = (float)c20*g + bias[jb];   if (v<0) v=-v;   if (v>amax) amax=v;
            v = (float)c21*g + bias[jb+1]; if (v<0) v=-v;   if (v>amax) amax=v;
            v = (float)c30*g + bias[jb];   if (v<0) v=-v;   if (v>amax) amax=v;
            v = (float)c31*g + bias[jb+1]; if (v<0) v=-v;   if (v>amax) amax=v;
        }
    }
    float sa = TM_QACT_MAX / (amax > 0.0f ? amax : 1.0f);
    /* Fixed-point pass in Q30 (int64): coefficient rounding error * acc must
     * stay below 0.5 q15 unit; a Q15 coefficient loses precision whenever acc
     * is large (acc can approach 2^31), so Q30 + int64 is required. */
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
    return amax / TM_QACT_MAX;
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
