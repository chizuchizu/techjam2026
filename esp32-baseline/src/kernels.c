/*
 * kernels.c - implementation of the low-level kernels.
 */
#include "kernels.h"

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
void tm_gemm_q12(const float* A, const int16_t* Wq, float w_scale,
                 const float* bias, float* C,
                 int M, int K, int N, int rowStride) {
    /* find activation scale + quantize A into a static scratch */
    static int16_t a16[TM_S * TM_D];
    float amax = 0.0f;
    for (int i = 0; i < M * K; i++) {
        float av = A[i] < 0.0f ? -A[i] : A[i];
        if (av > amax) amax = av;
    }
    if (amax == 0.0f) amax = 1.0f;          /* all-zero slice guard */
    float sa = TM_QACT_MAX / amax;
    float sa_inv = 1.0f / sa;

    for (int i = 0; i < M * K; i++) {
        float v = A[i] * sa;
        int q = (int)llrintf(v);            /* round-half-even */
        if (q > TM_QACT_MAX) q = (int)TM_QACT_MAX;
        if (q < -TM_QACT_MAX) q = -(int)TM_QACT_MAX;
        a16[i] = (int16_t)q;
    }

    const float g = sa_inv * w_scale;
    for (int i = 0; i < M; i++) {
        const int16_t* arow = a16 + (size_t)i * K;
        for (int j = 0; j < N; j++) {
            const int16_t* wrow = Wq + (size_t)j * K;
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc = (int32_t)((uint32_t)acc +
                                (uint32_t)((int32_t)arow[k] * (int32_t)wrow[k]));
            C[(size_t)i * rowStride + j] = (float)acc * g + (bias ? bias[j] : 0.0f);
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
