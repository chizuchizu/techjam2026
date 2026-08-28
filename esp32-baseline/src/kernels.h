/*
 * kernels.h - low-level numerics kernels (fp32 + fixed-point FAST path).
 * See tm_config.h for model geometry and mode definitions.
 */
#ifndef TM_KERNELS_H
#define TM_KERNELS_H

#include "tm_config.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------- fixed-point GEMM (FAST) ----------------
 * C[M,N] = (A_q15 . W_q12^T) * (1/sa) * w_scale + bias
 * A is fp32 input, quantized inside to Q15 with a per-call scale sa.
 * W is pre-quantized Q12; w_scale = max|W|/2047 (stored in the blob).
 * Accumulation is int32 (saturating-safe: realistic bounds ~9x below INT32_MAX).
 */
void tm_gemm_q12(const float* A, const int16_t* Wq, float w_scale,
                 const float* bias, float* C,
                 int M, int K, int N);

/* ---------------- fp32 GEMM (EXACT) ----------------
 * C[M,N] = A[M,K] . W[N,K]^T + bias[N]   (torchnn.Linear layout)
 */
void tm_gemm_f32(const float* A, const float* W, const float* bias,
                 float* C, int M, int K, int N);

/* ---------------- LayerNorm ----------------
 * two-pass fp32: mean, unbiased-norm var, eps=TM_LN_EPS.
 */
void tm_layernorm(const float* in, const float* gamma, const float* beta,
                  float* out, int S, int D);

/* ---------------- GELU (exact="none" reference order) ----------------
 * gelu(x) = 0.5*x*(1+erf(x/sqrt2)); erf via deg-11 minimax poly in
 *           t = |x|/sqrt2 * 0.5 - 1.  (valid |x|<4, clamped; max err ~4.6e-5)
 */
void tm_gelu_inplace(float* x, int n);

/* ---------------- elementwise add: y[i] += x[i] ---------------- */
void tm_add_inplace(const float* x, float* y, int n);

/* ---------------- fast exp for y<=0 (e^y, rel err ~1e-4) ----------------
 * exp2-style split: e^y = 2^(f + n); f in [0,1) via 5-term Taylor poly.
 */
float tm_exp_fast(float y);
#define TM_EXP_FAST_MIN -40.0f   /* below this -> 0 (exp(-40)~4e-18) */

/* fp32 expf for the exact path (libgcc soft-float on the C3). */
float tm_exp_f32(float y);

#ifdef __cplusplus
}
#endif

#endif /* TM_KERNELS_H */
