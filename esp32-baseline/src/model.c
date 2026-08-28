/*
 * model.c - forward pass: L transformer layers + final LayerNorm (case 2).
 *
 * Per layer:
 *   norm1 -> (Q,K,V) proj -> multi-head causal attention -> O proj -> +res
 *   norm2 -> FFN1 -> GELU -> FFN2 -> +res
 *   final: LayerNorm
 *
 * Memory layout (single static arena, ~272 KB total on the C3):
 *   x     : 64 KB (input / residual)
 *   buf1  : 64 KB (normed activations / layer outputs)
 *   buf2  : 64 KB (attention context / ffn outputs)
 *   qh/kh/vh : 16 KB each (per-head projections)
 *   a16   : 32 KB (Q15 activation scratch inside tm_gemm_q12)
 *   out   : 64 KB (result)
 * Everything else is streaming (softmax row = 32 floats on stack).
 *
 * FAST mode quantizes the six per-layer projection GEMMs (Q15 x Q12);
 * attention QK/PV, LayerNorm and GELU stay in fp32. EXACT mode runs
 * everything in fp32 (slow on the no-FPU C3).
 */
#include "model.h"
#include "kernels.h"

#include <math.h>
#include <string.h>

static int g_mode = TM_MODE_DEFAULT;

void tm_set_mode(int mode) { g_mode = mode; }
int tm_get_mode(void) { return g_mode; }

/* static arena (only one forward at a time) */
static float g_x[TM_S * TM_D];
static float g_buf1[TM_S * TM_D];
static float g_buf2[TM_S * TM_D];
static float g_out[TM_S * TM_D];
static float g_qh[TM_S * TM_HD];
static float g_kh[TM_S * TM_HD];
static float g_vh[TM_S * TM_HD];

static uint32_t woff(int layer, int blk) {
    uint32_t o = (uint32_t)layer * TM_W_LAYER_FLOATS;
    switch (blk) {
        case TM_W_BLK_N1W: return o + 0;
        case TM_W_BLK_N1B: return o + 1 * TM_D;
        case TM_W_BLK_QW:  return o + 2 * TM_D;
        case TM_W_BLK_QB:  return o + 2 * TM_D + 1 * TM_D * TM_D;
        case TM_W_BLK_KW:  return o + 3 * TM_D + 1 * TM_D * TM_D;
        case TM_W_BLK_KB:  return o + 3 * TM_D + 2 * TM_D * TM_D;
        case TM_W_BLK_VW:  return o + 4 * TM_D + 2 * TM_D * TM_D;
        case TM_W_BLK_VB:  return o + 4 * TM_D + 3 * TM_D * TM_D;
        case TM_W_BLK_OW:  return o + 5 * TM_D + 3 * TM_D * TM_D;
        case TM_W_BLK_OB:  return o + 5 * TM_D + 4 * TM_D * TM_D;
        case TM_W_BLK_N2W: return o + 6 * TM_D + 4 * TM_D * TM_D;
        case TM_W_BLK_N2B: return o + 7 * TM_D + 4 * TM_D * TM_D;
        case TM_W_BLK_F1W: return o + 8 * TM_D + 4 * TM_D * TM_D;
        case TM_W_BLK_F1B: return o + 8 * TM_D + 4 * TM_D * TM_D + TM_F * TM_D;
        case TM_W_BLK_F2W: return o + 8 * TM_D + 4 * TM_D * TM_D + TM_F * TM_D + TM_F;
        case TM_W_BLK_F2B: return o + 8 * TM_D + 4 * TM_D * TM_D + TM_F * TM_D + TM_F + TM_D * TM_F;
        default: return 0;
    }
}

static float ldot(const float* a, const float* b, int n) {
    float s = 0.0f;
    for (int k = 0; k < n; k++) s += a[k] * b[k];
    return s;
}

/* Streaming online-rescale causal attention for one head over all S tokens.
 * qh/kh/vh: [S,HD]; result written into ctx[i*D + head*HD]. */
static void attn_head(float* ctx, const float* qh, const float* kh,
                      const float* vh, int head) {
    for (int i = 0; i < TM_S; i++) {
        const float* qi = qh + (size_t)i * TM_HD;
        float m = -INFINITY, lsum = 0.0f;
        float acc[TM_HD];
        for (int d = 0; d < TM_HD; d++) acc[d] = 0.0f;
        for (int j = 0; j <= i; j++) {
            const float* kj = kh + (size_t)j * TM_HD;
            const float* vj = vh + (size_t)j * TM_HD;
            float s = ldot(qi, kj, TM_HD) * TM_ATTN_SCALE;
            if (s > m) {                       /* new running max */
                float m2 = s;
                float r = (g_mode == TM_MODE_FAST)
                              ? tm_exp_fast(m - m2)   /* <=1, m2>m */
                              : tm_exp_f32(m - m2);
                lsum *= r;
                for (int d = 0; d < TM_HD; d++) acc[d] *= r;
                m = m2;
            }
            float p = (g_mode == TM_MODE_FAST) ? tm_exp_fast(s - m)
                                               : tm_exp_f32(s - m);
            lsum += p;
            for (int d = 0; d < TM_HD; d++) acc[d] += p * vj[d];
        }
        float* o = ctx + (size_t)i * TM_D + head * TM_HD;
        float inv = 1.0f / lsum;
        for (int d = 0; d < TM_HD; d++) o[d] = acc[d] * inv;
    }
}

void tm_scan_q12(const void* blob, TMQ12Weights* out) {
    const uint8_t* p = (const uint8_t*)blob;
    for (int l = 0; l < TM_L; l++) {
        for (int m = 0; m < 6; m++) {
            uint32_t count; float wscale;
            memcpy(&count, p, 4); p += 4;
            memcpy(&wscale, p, 4); p += 4;
            out->ws[l][m] = wscale;
            out->q[l][m]  = (const int16_t*)p;
            p += (size_t)count * 2;
        }
    }
}

void tm_forward(const float* xin, float* yout,
                const float* W, const TMQ12Weights* q12) {
    memcpy(g_x, xin, sizeof g_x);
    int fast = (g_mode == TM_MODE_FAST);

    for (int l = 0; l < TM_L; l++) {
        /* ---- norm1 ---- */
        tm_layernorm(g_x,
                     W + woff(l, TM_W_BLK_N1W), W + woff(l, TM_W_BLK_N1B),
                     g_buf1, TM_S, TM_D);

        /* ---- attention ---- */
        for (int h = 0; h < TM_H; h++) {
            if (fast) {
                /* per-head Q/K/V projections via Q15 x Q12 rows [h*HD:(h+1)*HD] */
                const int16_t* wq = q12->q[l][0] + (size_t)h * TM_HD * TM_D;
                const int16_t* wk = q12->q[l][1] + (size_t)h * TM_HD * TM_D;
                const int16_t* wv = q12->q[l][2] + (size_t)h * TM_HD * TM_D;
                float sq = q12->ws[l][0], sk = q12->ws[l][1], sv = q12->ws[l][2];
                tm_gemm_q12(g_buf1, wq, sq, W + woff(l, TM_W_BLK_QB) + h * TM_HD,
                            g_qh, TM_S, TM_D, TM_HD);
                tm_gemm_q12(g_buf1, wk, sk, W + woff(l, TM_W_BLK_KB) + h * TM_HD,
                            g_kh, TM_S, TM_D, TM_HD);
                tm_gemm_q12(g_buf1, wv, sv, W + woff(l, TM_W_BLK_VB) + h * TM_HD,
                            g_vh, TM_S, TM_D, TM_HD);
            } else {
                tm_gemm_f32(g_buf1, W + woff(l, TM_W_BLK_QW) + (size_t)h * TM_HD * TM_D,
                            W + woff(l, TM_W_BLK_QB) + h * TM_HD,
                            g_qh, TM_S, TM_D, TM_HD);
                tm_gemm_f32(g_buf1, W + woff(l, TM_W_BLK_KW) + (size_t)h * TM_HD * TM_D,
                            W + woff(l, TM_W_BLK_KB) + h * TM_HD,
                            g_kh, TM_S, TM_D, TM_HD);
                tm_gemm_f32(g_buf1, W + woff(l, TM_W_BLK_VW) + (size_t)h * TM_HD * TM_D,
                            W + woff(l, TM_W_BLK_VB) + h * TM_HD,
                            g_vh, TM_S, TM_D, TM_HD);
            }
            attn_head(g_buf2, g_qh, g_kh, g_vh, h);
        }

        /* ---- out projection (ctx in g_buf2 -> out into g_buf2 in place) ---- */
        if (fast) {
            /* reuse g_vh as temporaries: keep ctx, write result to g_buf1 */
            tm_gemm_q12(g_buf2, q12->q[l][3], q12->ws[l][3],
                        W + woff(l, TM_W_BLK_OB), g_buf1, TM_S, TM_D, TM_D);
        } else {
            tm_gemm_f32(g_buf2, W + woff(l, TM_W_BLK_OW),
                        W + woff(l, TM_W_BLK_OB), g_buf1, TM_S, TM_D, TM_D);
        }
        /* ---- residual ---- */
        tm_add_inplace(g_buf1, g_x, TM_S * TM_D);

        /* ---- norm2 ---- */
        tm_layernorm(g_x,
                     W + woff(l, TM_W_BLK_N2W), W + woff(l, TM_W_BLK_N2B),
                     g_buf1, TM_S, TM_D);

        /* ---- FFN ---- */
        if (fast) {
            tm_gemm_q12(g_buf1, q12->q[l][4], q12->ws[l][4],
                        W + woff(l, TM_W_BLK_F1B), g_buf2, TM_S, TM_D, TM_F);
        } else {
            tm_gemm_f32(g_buf1, W + woff(l, TM_W_BLK_F1W),
                        W + woff(l, TM_W_BLK_F1B), g_buf2, TM_S, TM_D, TM_F);
        }
        tm_gelu_inplace(g_buf2, TM_S * TM_F);
        if (fast) {
            tm_gemm_q12(g_buf2, q12->q[l][5], q12->ws[l][5],
                        W + woff(l, TM_W_BLK_F2B), g_buf1, TM_S, TM_F, TM_D);
        } else {
            tm_gemm_f32(g_buf2, W + woff(l, TM_W_BLK_F2W),
                        W + woff(l, TM_W_BLK_F2B), g_buf1, TM_S, TM_F, TM_D);
        }
        /* ---- residual ---- */
        tm_add_inplace(g_buf1, g_x, TM_S * TM_D);
    }

    /* ---- final norm ---- */
    tm_layernorm(g_x, W + TM_W_FINALW, W + TM_W_FINALB, g_out, TM_S, TM_D);
    memcpy(yout, g_out, sizeof g_out);
}
