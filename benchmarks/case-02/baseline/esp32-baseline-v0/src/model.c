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

/* ---- tinyprof instrumentation (opt-in, [env:esp32-tinyprof-v0]) ----------
 * This file is commit 79f284a restored verbatim; the only additions are the
 * TP_B/TP_E zone brackets below and this block. Without -DTINYPROF_LIB they
 * all compile to nothing, so the firmware that produced the published 42.15 s
 * is still exactly what the default env builds. */
#ifdef TINYPROF_LIB
#include "tinyprof.h"

/* Overridden on device by main.cpp (Serial.print) and on host by the driver. */
__attribute__((weak)) void tm_prof_emit(const char* line) { (void)line; }
void tp_emit(const char* line) { tm_prof_emit(line); }

const char* tp_build_tag(void) { return TINYPROF_TAG; }

static const tp_shape_t tm_shape = { TM_S, TM_D, TM_H, TM_F, TM_L, 160 };
const tp_shape_t* tp_shape(void) { return &tm_shape; }

/* The pre-optimisation arena: four fp32 S*D buffers (x, buf1, buf2, out) plus
 * three fp32 per-head buffers. The optimised build later folded g_out away and
 * moved the head buffers to int16, which is most of the SRAM difference the
 * report shows. */
static const tp_arena_t tm_arenas[] = {
    { "g_x",    (uint32_t)(TM_S * TM_D * 4),  "fp32_activation" },
    { "g_buf1", (uint32_t)(TM_S * TM_D * 4),  "fp32_activation" },
    { "g_buf2", (uint32_t)(TM_S * TM_D * 4),  "fp32_activation" },
    { "g_out",  (uint32_t)(TM_S * TM_D * 4),  "fp32_activation" },
    { "g_qh",   (uint32_t)(TM_S * TM_HD * 2), "q15_head"        },
    { "g_kh",   (uint32_t)(TM_S * TM_HD * 2), "q15_head"        },
    { "g_vh",   (uint32_t)(TM_S * TM_HD * 2), "q15_head"        },
};
const tp_arena_t* tp_arena_table(void) { return tm_arenas; }
int tp_arena_count(void) { return (int)(sizeof tm_arenas / sizeof tm_arenas[0]); }

void tm_profile_dump(void) { tp_dump(); }

/* This build brackets the three attention phases per (i, j) rather than per
 * query row: unlike the optimised kernels, its QK dot, running-max rescale and
 * PV accumulate are fused into one j loop and cannot be separated per row
 * without restructuring the arithmetic being measured. The call counts
 * therefore differ from the optimised capture by construction; the report says
 * so rather than presenting the two counts as comparable.
 *
 * That granularity costs 3 * S*(S+1)/2 * H * L = 132,096 zones per forward.
 * Against 42 s on the C3 that is ~0.1% - fine. On a host build with an FPU,
 * where the same forward takes 89 ms, the probes cost more than the work. So
 * the phase split is a switch: -DTINYPROF_ATTN_PHASES=0 keeps only the `attn`
 * total, and is the right choice whenever the probe overhead the tool reports
 * back is a material fraction of the measurement. */
#ifndef TINYPROF_ATTN_PHASES
#define TINYPROF_ATTN_PHASES 1
#endif
#if TINYPROF_ATTN_PHASES
#define TP_BA(i) TP_B(i)
#define TP_EA(i) TP_E(i)
#else
#define TP_BA(i) do {} while (0)
#define TP_EA(i) do {} while (0)
#endif

#else
#define TP_B(i) do {} while (0)
#define TP_E(i) do {} while (0)
#define TP_BA(i) do {} while (0)
#define TP_EA(i) do {} while (0)
#define tp_wall_begin() do {} while (0)
#define tp_wall_end()   do {} while (0)
__attribute__((weak)) void tm_prof_emit(const char* line) { (void)line; }
void tm_profile_dump(void) { }
#endif

static int g_mode = TM_MODE_DEFAULT;

void tm_set_mode(int mode) { g_mode = mode; }
int tm_get_mode(void) { return g_mode; }

/* static arena (only one forward at a time). Sized for the C3's 320 KB
 * SRAM: x + buf1 + buf2 (64 KB fp32 each) + per-head q/k/v (fp32 48 KB)
 * + a16 (32 KB Q15 scratch in kernels.c)). g_out was removed: the final
 * LayerNorm writes into g_buf1, exposed via tm_output(). */
static float g_x[TM_S * TM_D];
static float g_buf1[TM_S * TM_D];
static float g_buf2[TM_S * TM_D];
/* per-head Q/K/V stored as int16 (Q15) + a per-head-slice float scale,
 * dequantized on read inside attn_head. Halves the 3 head buffers
 * (48 -> 24 KB). fp32 staging during projection reuses g_buf2. */
static int16_t g_qh[TM_S * TM_HD];
static int16_t g_kh[TM_S * TM_HD];
static int16_t g_vh[TM_S * TM_HD];
static float g_qs, g_ks, g_vs;   /* dequant scales (amax/32767) per head */

float* tm_input(void)  { return g_x; }
float* tm_output(void) { return g_buf1; }  /* final norm result */

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

/* Quantize an [S,HD] fp32 slice (strided over the 128-col ctx view) into
 * int16 Q15 (per-slice amax scale). src row i lives at src[i*rowStride .. +HD).
 * Returns the dequant scale = amax/32767 so value = (float)q16 * scale. */
static float quant_head(const float* src, int rowStride, int16_t* dst) {
    float amax = 0.0f;
    for (int i = 0; i < TM_S; i++) {
        const float* row = src + (size_t)i * rowStride;
        for (int d = 0; d < TM_HD; d++) {
            float v = row[d] < 0.0f ? -row[d] : row[d];
            if (v > amax) amax = v;
        }
    }
    if (amax == 0.0f) amax = 1.0f;
    float sa = TM_QACT_MAX / amax;
    for (int i = 0; i < TM_S; i++) {
        const float* row = src + (size_t)i * rowStride;
        int16_t* drow = dst + (size_t)i * TM_HD;
        for (int d = 0; d < TM_HD; d++) {
            int q = (int)llrintf(row[d] * sa);
            if (q > TM_QACT_MAX) q = (int)TM_QACT_MAX;
            if (q < -TM_QACT_MAX) q = -(int)TM_QACT_MAX;
            drow[d] = (int16_t)q;
        }
    }
    return amax / (float)TM_QACT_MAX;
}

/* Streaming online-rescale causal attention for one head over all S tokens.
 * qh/kh/vh are int16 [S,HD] + per-buffer dequant scales; the QK/PV math is
 * fp32 (dequantized). Result written into ctx[i*D + head*HD]. */
static void attn_head(float* ctx, const int16_t* qh, float sq,
                      const int16_t* kh, float sk,
                      const int16_t* vh, float sv, int head) {
    for (int i = 0; i < TM_S; i++) {
        const int16_t* qi16 = qh + (size_t)i * TM_HD;
        float m = -INFINITY, lsum = 0.0f;
        float acc[TM_HD];
        for (int d = 0; d < TM_HD; d++) acc[d] = 0.0f;
        for (int j = 0; j <= i; j++) {
            const int16_t* kj16 = kh + (size_t)j * TM_HD;
            const int16_t* vj16 = vh + (size_t)j * TM_HD;
            /* q row dequantized once per i (recompute per j here: cheap) */
            TP_BA(TP_ATTN_QK);
            float s = 0.0f;
            for (int d = 0; d < TM_HD; d++)
                s += (float)qi16[d] * sq * (float)kj16[d] * sk;
            s *= TM_ATTN_SCALE;
            TP_EA(TP_ATTN_QK);
            TP_BA(TP_ATTN_EXP);
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
            TP_EA(TP_ATTN_EXP);
            TP_BA(TP_ATTN_PV);
            for (int d = 0; d < TM_HD; d++)
                acc[d] += p * (float)vj16[d] * sv;
            TP_EA(TP_ATTN_PV);
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
    if (xin != g_x) memcpy(g_x, xin, sizeof g_x);
    int fast = (g_mode == TM_MODE_FAST);

    tp_wall_begin();
    for (int l = 0; l < TM_L; l++) {
        /* ---- norm1 ---- */
        TP_B(TP_NORM1);
        tm_layernorm(g_x,
                     W + woff(l, TM_W_BLK_N1W), W + woff(l, TM_W_BLK_N1B),
                     g_buf1, TM_S, TM_D);
        TP_E(TP_NORM1);

        /* ---- attention ----
         * Each head's Q/K/V projection is GEMMed directly into its own
         * final ctx slice g_buf2[i*TM_D + h*TM_HD + d] (rowStride=TM_D),
         * quantized to int16 Q15 head slices (g_qh/g_kh/g_vh + per-head
         * scales, saves 24 KB), then attn_head overwrites that same slice
         * with the head output. No staging aliases another head's slice. */
        {
            const int h_order[TM_H] = {1, 2, 3, 0};
            for (int t = 0; t < TM_H; t++) {
                int h = h_order[t];
                TP_B(TP_QKV);
                if (fast) {
                    /* per-head Q/K/V projections via Q15 x Q12 rows [h*HD:(h+1)*HD] */
                    const int16_t* wq = q12->q[l][0] + (size_t)h * TM_HD * TM_D;
                    const int16_t* wk = q12->q[l][1] + (size_t)h * TM_HD * TM_D;
                    const int16_t* wv = q12->q[l][2] + (size_t)h * TM_HD * TM_D;
                    tm_gemm_q12(g_buf1, wq, q12->ws[l][0],
                                W + woff(l, TM_W_BLK_QB) + h * TM_HD,
                                g_buf2 + h * TM_HD, TM_S, TM_D, TM_HD, TM_D);
                    TP_B(TP_QUANT);
                    g_qs = quant_head(g_buf2 + h * TM_HD, TM_D, g_qh);
                    TP_E(TP_QUANT);
                    tm_gemm_q12(g_buf1, wk, q12->ws[l][1],
                                W + woff(l, TM_W_BLK_KB) + h * TM_HD,
                                g_buf2 + h * TM_HD, TM_S, TM_D, TM_HD, TM_D);
                    TP_B(TP_QUANT);
                    g_ks = quant_head(g_buf2 + h * TM_HD, TM_D, g_kh);
                    TP_E(TP_QUANT);
                    tm_gemm_q12(g_buf1, wv, q12->ws[l][2],
                                W + woff(l, TM_W_BLK_VB) + h * TM_HD,
                                g_buf2 + h * TM_HD, TM_S, TM_D, TM_HD, TM_D);
                    TP_B(TP_QUANT);
                    g_vs = quant_head(g_buf2 + h * TM_HD, TM_D, g_vh);
                    TP_E(TP_QUANT);
                } else {
                    tm_gemm_f32(g_buf1, W + woff(l, TM_W_BLK_QW) + (size_t)h * TM_HD * TM_D,
                                W + woff(l, TM_W_BLK_QB) + h * TM_HD,
                                g_buf2 + h * TM_HD, TM_S, TM_D, TM_HD, TM_D);
                    TP_B(TP_QUANT);
                    g_qs = quant_head(g_buf2 + h * TM_HD, TM_D, g_qh);
                    TP_E(TP_QUANT);
                    tm_gemm_f32(g_buf1, W + woff(l, TM_W_BLK_KW) + (size_t)h * TM_HD * TM_D,
                                W + woff(l, TM_W_BLK_KB) + h * TM_HD,
                                g_buf2 + h * TM_HD, TM_S, TM_D, TM_HD, TM_D);
                    TP_B(TP_QUANT);
                    g_ks = quant_head(g_buf2 + h * TM_HD, TM_D, g_kh);
                    TP_E(TP_QUANT);
                    tm_gemm_f32(g_buf1, W + woff(l, TM_W_BLK_VW) + (size_t)h * TM_HD * TM_D,
                                W + woff(l, TM_W_BLK_VB) + h * TM_HD,
                                g_buf2 + h * TM_HD, TM_S, TM_D, TM_HD, TM_D);
                    TP_B(TP_QUANT);
                    g_vs = quant_head(g_buf2 + h * TM_HD, TM_D, g_vh);
                    TP_E(TP_QUANT);
                }
                TP_E(TP_QKV);
                TP_B(TP_ATTN);
                attn_head(g_buf2, g_qh, g_qs, g_kh, g_ks, g_vh, g_vs, h);
                TP_E(TP_ATTN);
            }
        }

        /* ---- out projection (ctx in g_buf2 -> out into g_buf2 in place) ---- */
        TP_B(TP_OPROJ);
        if (fast) {
            /* reuse g_vh as temporaries: keep ctx, write result to g_buf1 */
            tm_gemm_q12(g_buf2, q12->q[l][3], q12->ws[l][3],
                        W + woff(l, TM_W_BLK_OB), g_buf1, TM_S, TM_D, TM_D, TM_D);
        } else {
            tm_gemm_f32(g_buf2, W + woff(l, TM_W_BLK_OW),
                        W + woff(l, TM_W_BLK_OB), g_buf1, TM_S, TM_D, TM_D, TM_D);
        }
        TP_E(TP_OPROJ);
        /* ---- residual ---- */
        TP_B(TP_RES1);
        tm_add_inplace(g_buf1, g_x, TM_S * TM_D);
        TP_E(TP_RES1);

        /* ---- norm2 ---- */
        TP_B(TP_NORM2);
        tm_layernorm(g_x,
                     W + woff(l, TM_W_BLK_N2W), W + woff(l, TM_W_BLK_N2B),
                     g_buf1, TM_S, TM_D);
        TP_E(TP_NORM2);

        /* ---- FFN ---- */
        TP_B(TP_F1);
        if (fast) {
            tm_gemm_q12(g_buf1, q12->q[l][4], q12->ws[l][4],
                        W + woff(l, TM_W_BLK_F1B), g_buf2, TM_S, TM_D, TM_F, TM_F);
        } else {
            tm_gemm_f32(g_buf1, W + woff(l, TM_W_BLK_F1W),
                        W + woff(l, TM_W_BLK_F1B), g_buf2, TM_S, TM_D, TM_F, TM_F);
        }
        TP_E(TP_F1);
        TP_B(TP_F2);
        TP_B(TP_GELU);
        tm_gelu_inplace(g_buf2, TM_S * TM_F);
        TP_E(TP_GELU);
        if (fast) {
            tm_gemm_q12(g_buf2, q12->q[l][5], q12->ws[l][5],
                        W + woff(l, TM_W_BLK_F2B), g_buf1, TM_S, TM_F, TM_D, TM_D);
        } else {
            tm_gemm_f32(g_buf2, W + woff(l, TM_W_BLK_F2W),
                        W + woff(l, TM_W_BLK_F2B), g_buf1, TM_S, TM_F, TM_D, TM_D);
        }
        TP_E(TP_F2);
        /* ---- residual ---- */
        TP_B(TP_RES2);
        tm_add_inplace(g_buf1, g_x, TM_S * TM_D);
        TP_E(TP_RES2);
    }

    /* ---- final norm ---- */
    TP_B(TP_FINAL);
    tm_layernorm(g_x, W + TM_W_FINALW, W + TM_W_FINALB, g_buf1, TM_S, TM_D);
    TP_E(TP_FINAL);
    tp_wall_end();
    memcpy(yout, g_buf1, sizeof g_buf1);
}
