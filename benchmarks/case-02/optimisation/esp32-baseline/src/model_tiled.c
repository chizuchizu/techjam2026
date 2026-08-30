/*
 * model_tiled.c - FAST-only, sequentially tiled transformer forward.
 *
 * This is an opt-in replacement for model.c (define TM_TILED_FORWARD).  The
 * full-sequence residual remains resident, because every layer updates it in
 * place.  Everything else is either Q15 or limited to TM_TILE_ROWS:
 *
 *   residual / input / final output       S*D int32/fp32 (one union)
 *   attention context                    S*D int16
 *   current head K and V                 2*S*HD int16
 *   projection / FFN accumulator         TILE*max(D,F) int32
 *   normalized activation / FFN1         kernels.c TILE*D int16 scratch
 *   current Q tile                       TILE*HD int16
 *
 * K and V are initially quantized per row tile, then requantized to one scale
 * for the complete head.  Q may retain a tile-local scale: every score in one
 * query row has the same Q scale, so causal softmax remains well-defined.
 * A preliminary V-only sweep finds a safe, shared context scale across heads;
 * this lets the complete context feed one full output projection.
 */
#ifdef TM_TILED_FORWARD

#include "model.h"
#include "kernels.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifndef TM_TILE_ROWS
#define TM_TILE_ROWS 16
#endif

#if TM_TILE_ROWS <= 0 || (TM_S % TM_TILE_ROWS) != 0
#error "TM_TILE_ROWS must be positive and divide TM_S"
#endif
#if (TM_TILE_ROWS % 8) != 0
#error "TM_TILE_ROWS must be a multiple of 8 for the FAST head kernel"
#endif
#if TM_F > TM_D
#error "The shared kernels.c a16 tile requires TM_F <= TM_D"
#endif
#if TM_F > TM_D
#define TM_TILE_WIDTH TM_F
#else
#define TM_TILE_WIDTH TM_D
#endif

/* TM_A16_ROWS is consumed by kernels.c.  Firmware and host builds selecting
 * this model must define TM_A16_ROWS=TM_TILE_ROWS as a build flag so that the
 * shared activation workspace is physically reduced as intended. */

static int g_mode = TM_MODE_FAST;

typedef union {
    float f[TM_S * TM_D];
    int32_t q[TM_S * TM_D];
} TMResidual;

static TMResidual g_x;
static int16_t g_ctxq[TM_S * TM_D];
static int16_t g_kh[TM_S * TM_HD];
static int16_t g_vh[TM_S * TM_HD];
static int32_t g_scratch[TM_TILE_ROWS * TM_TILE_WIDTH];
static int16_t g_head_out[TM_TILE_ROWS * TM_HD];
static int64_t g_scores[TM_S];
static int32_t g_probs[TM_S];

static const float g_res_scale = TM_RES_SPAN / 2147483648.0f;

__attribute__((weak)) void tm_prof_emit(const char* line) { (void)line; }
void tm_profile_dump(void) {
    extern void tm_kbench_dump(void);
    tm_kbench_dump();
}

/* A tiled build deliberately contains only the validated FAST arithmetic. */
void tm_set_mode(int mode) {
    (void)mode;
    g_mode = TM_MODE_FAST;
}
int tm_get_mode(void) { return g_mode; }

float* tm_input(void) { return g_x.f; }
float* tm_output(void) { return g_x.f; }

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

void tm_scan_q12(const void* blob, TMQ12Weights* out) {
    const uint8_t* p = (const uint8_t*)blob;
    for (int l = 0; l < TM_L; l++) {
        for (int m = 0; m < 6; m++) {
            uint32_t count;
            float wscale;
            memcpy(&count, p, 4); p += 4;
            memcpy(&wscale, p, 4); p += 4;
            out->ws[l][m] = wscale;
            out->q[l][m] = (const int16_t*)p;
            p += (size_t)count * 2;
        }
    }
}

static float norm1_tile(int layer, int row0, const float* W) {
    return tm_bn_q15_res(g_x.q + (size_t)row0 * TM_D, g_res_scale,
                         W + woff(layer, TM_W_BLK_N1W),
                         W + woff(layer, TM_W_BLK_N1B),
                         tm_gemm_a16(), TM_TILE_ROWS, TM_D);
}

/* Convert q*local_scale to q'*global_scale.  The projection kernels choose
 * local_scale from an exact tile amax; global_scale is max(local_scale), so
 * this cannot overflow Q15 apart from harmless fp rounding at the endpoint. */
static void requant_tiles(int16_t* data, const float* scales, float global_scale,
                          int cols) {
    const int Q = (int)TM_QACT_MAX;
    for (int t = 0; t < TM_S / TM_TILE_ROWS; t++) {
        float ratio = scales[t] / global_scale;
        int16_t* p = data + (size_t)t * TM_TILE_ROWS * cols;
        for (int i = 0; i < TM_TILE_ROWS * cols; i++) {
            float z = (float)p[i] * ratio;
            int q = (int)(z + (z >= 0.0f ? 0.5f : -0.5f));
            if (q > Q) q = Q;
            if (q < -Q) q = -Q;
            p[i] = (int16_t)q;
        }
    }
}

/* Find max |V| for every head without retaining V.  This extra projection
 * sweep is what permits one shared context quantization scale while storing
 * only one head's V at a time. */
static float scan_context_bound(int layer, const float* W,
                                const TMQ12Weights* q12) {
    float head_max[TM_H];
    for (int h = 0; h < TM_H; h++) head_max[h] = 0.0f;

    for (int row0 = 0; row0 < TM_S; row0 += TM_TILE_ROWS) {
        float sa = norm1_tile(layer, row0, W);
        for (int h = 0; h < TM_H; h++) {
            float sv = tm_gemm_head_q15_m(
                tm_gemm_a16(), 1.0f / sa,
                q12->q[layer][2] + (size_t)h * TM_HD * TM_D,
                q12->ws[layer][2],
                W + woff(layer, TM_W_BLK_VB) + h * TM_HD,
                g_scratch, g_head_out, TM_TILE_ROWS, TM_D);
            float vmax = sv * TM_QACT_MAX;
            if (vmax > head_max[h]) head_max[h] = vmax;
        }
    }
    float ctx_max = 0.0f;
    for (int h = 0; h < TM_H; h++)
        if (head_max[h] > ctx_max) ctx_max = head_max[h];
    return ctx_max > 0.0f ? ctx_max : 1.0f;
}

/* Build the current head's complete K and V using tile-local normalization
 * and projection scales, then put each buffer onto one head-global scale. */
static void build_kv_head(int layer, int head, const float* W,
                          const TMQ12Weights* q12, float* sk, float* sv) {
    float k_scales[TM_S / TM_TILE_ROWS];
    float v_scales[TM_S / TM_TILE_ROWS];
    float kg = 0.0f, vg = 0.0f;
    int t = 0;
    for (int row0 = 0; row0 < TM_S; row0 += TM_TILE_ROWS, t++) {
        float sa = norm1_tile(layer, row0, W);
        k_scales[t] = tm_gemm_head_q15_m(
            tm_gemm_a16(), 1.0f / sa,
            q12->q[layer][1] + (size_t)head * TM_HD * TM_D,
            q12->ws[layer][1],
            W + woff(layer, TM_W_BLK_KB) + head * TM_HD,
            g_scratch, g_kh + (size_t)row0 * TM_HD,
            TM_TILE_ROWS, TM_D);
        v_scales[t] = tm_gemm_head_q15_m(
            tm_gemm_a16(), 1.0f / sa,
            q12->q[layer][2] + (size_t)head * TM_HD * TM_D,
            q12->ws[layer][2],
            W + woff(layer, TM_W_BLK_VB) + head * TM_HD,
            g_scratch, g_vh + (size_t)row0 * TM_HD,
            TM_TILE_ROWS, TM_D);
        if (k_scales[t] > kg) kg = k_scales[t];
        if (v_scales[t] > vg) vg = v_scales[t];
    }
    if (!(kg > 0.0f)) kg = 1.0f / TM_QACT_MAX;
    if (!(vg > 0.0f)) vg = 1.0f / TM_QACT_MAX;
    requant_tiles(g_kh, k_scales, kg, TM_HD);
    requant_tiles(g_vh, v_scales, vg, TM_HD);
    *sk = kg;
    *sv = vg;
}

/* Integer causal attention for one query row.  This is the scalar schedule
 * from model.c, with Q addressed in a tile and K/V retained for one head. */
static void attend_row(const int16_t* q, float sq, float sk, float sv,
                       float ctx_quant, int row, int head) {
    const float score_scale = sq * sk * TM_ATTN_SCALE;
    const uint64_t exp_c =
        (uint64_t)(score_scale * 6553.5f * 4294967296.0f + 0.5f);
    int64_t max_score = INT64_MIN;

    for (int j = 0; j <= row; j++) {
        const int16_t* k = g_kh + (size_t)j * TM_HD;
        int64_t dot = 0;
        for (int d = 0; d < TM_HD; d++)
            dot += (int32_t)q[d] * (int32_t)k[d];
        g_scores[j] = dot;
        if (dot > max_score) max_score = dot;
    }

    int32_t prob_sum = 0;
    for (int j = 0; j <= row; j++) {
        uint64_t mag = (uint64_t)(max_score - g_scores[j]);
        uint64_t y = (mag * exp_c) >> 32;
        if (y > 65535u) y = 65535u;
        int32_t idx = (int32_t)(y >> 7);
        int32_t off = (int32_t)(y & 127u);
        int32_t p = (int32_t)tm_attn_exp_lut[idx] +
            ((((int32_t)tm_attn_exp_lut[idx + 1] -
               (int32_t)tm_attn_exp_lut[idx]) * off + 64) >> 7);
        g_probs[j] = p;
        prob_sum += p;
    }

    const int32_t Q = (int32_t)TM_QACT_MAX;
    int32_t f15 = (int32_t)((int64_t)Q * Q /
                            (prob_sum > 0 ? (int64_t)prob_sum : 1));
    for (int j = 0; j <= row; j++)
        g_probs[j] = (int32_t)(((int64_t)g_probs[j] * f15 + 0x4000) >> 15);

    float scale = sv * ctx_quant / TM_QACT_MAX;
    int shift = 0;
    while (scale < 268435456.0f && shift < 63) {
        scale += scale;
        shift++;
    }
    while (scale >= 1073741824.0f && shift > 1) {
        scale *= 0.5f;
        shift--;
    }
    int32_t mult = (int32_t)scale;
    if (shift < 1) {
        mult = (int32_t)(scale * 0.5f);
        shift = 1;
    }

    int16_t* out = g_ctxq + (size_t)row * TM_D + head * TM_HD;
    for (int d = 0; d < TM_HD; d++) {
        int32_t acc = 0;
        for (int j = 0; j <= row; j++)
            acc += g_probs[j] * (int32_t)g_vh[(size_t)j * TM_HD + d];
        int64_t z = (int64_t)acc * mult + (1LL << (shift - 1));
        int64_t qctx = z >> shift;
        if (qctx > Q) qctx = Q;
        if (qctx < -Q) qctx = -Q;
        out[d] = (int16_t)qctx;
    }
}

static void run_attention_head(int layer, int head, float sk, float sv,
                               float ctx_quant, const float* W,
                               const TMQ12Weights* q12) {
    for (int row0 = 0; row0 < TM_S; row0 += TM_TILE_ROWS) {
        float sa = norm1_tile(layer, row0, W);
        float sq = tm_gemm_head_q15_m(
            tm_gemm_a16(), 1.0f / sa,
            q12->q[layer][0] + (size_t)head * TM_HD * TM_D,
            q12->ws[layer][0],
            W + woff(layer, TM_W_BLK_QB) + head * TM_HD,
            g_scratch, g_head_out, TM_TILE_ROWS, TM_D);
        const int16_t* qt = g_head_out;
        for (int r = 0; r < TM_TILE_ROWS; r++)
            attend_row(qt + (size_t)r * TM_HD, sq, sk, sv,
                       ctx_quant, row0 + r, head);
    }
}

static void ffn_tiles(int layer, const float* W, const TMQ12Weights* q12) {
    for (int row0 = 0; row0 < TM_S; row0 += TM_TILE_ROWS) {
        int32_t* residual = g_x.q + (size_t)row0 * TM_D;
        int16_t* act = tm_gemm_a16();
        float sa = tm_bn_q15_res(
            residual, g_res_scale,
            W + woff(layer, TM_W_BLK_N2W),
            W + woff(layer, TM_W_BLK_N2B),
            act, TM_TILE_ROWS, TM_D);
        float sa2 = tm_gemm_core5_q15(
            act, 1.0f / sa, q12->q[layer][4], q12->ws[layer][4],
            W + woff(layer, TM_W_BLK_F1B), g_scratch, act,
            TM_TILE_ROWS, TM_D, TM_F, TM_F);
        tm_gelu_q15_lut(act, TM_TILE_ROWS * TM_F, TM_QACT_MAX / sa2);
        tm_gemm_core5_resid(
            act, 1.0f / sa2, q12->q[layer][5], q12->ws[layer][5],
            W + woff(layer, TM_W_BLK_F2B), residual, g_res_scale,
            TM_TILE_ROWS, TM_F, TM_D, TM_D);
    }
}

void tm_forward(const float* xin, float* yout,
                const float* W, const TMQ12Weights* q12) {
    tm_quant_res_i32(xin, g_x.q, TM_S * TM_D, g_res_scale);

    for (int layer = 0; layer < TM_L; layer++) {
        float ctx_max = scan_context_bound(layer, W, q12);
        float ctx_quant = TM_QACT_MAX / ctx_max * 0.9999f;

        for (int head = 0; head < TM_H; head++) {
            float sk, sv;
            build_kv_head(layer, head, W, q12, &sk, &sv);
            run_attention_head(layer, head, sk, sv, ctx_quant, W, q12);
        }

        tm_gemm_core5_resid(
            g_ctxq, 1.0f / ctx_quant, q12->q[layer][3], q12->ws[layer][3],
            W + woff(layer, TM_W_BLK_OB), g_x.q, g_res_scale,
            TM_S, TM_D, TM_D, TM_D);
        ffn_tiles(layer, W, q12);
    }

    tm_ln_final_res(g_x.q, g_res_scale,
                    W + TM_W_FINALW, W + TM_W_FINALB,
                    g_x.f, TM_S, TM_D);
    if (yout != g_x.f)
        memcpy(yout, g_x.f, sizeof g_x.f);
}

#endif /* TM_TILED_FORWARD */
