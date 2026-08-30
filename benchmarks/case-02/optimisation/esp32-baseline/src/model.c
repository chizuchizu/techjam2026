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
#define TM_PROFILE 1

/* The tiled FAST build supplies the same public API from model_tiled.c.
 * Keep the default translation unit unchanged unless that opt-in flag is set;
 * PlatformIO compiles every source in src/, so selecting only one file is not
 * sufficient for the firmware build. */
#ifndef TM_TILED_FORWARD

/* may be overridden (device: Serial.print per line) */
__attribute__((weak)) void tm_prof_emit(const char* line) { (void)line; }

/* ================= on-device per-kernel profiling (TM_PROFILE) ================= */
#ifdef TM_PROFILE
#if defined(ESP32) || defined(ARDUINO_ARCH_ESP32)
#include <esp_timer.h>
#define TM_PROF_NOW() ((int64_t)esp_timer_get_time())
#else
#include <time.h>
static inline int64_t tm_prof_now_host(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000 + (int64_t)ts.tv_nsec / 1000;
}
#define TM_PROF_NOW() tm_prof_now_host()
#endif
#define TM_PROF_SLOTS 15
enum {
    P_NORM1 = 0, P_QKV, P_QUANT, P_ATTN, P_OPROJ, P_RES1,
    P_NORM2, P_F1, P_GELU, P_F2, P_RES2, P_FINAL, P_ATTN_QK, P_ATTN_EXP, P_ATTN_PV
};
static int64_t  p_start[TM_PROF_SLOTS];
static uint64_t p_acc[TM_PROF_SLOTS];
static uint32_t p_cnt[TM_PROF_SLOTS];
static const char* const p_name[TM_PROF_SLOTS] = {
    "norm1", "qkv", "quant", "attn", "oproj", "res1",
    "norm2", "f1", "gelu", "f2", "res2", "final",
    "attn_qk", "attn_exp", "attn_pv"
};
#define PB(i) do { p_start[(i)] = TM_PROF_NOW(); } while (0)
#define PE(i) do { int64_t d = TM_PROF_NOW() - p_start[(i)]; \
                   if (d > 0) { p_acc[(i)] += (uint64_t)d; p_cnt[(i)]++; } } while (0)
static uint64_t prof_total_us = 0;
#define PBT() do { prof_total_us = (uint64_t)TM_PROF_NOW(); } while (0)
#define PET() do { prof_total_us = (uint64_t)TM_PROF_NOW() - prof_total_us; } while (0)
#else
#define PB(i) do {} while (0)
#define PE(i) do {} while (0)
#define PBT() do {} while (0)
#define PET() do {} while (0)
#endif
#ifdef TM_PROFILE
#include <stdio.h>
#endif
void tm_profile_dump(void) {
#ifdef TM_PROFILE
    static char line[160];
    for (int i = 0; i < TM_PROF_SLOTS; i++) {
        if (p_cnt[i] && p_acc[i]) {
            (void)snprintf(line, sizeof line,
                "%s  total_us=%llu n=%u avg_us=%.1f  (%.1f%%)\n",
                p_name[i], (unsigned long long)p_acc[i], p_cnt[i],
                (double)p_acc[i] / (double)p_cnt[i],
                100.0 * (double)p_acc[i] / (double)(prof_total_us ? prof_total_us : 1));
            tm_prof_emit(line);
        }
    }
    (void)snprintf(line, sizeof line, "TOTAL ~ %llu us total_wall\n", prof_total_us);
    tm_prof_emit(line);
#endif
    extern void tm_kbench_dump(void);
    tm_kbench_dump();
}


static int g_mode = TM_MODE_DEFAULT;

void tm_set_mode(int mode) { g_mode = mode; }
int tm_get_mode(void) { return g_mode; }

/* static arena (only one forward at a time). Sized for the C3's 320 KB
 * SRAM: x + buf1 + buf2 (64 KB fp32 each) + per-head q/k/v (fp32 48 KB)
 * + a16 (32 KB Q15 scratch in kernels.c)). g_out was removed: the final
 * LayerNorm writes into g_buf1, exposed via tm_output(). */
/* R1 (FAST): the layer residual is carried as Q15 int16 at fixed scale
 * g_res_sa (real value = q15 * g_res_sa); EXACT uses the fp32 view. One
 * 64 KB buffer, reinterpreted by mode (zero extra SRAM). */
static union { float f[TM_S * TM_D]; int32_t s[TM_S * TM_D]; } g_x;
static float g_res_sa = 1.0f;      /* value per Q15 LSB of the residual */
static float g_buf1[TM_S * TM_D];
static float g_buf2[TM_S * TM_D];
/* per-head Q/K/V stored as int16 (Q15) + a per-head-slice float scale,
 * dequantized on read inside attn_head. Halves the 3 head buffers
 * (48 -> 24 KB). fp32 staging during projection reuses g_buf2. */
static int16_t g_qh[TM_S * TM_HD];
/* FAST-only aliases, zero extra SRAM:
 *   g_ctxq  == g_buf2 int16 view  (attn Q15 ctx, interleaved [i][h*HD+d]);
 *             g_buf2 is free from end-of-qkv till oproj/reuse, and oproj writes g_buf1.
 *   v_all   == g_buf1 int16 view  (all-head V, head-major); needed only until the
 *             last attn; oproj then overwrites g_buf1 (since single core4 writes C).
 *   acc     == g_buf1 int16+32KB (int32 view) head-gemm scratch.
 * One global ctx scale per layer keeps the single K=128 core4 for oproj. */
#define g_ctxq ((int16_t *)(g_buf2))
#define v_all  ((int16_t *)(g_buf1))
#define g_acc  tm_acc_scratch()
static inline int32_t* tm_acc_scratch(void) {
    /* int32 gemm scratch for the (Q,K,V) head projections (FAST phase A).
     * v_all occupies the head of g_buf1: H*HD==D so that is always
     * 2*S*D bytes.  The scratch goes right after v_all while that stays
     * inside g_buf1 (H>=2); for H==1 (HD==D) g_buf1 has no room, so it
     * borrows g_buf2's unused int32 space instead.  Safe there: g_ctxq
     * only writes the low (int16) half of g_buf2, and only after every
     * head's V projection has already consumed the scratch. */
    if ((size_t)(2 * TM_S * TM_H * TM_HD + TM_S * TM_HD * 4) <=
        (size_t)(TM_S * TM_D * 4))
        return (int32_t *)(((unsigned char *)g_buf1) + 2 * TM_S * TM_H * TM_HD);
    return (int32_t *)(g_buf2);
}
static float g_ctx_sa;               /* per-layer global ctx Q15 scale (FAST) */
static float g_vs_h[TM_H];           /* per-head V scales (FAST phase A) */

static int16_t g_kh[TM_S * TM_HD];
static int16_t g_vh[TM_S * TM_HD];
static float g_qs, g_ks, g_vs;   /* dequant scales (amax/32767) per head */

float* tm_input(void)  { return g_x.f; }
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
            float v = row[d] * sa;
            int q;
            if (v >= 0.0f) { q = (int)(v + 0.5f); if (q > TM_QACT_MAX) q = (int)TM_QACT_MAX; }
            else           { q = (int)(v - 0.5f); if (q < -TM_QACT_MAX) q = -(int)TM_QACT_MAX; }
            drow[d] = (int16_t)q;
        }
    }
    return amax / (float)TM_QACT_MAX;
}

/* Integer causal attention for one head over all S tokens (two-pass).
 *
 * qh/kh/vh are int16 Q15 [S,HD] with per-buffer dequant scales.
 *   QK:  s = (sum_d qi16[d]*kj16[d]) * (sq*sk*TM_ATTN_SCALE)
 *        dot accumulated in int64 (Q15*Q15 ~ 2^30; 32 terms can reach 2^35)
 *        -> one multiply per (i,j) pair instead of 32 fp32 soft-float ops.
 *   softmax (two-pass, exact max, no online rescale):
 *        FAST: integer exp via 513-entry LUT on [-10,0] (TFLM-style linear
 *              interp).  diff*M>>15 maps the logit range onto the LUT.
 *        EXACT: fp32 expf kept (reference fidelity).
 *   PV:  p15 (Q15, int) * v Q15 accumulated in int64; one fp32 dequant at
 *        row end o[d] = acc[d] * (sv/lsum15).
 * Result written into ctx[i*D + head*HD]. */
/* attention exp LUT: shared with the multiboard shard (kernels.c) */
#define g_exp_lut tm_attn_exp_lut

static float g_qkv_sa;                  /* its Q15 scale */
static int64_t g_attn_score[TM_S];   /* per-row dot products (1 KB) */

static void attn_head(float* ctx, const int16_t* qh, float sq,
                      const int16_t* kh, float sk,
                      const int16_t* vh, float sv, int head) {
    const float gsc = sq * sk * TM_ATTN_SCALE;   /* > 0 */
    const uint64_t g_exp_c = (uint64_t)(gsc * 6553.5f * 4294967296.0f + 0.5f); /* *2^32 */
    const int fast = (g_mode == TM_MODE_FAST);
    static int32_t g_p15[TM_S];
    const float g_sctx = fast ? g_ctx_sa : 0.0f;
    for (int i = 0; i < TM_S; i++) {
        const int16_t* qi16 = qh + (size_t)i * TM_HD;
        int32_t maxL = INT32_MIN, maxH = INT32_MIN;
        PB(P_ATTN_QK);
        /* j-unrolled QK: 4 causal rows share the qi16 elements, each with its
         * own int32 hi/lo limb accumulator (identical integer math to the int64
         * dot, per-dot accumulation order unchanged -> bit-exact vs opt18). */
        {
            int j = 0;
            const int nj = i + 1;
            for (; j + 3 < nj; j += 4) {
                const int16_t* k0 = kh + (size_t)(j + 0) * TM_HD;
                const int16_t* k1 = kh + (size_t)(j + 1) * TM_HD;
                const int16_t* k2 = kh + (size_t)(j + 2) * TM_HD;
                const int16_t* k3 = kh + (size_t)(j + 3) * TM_HD;
                int32_t L0 = 0, H0 = 0, L1 = 0, H1 = 0, L2 = 0, H2 = 0, L3 = 0, H3 = 0;
                int d = 0;
                for (; d + 1 < TM_HD; d += 2) {
                    const int32_t q0 = (int32_t)(int16_t)qi16[d];
                    const int32_t q1 = (int32_t)(int16_t)qi16[d + 1];
                    { int32_t p = q0 * (int32_t)(int16_t)k0[d] + q1 * (int32_t)(int16_t)k0[d+1];
                      uint32_t up = (uint32_t)L0 + (uint32_t)p;
                      H0 += (int32_t)(up < (uint32_t)L0) + (int32_t)(p < 0 ? -1 : 0); L0 = (int32_t)up; }
                    { int32_t p = q0 * (int32_t)(int16_t)k1[d] + q1 * (int32_t)(int16_t)k1[d+1];
                      uint32_t up = (uint32_t)L1 + (uint32_t)p;
                      H1 += (int32_t)(up < (uint32_t)L1) + (int32_t)(p < 0 ? -1 : 0); L1 = (int32_t)up; }
                    { int32_t p = q0 * (int32_t)(int16_t)k2[d] + q1 * (int32_t)(int16_t)k2[d+1];
                      uint32_t up = (uint32_t)L2 + (uint32_t)p;
                      H2 += (int32_t)(up < (uint32_t)L2) + (int32_t)(p < 0 ? -1 : 0); L2 = (int32_t)up; }
                    { int32_t p = q0 * (int32_t)(int16_t)k3[d] + q1 * (int32_t)(int16_t)k3[d+1];
                      uint32_t up = (uint32_t)L3 + (uint32_t)p;
                      H3 += (int32_t)(up < (uint32_t)L3) + (int32_t)(p < 0 ? -1 : 0); L3 = (int32_t)up; }
                }
                for (; d < TM_HD; d++) {
                    const int32_t q0 = (int32_t)(int16_t)qi16[d];
                    { int32_t p = q0 * (int32_t)(int16_t)k0[d];
                      uint32_t up = (uint32_t)L0 + (uint32_t)p;
                      H0 += (int32_t)(up < (uint32_t)L0) + (int32_t)(p < 0 ? -1 : 0); L0 = (int32_t)up; }
                    { int32_t p = q0 * (int32_t)(int16_t)k1[d];
                      uint32_t up = (uint32_t)L1 + (uint32_t)p;
                      H1 += (int32_t)(up < (uint32_t)L1) + (int32_t)(p < 0 ? -1 : 0); L1 = (int32_t)up; }
                    { int32_t p = q0 * (int32_t)(int16_t)k2[d];
                      uint32_t up = (uint32_t)L2 + (uint32_t)p;
                      H2 += (int32_t)(up < (uint32_t)L2) + (int32_t)(p < 0 ? -1 : 0); L2 = (int32_t)up; }
                    { int32_t p = q0 * (int32_t)(int16_t)k3[d];
                      uint32_t up = (uint32_t)L3 + (uint32_t)p;
                      H3 += (int32_t)(up < (uint32_t)L3) + (int32_t)(p < 0 ? -1 : 0); L3 = (int32_t)up; }
                }
                g_attn_score[j+0] = ((int64_t)(int32_t)H0 << 32) | (uint32_t)L0;
                if (H0 > maxH || (H0 == maxH && (uint32_t)L0 > (uint32_t)maxL)) { maxH = H0; maxL = L0; }
                g_attn_score[j+1] = ((int64_t)(int32_t)H1 << 32) | (uint32_t)L1;
                if (H1 > maxH || (H1 == maxH && (uint32_t)L1 > (uint32_t)maxL)) { maxH = H1; maxL = L1; }
                g_attn_score[j+2] = ((int64_t)(int32_t)H2 << 32) | (uint32_t)L2;
                if (H2 > maxH || (H2 == maxH && (uint32_t)L2 > (uint32_t)maxL)) { maxH = H2; maxL = L2; }
                g_attn_score[j+3] = ((int64_t)(int32_t)H3 << 32) | (uint32_t)L3;
                if (H3 > maxH || (H3 == maxH && (uint32_t)L3 > (uint32_t)maxL)) { maxH = H3; maxL = L3; }
            }
            for (; j < nj; j++) {
                const int16_t* kj16 = kh + (size_t)j * TM_HD;
                int32_t L = 0, H = 0;
                int d = 0;
                for (; d + 1 < TM_HD; d += 2) {
                    int32_t p = (int32_t)(int16_t)qi16[d] * (int32_t)kj16[d]
                              + (int32_t)(int16_t)qi16[d+1] * (int32_t)kj16[d+1];
                    uint32_t up = (uint32_t)L + (uint32_t)p;
                    H += (int32_t)(up < (uint32_t)L) + (int32_t)(p < 0 ? -1 : 0);
                    L = (int32_t)up;
                }
                for (; d < TM_HD; d++) {
                    int32_t p = (int32_t)(int16_t)qi16[d] * (int32_t)kj16[d];
                    uint32_t up = (uint32_t)L + (uint32_t)p;
                    H += (int32_t)(up < (uint32_t)L) + (int32_t)(p < 0 ? -1 : 0);
                    L = (int32_t)up;
                }
                g_attn_score[j] = ((int64_t)(int32_t)H << 32) | (uint32_t)L;
                if (H > maxH || (H == maxH && (uint32_t)L > (uint32_t)maxL)) { maxH = H; maxL = L; }
            }
        }
        PE(P_ATTN_QK);
        const int64_t maxs = ((int64_t)(int32_t)maxH << 32) | (uint32_t)maxL;
        /* exp pass: p15 per j (kept in g_p15) + lsum */
        int32_t lsum15 = 0;
        PB(P_ATTN_EXP);
        for (int j = 0; j <= i; j++) {
            int64_t diff = g_attn_score[j] - maxs;   /* <= 0 */
            int32_t p15;
            if (fast) {
                /* all-integer LUT index.  y16 = trunc(diff*gsc*6553.5) with a
                 * fixed-point multiplier g_exp_c = round(gsc*6553.5*2^16):
                 *   mag = |diff| <= 65535/(gsc*6553.5) ~ 56  ->  mag*g_exp_c
                 *   < 2^33, uint64-safe.  y16 = -(mag*g_exp_c >> 16) with the
                 *   same trunc-to-zero the fp32 path produced. */
                uint64_t mag = (uint64_t)(0LL - diff);
                uint64_t y = (mag * g_exp_c) >> 32;      /* |y16| */
                if (y > 65535u) y = 65535u;
                int32_t idx = (int32_t)(y >> 7);
                int32_t off = (int32_t)(y & 127u);
                p15 = (int32_t)g_exp_lut[idx] +
                      ((((int32_t)g_exp_lut[idx+1] - (int32_t)g_exp_lut[idx]) * off + 64) >> 7);
            } else {
                float p = expf((float)diff * gsc);
                p15 = (int32_t)(p * 32768.0f + 0.5f);
            }
            g_p15[j] = p15;
            lsum15 += p15;
        }
        PE(P_ATTN_EXP);
        float* o = ctx + (size_t)i * TM_D + head * TM_HD;
        PB(P_ATTN_PV);
        if (fast) {
            /* int32 PV via per-row weight rescale to sum 32767:
             * ctx is a weighted average, so c = sum_j p'_j*v_j with p' scaled so
             * sum(p') ~= 32767 fits |c| <= 32767^2 in int32 -> a single 16x16 mul
             * per MAC, no 64-bit accum, no mulh.  ctx_q = round(c * sv*g_sctx/32767),
             * one fp32 mul + ftrunc per output, same as before. */
            const int32_t QM = (int32_t)TM_QACT_MAX;
            int32_t f15 = (int32_t)((int64_t)QM * QM /
                                    (lsum15 > 0 ? (int64_t)lsum15 : 1));
            for (int j = 0; j <= i; j++)
                g_p15[j] = ((int64_t)g_p15[j] * f15 + 0x4000) >> 15;
            /* integer ctx epilogue: ctx_q = round(c0 * rot) with rot = m/2^sh,
             * m in [2^28, 2^30) -> exact int64 product, no fp32 per element.
             * |real_ctx| <= ctxmax => |ctx_q| <= QACT, so sh >= 1 in practice. */
            float t = sv * g_sctx / TM_QACT_MAX;
            int sh = 0;
            while (t < 268435456.0f && sh < 63) { t += t; sh++; }
            while (t >= 1073741824.0f && sh > 1) { t *= 0.5f; sh--; }
            int32_t m = (int32_t)t;
            if (sh < 1) { m = (int32_t)(t * 0.5f); sh = 1; }
            for (int db = 0; db < TM_HD; db += 8) {
                int32_t c0 = 0, c1 = 0, c2 = 0, c3 = 0, c4 = 0, c5 = 0, c6 = 0, c7 = 0;
                for (int j = 0; j <= i; j++) {
                    const int16_t* vj = vh + (size_t)j * TM_HD;
                    int32_t p = g_p15[j];
                    c0 += p * (int32_t)vj[db+0];
                    c1 += p * (int32_t)vj[db+1];
                    c2 += p * (int32_t)vj[db+2];
                    c3 += p * (int32_t)vj[db+3];
                    c4 += p * (int32_t)vj[db+4];
                    c5 += p * (int32_t)vj[db+5];
                    c6 += p * (int32_t)vj[db+6];
                    c7 += p * (int32_t)vj[db+7];
                }
                int16_t* oq = g_ctxq + (size_t)i * TM_D + head * TM_HD + db;
                oq[0] = (int16_t)(((int64_t)c0 * m + (1LL << (sh - 1))) >> sh);
                oq[1] = (int16_t)(((int64_t)c1 * m + (1LL << (sh - 1))) >> sh);
                oq[2] = (int16_t)(((int64_t)c2 * m + (1LL << (sh - 1))) >> sh);
                oq[3] = (int16_t)(((int64_t)c3 * m + (1LL << (sh - 1))) >> sh);
                oq[4] = (int16_t)(((int64_t)c4 * m + (1LL << (sh - 1))) >> sh);
                oq[5] = (int16_t)(((int64_t)c5 * m + (1LL << (sh - 1))) >> sh);
                oq[6] = (int16_t)(((int64_t)c6 * m + (1LL << (sh - 1))) >> sh);
                oq[7] = (int16_t)(((int64_t)c7 * m + (1LL << (sh - 1))) >> sh);
            }
        } else {
            float inv = (lsum15 > 0) ? sv / (float)lsum15 : 0.0f;
            for (int db = 0; db < TM_HD; db += 4) {
                int64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
                for (int j = 0; j <= i; j++) {
                    const int16_t* vj = vh + (size_t)j * TM_HD;
                    int32_t p = g_p15[j];
                    a0 += (int64_t)p * (int32_t)vj[db];
                    a1 += (int64_t)p * (int32_t)vj[db+1];
                    a2 += (int64_t)p * (int32_t)vj[db+2];
                    a3 += (int64_t)p * (int32_t)vj[db+3];
                }
                o[db]   = (float)a0 * inv;
                o[db+1] = (float)a1 * inv;
                o[db+2] = (float)a2 * inv;
                o[db+3] = (float)a3 * inv;
            }
        }
        PE(P_ATTN_PV);
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

static void tm_head_order(int* ho) {
    /* per-head visit order (only affects buffer locality, not correctness;
     * keeps the tuned {1,2,3,0} for the case-2 H==4 shape) */
    for (int i = 0; i < TM_H; i++) ho[i] = i;
    if (TM_H == 4) { const int f[4] = {1, 2, 3, 0}; for (int i = 0; i < 4; i++) ho[i] = f[i]; }
}

void tm_forward(const float* xin, float* yout,
                const float* W, const TMQ12Weights* q12) {
    int fast = (g_mode == TM_MODE_FAST);
    if (fast) {
        /* R1: residual carried as Q15 at fixed scale: full-scale 16.0
         * (measured residual magnitude stays within ~±4.6 over all layers
         * and seeds; 16 gives 3.5x headroom while keeping LSB precision
         * ~2.4e-4, well under the 2e-3 output gate). */
        g_res_sa = TM_RES_SPAN / 2147483648.0f;
    /* g_res_sa: value per int32 unit of the residual; span/2^31 => numerically exact */
        tm_quant_res_i32(xin, g_x.s, TM_S * TM_D, g_res_sa);
    } else {
        if (xin != g_x.f) memcpy(g_x.f, xin, sizeof g_x.f);
    }
    PBT();

    for (int l = 0; l < TM_L; l++) {
        /* ---- norm1 (FAST) ---- */
        PB(P_NORM1);
        if (fast) {
            /* fused LN -> a16 Q15: stats + O(D) amax bound + single
             * normalize/round pass. Removes the fp32 LN output, the separate
             * amax scan, and the separate quantize pass for qkv. */
            g_qkv_sa = tm_bn_q15_res(g_x.s, g_res_sa,
                                 W + woff(l, TM_W_BLK_N1W), W + woff(l, TM_W_BLK_N1B),
                                 tm_gemm_a16(), TM_S, TM_D);
        } else {
            tm_layernorm(g_x.f,
                         W + woff(l, TM_W_BLK_N1W), W + woff(l, TM_W_BLK_N1B),
                         g_buf1, TM_S, TM_D);
        }
        PE(P_NORM1);
        /* ---- attention ----
         * FAST two-phase: (A) project all 4 heads' V into v_all (g_buf1 int16,
         * head-major) and fold each head's |v| max into ONE global ctx scale
         * |ctx| <= sv_h * max|v_q15|_h, so attention can emit Q15 ctx directly
         * (no oproj A-quant) with a single K=128 core4.  (B) per head: Q/K
         * projection + attend.  v_all is dead once the last attn runs, then
         * oproj overwrites g_buf1; ctx lives in g_ctxq (g_buf2 alias, free here).
         * EXACT keeps the original interleaved fp32 flow. */
        if (fast) {
            const float sainv = 1.0f / g_qkv_sa;
            float ctx_max = 0.0f;
            PB(P_QKV);
            for (int h = 0; h < TM_H; h++) {
                PB(P_QUANT);
                g_vs_h[h] = tm_gemm_head_q15(
                            tm_gemm_a16(), sainv, q12->q[l][2] + (size_t)h * TM_HD * TM_D,
                            q12->ws[l][2], W + woff(l, TM_W_BLK_VB) + h * TM_HD,
                            g_acc, v_all + (size_t)h * TM_S * TM_HD, TM_D);
                PE(P_QUANT);
                int16_t* vh = v_all + (size_t)h * TM_S * TM_HD;
                int32_t vmax = 0;
                for (int32_t j = 0; j < TM_S * TM_HD; j++) {
                    int32_t a = vh[j] < 0 ? -vh[j] : vh[j];
                    if (a > vmax) vmax = a;
                }
                float bnd = g_vs_h[h] * (float)(vmax ? vmax : 1);
                if (bnd > ctx_max) ctx_max = bnd;
            }
            PE(P_QKV);
            g_ctx_sa = TM_QACT_MAX / (ctx_max > 0.0f ? ctx_max : 1.0f) * 0.9999f;
            int h_order[TM_H]; tm_head_order(h_order);
            for (int t = 0; t < TM_H; t++) {
                int h = h_order[t];
                PB(P_QKV);
                PB(P_QUANT);
                g_qs = tm_gemm_head_q15(tm_gemm_a16(), sainv,
                                    q12->q[l][0] + (size_t)h * TM_HD * TM_D,
                                    q12->ws[l][0], W + woff(l, TM_W_BLK_QB) + h * TM_HD,
                                    g_acc, g_qh, TM_D);
                PE(P_QUANT);
                PB(P_QUANT);
                g_ks = tm_gemm_head_q15(tm_gemm_a16(), sainv,
                                    q12->q[l][1] + (size_t)h * TM_HD * TM_D,
                                    q12->ws[l][1], W + woff(l, TM_W_BLK_KB) + h * TM_HD,
                                    g_acc, g_kh, TM_D);
                PE(P_QUANT);
                PE(P_QKV);
                PB(P_ATTN);
                attn_head(g_buf2, g_qh, g_qs, g_kh, g_ks,
                          v_all + (size_t)h * TM_S * TM_HD, g_vs_h[h], h);
                PE(P_ATTN);
            }
        } else {
            int h_order[TM_H]; tm_head_order(h_order);
            for (int t = 0; t < TM_H; t++) {
                int h = h_order[t];
                PB(P_QKV);
                tm_gemm_f32(g_buf1, W + woff(l, TM_W_BLK_QW) + (size_t)h * TM_HD * TM_D,
                            W + woff(l, TM_W_BLK_QB) + h * TM_HD,
                            g_buf2 + h * TM_HD, TM_S, TM_D, TM_HD, TM_D);
                PB(P_QUANT);
                g_qs = quant_head(g_buf2 + h * TM_HD, TM_D, g_qh);
                PE(P_QUANT);
                tm_gemm_f32(g_buf1, W + woff(l, TM_W_BLK_KW) + (size_t)h * TM_HD * TM_D,
                            W + woff(l, TM_W_BLK_KB) + h * TM_HD,
                            g_buf2 + h * TM_HD, TM_S, TM_D, TM_HD, TM_D);
                PB(P_QUANT);
                g_ks = quant_head(g_buf2 + h * TM_HD, TM_D, g_kh);
                PE(P_QUANT);
                tm_gemm_f32(g_buf1, W + woff(l, TM_W_BLK_VW) + (size_t)h * TM_HD * TM_D,
                            W + woff(l, TM_W_BLK_VB) + h * TM_HD,
                            g_buf2 + h * TM_HD, TM_S, TM_D, TM_HD, TM_D);
                PB(P_QUANT);
                g_vs = quant_head(g_buf2 + h * TM_HD, TM_D, g_vh);
                PE(P_QUANT);
                PE(P_QKV);
                PB(P_ATTN);
                attn_head(g_buf2, g_qh, g_qs, g_kh, g_ks, g_vh, g_vs, h);
                PE(P_ATTN);
            }
        }

        /* ---- out projection ---- */
                PB(P_OPROJ);
        if (fast) {
            /* ctx already Q15 (global scale) in g_ctxq == g_buf2 -> raw core,
             * epilogue fuses the output into the Q15 residual in place */
            tm_gemm_core5_resid(g_ctxq, 1.0f / g_ctx_sa, q12->q[l][3], q12->ws[l][3],
                          W + woff(l, TM_W_BLK_OB), g_x.s, g_res_sa,
                          TM_S, TM_D, TM_D, TM_D);
        } else {
            tm_gemm_f32(g_buf2, W + woff(l, TM_W_BLK_OW),
                        W + woff(l, TM_W_BLK_OB), g_buf1, TM_S, TM_D, TM_D, TM_D);
        }
        PE(P_OPROJ);

        /* ---- residual ---- */
        PB(P_RES1);
        if (!fast) tm_add_inplace(g_buf1, g_x.f, TM_S * TM_D);   /* R1: fused into core */
        PE(P_RES1);

        /* ---- norm2 (FAST: fused LN -> a16 Q15 via tight amax bound; the
         * fp32 LN write and f1's separate amax+quant passes disappear) ---- */
        float sa_ffn = TM_QACT_MAX;
        PB(P_NORM2);
        if (fast) {
            sa_ffn = tm_bn_q15_res(g_x.s, g_res_sa,
                               W + woff(l, TM_W_BLK_N2W), W + woff(l, TM_W_BLK_N2B),
                               tm_gemm_a16(), TM_S, TM_D);
        } else {
            tm_layernorm(g_x.f,
                         W + woff(l, TM_W_BLK_N2W), W + woff(l, TM_W_BLK_N2B),
                         g_buf1, TM_S, TM_D);
        }
        PE(P_NORM2);

        /* ---- FFN ---- */
        float sa2 = 0.0f;
        PB(P_F1);
        if (fast) {
            /* Q15-output FFN1: integer epilogue (bias folded, int amax -> sa2).
             * Returns sa2 = QMAX/(amax*g) (inverse), like tm_gemm_amax. */
            int16_t* a16_in = tm_gemm_a16();
            sa2 = tm_gemm_core5_q15(a16_in, 1.0f / sa_ffn,
                                          q12->q[l][4], q12->ws[l][4],
                                          W + woff(l, TM_W_BLK_F1B),
                                          (int32_t *)g_buf2, a16_in,
                                          TM_S, TM_D, TM_F, TM_F);
        } else {
            tm_gemm_f32(g_buf1, W + woff(l, TM_W_BLK_F1W),
                        W + woff(l, TM_W_BLK_F1B), g_buf2, TM_S, TM_D, TM_F, TM_F);
        }
        PE(P_F1);

        PB(P_F2);
        PB(P_GELU);
        if (fast) {
            /* a16 now holds Q15 (f1 output); integer GELU in place. */
            tm_gelu_q15_lut(tm_gemm_a16(), TM_S * TM_F, TM_QACT_MAX / sa2);
            PE(P_GELU);
            tm_gemm_core5_resid(tm_gemm_a16(), 1.0f / sa2, q12->q[l][5], q12->ws[l][5],
                         W + woff(l, TM_W_BLK_F2B), g_x.s, g_res_sa, TM_S, TM_F, TM_D, TM_D);
        } else {
            tm_gelu_inplace(g_buf2, TM_S * TM_F);
            PE(P_GELU);
            tm_gemm_f32(g_buf2, W + woff(l, TM_W_BLK_F2W),
                        W + woff(l, TM_W_BLK_F2B), g_buf1, TM_S, TM_F, TM_D, TM_D);
        }
        PE(P_F2);

        /* ---- residual ---- */
        PB(P_RES2);
        if (!fast) tm_add_inplace(g_buf1, g_x.f, TM_S * TM_D);   /* R1: fused into core */
        PE(P_RES2);
    }

    /* ---- final norm ---- */
    PB(P_FINAL);
    if (fast) {
        tm_ln_final_res(g_x.s, g_res_sa, W + TM_W_FINALW, W + TM_W_FINALB,
                        g_buf1, TM_S, TM_D);
    } else {
        tm_layernorm(g_x.f, W + TM_W_FINALW, W + TM_W_FINALB, g_buf1, TM_S, TM_D);
    }
    memcpy(yout, g_buf1, sizeof g_buf1);
    PE(P_FINAL);
    PET();
}

#endif /* !TM_TILED_FORWARD */
