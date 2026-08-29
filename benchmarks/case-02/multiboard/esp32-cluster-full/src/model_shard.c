/*
 * model_shard.c - see model_shard.h for the partition.
 *
 * Everything outside attention is the single-board FAST path applied to
 * TMS_SLOC rows instead of TM_S, so the kernels are shared verbatim with
 * ../../optimisation/esp32-baseline/src/kernels.c.
 */
#include "model_shard.h"
#include "kernels.h"

#include <string.h>

/* Same flat fp32 layout as model.c (only the norm and bias blocks are read
 * on the FAST path; the projection matrices come from the Q12 blob). */
static uint32_t woff(int layer, int blk) {
    uint32_t o = (uint32_t)layer * TM_W_LAYER_FLOATS;
    switch (blk) {
        case TM_W_BLK_N1W: return o + 0;
        case TM_W_BLK_N1B: return o + 1 * TM_D;
        case TM_W_BLK_QB:  return o + 2 * TM_D + 1 * TM_D * TM_D;
        case TM_W_BLK_KB:  return o + 3 * TM_D + 2 * TM_D * TM_D;
        case TM_W_BLK_VB:  return o + 4 * TM_D + 3 * TM_D * TM_D;
        case TM_W_BLK_OB:  return o + 5 * TM_D + 4 * TM_D * TM_D;
        case TM_W_BLK_N2W: return o + 6 * TM_D + 4 * TM_D * TM_D;
        case TM_W_BLK_N2B: return o + 7 * TM_D + 4 * TM_D * TM_D;
        case TM_W_BLK_F1B: return o + 8 * TM_D + 4 * TM_D * TM_D + TM_F * TM_D;
        case TM_W_BLK_F2B: return o + 8 * TM_D + 4 * TM_D * TM_D + TM_F * TM_D
                                    + TM_F + TM_D * TM_F;
        default: return 0;
    }
}

/* exp LUT shared with model.c's attention: 513 entries of 32767*exp(-i/51.2),
 * i.e. y-units of 1/6553.5 nat with 7 bits of linear interpolation. */
extern const int16_t tm_attn_exp_lut[513];

/* Choose (M, sh) with M = round(gsc * 6553.5 * 2^sh) in [2^26, 2^27) so that
 * logit = (dot * M) >> sh lands in LUT y-units for any per-source scale.
 * |dot| <= TM_HD * 32767^2 = 3.44e10 and M < 1.34e8, so the int64 product
 * stays below 4.7e18 < INT64_MAX. */
static void logit_scale(float gsc, int64_t* out_m, int* out_sh) {
    float c = gsc * 6553.5f;
    int n = 0;
    if (!(c > 0.0f)) { *out_m = 0; *out_sh = 0; return; }
    while (c < 67108864.0f && n < 62) { c *= 2.0f; n++; }
    while (c >= 134217728.0f && n > 0) { c *= 0.5f; n--; }
    *out_m = (int64_t)c;
    *out_sh = n;
}

/* per-row scratch (rows are processed one at a time, node-sequentially) */
static int64_t s_l_own [TMS_SLOC];
static int64_t s_l_peer[TMS_SLOC];
static int32_t s_p_own [TMS_SLOC];
static int32_t s_p_peer[TMS_SLOC];

static inline int32_t exp_lut_q15(int64_t diff /* <= 0 */) {
    uint64_t y = (uint64_t)(-diff);
    if (y > 65535u) y = 65535u;
    int32_t idx = (int32_t)(y >> 7);
    int32_t off = (int32_t)(y & 127u);
    return (int32_t)tm_attn_exp_lut[idx] +
           ((((int32_t)tm_attn_exp_lut[idx + 1] - (int32_t)tm_attn_exp_lut[idx])
             * off + 64) >> 7);
}

static void attn_head_shard(TMShard* st, int h, float sq) {
    const int node = st->node;
    const int peer = TMS_NODES - 1 - node;
    const size_t blk = (size_t)TMS_SLOC * TM_HD;
    const size_t base = (size_t)h * TMS_HEAD_ELEMS;

    const int16_t* qh = st->qh;
    const int16_t* ko = st->kv_own  + base;
    const int16_t* vo = st->kv_own  + base + blk;
    const int16_t* kp = st->kv_peer + base;
    const int16_t* vp = st->kv_peer + base + blk;

    const float svo = st->hdr_own [2 * h + 1];
    const float svp = st->hdr_peer[2 * h + 1];

    int64_t mo, mp; int sho, shp;
    logit_scale(sq * st->hdr_own [2 * h] * TM_ATTN_SCALE, &mo, &sho);
    logit_scale(sq * st->hdr_peer[2 * h] * TM_ATTN_SCALE, &mp, &shp);

    for (int a = 0; a < TMS_SLOC; a++) {
        const int gi = a * TMS_NODES + node;   /* global row index */
        const int n_own = a + 1;               /* own j: global a'*N+node <= gi */
        const int t = gi - peer;
        int n_peer = (t < 0) ? 0 : (t / TMS_NODES) + 1;
        if (n_peer > TMS_SLOC) n_peer = TMS_SLOC;

        const int16_t* qi = qh + (size_t)a * TM_HD;
        int64_t maxl = INT64_MIN;

        for (int b = 0; b < n_own; b++) {
            const int16_t* kb = ko + (size_t)b * TM_HD;
            int64_t dot = 0;
            for (int d = 0; d + 1 < TM_HD; d += 2)
                dot += (int32_t)qi[d] * (int32_t)kb[d]
                     + (int32_t)qi[d + 1] * (int32_t)kb[d + 1];
            int64_t l = (dot * mo) >> sho;
            s_l_own[b] = l;
            if (l > maxl) maxl = l;
        }
        for (int b = 0; b < n_peer; b++) {
            const int16_t* kb = kp + (size_t)b * TM_HD;
            int64_t dot = 0;
            for (int d = 0; d + 1 < TM_HD; d += 2)
                dot += (int32_t)qi[d] * (int32_t)kb[d]
                     + (int32_t)qi[d + 1] * (int32_t)kb[d + 1];
            int64_t l = (dot * mp) >> shp;
            s_l_peer[b] = l;
            if (l > maxl) maxl = l;
        }

        int32_t lsum = 0;
        for (int b = 0; b < n_own; b++)  { int32_t p = exp_lut_q15(s_l_own[b]  - maxl); s_p_own[b]  = p; lsum += p; }
        for (int b = 0; b < n_peer; b++) { int32_t p = exp_lut_q15(s_l_peer[b] - maxl); s_p_peer[b] = p; lsum += p; }

        const float ino = (lsum > 0) ? svo / (float)lsum : 0.0f;
        const float inp = (lsum > 0) ? svp / (float)lsum : 0.0f;
        float* o = st->b2 + (size_t)a * TM_D + h * TM_HD;

        for (int db = 0; db < TM_HD; db += 4) {
            int64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            for (int b = 0; b < n_own; b++) {
                const int16_t* vj = vo + (size_t)b * TM_HD;
                int32_t p = s_p_own[b];
                a0 += (int64_t)p * (int32_t)vj[db];
                a1 += (int64_t)p * (int32_t)vj[db + 1];
                a2 += (int64_t)p * (int32_t)vj[db + 2];
                a3 += (int64_t)p * (int32_t)vj[db + 3];
            }
            int64_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            for (int b = 0; b < n_peer; b++) {
                const int16_t* vj = vp + (size_t)b * TM_HD;
                int32_t p = s_p_peer[b];
                c0 += (int64_t)p * (int32_t)vj[db];
                c1 += (int64_t)p * (int32_t)vj[db + 1];
                c2 += (int64_t)p * (int32_t)vj[db + 2];
                c3 += (int64_t)p * (int32_t)vj[db + 3];
            }
            o[db]     = (float)a0 * ino + (float)c0 * inp;
            o[db + 1] = (float)a1 * ino + (float)c1 * inp;
            o[db + 2] = (float)a2 * ino + (float)c2 * inp;
            o[db + 3] = (float)a3 * ino + (float)c3 * inp;
        }
    }
}

/* Q12 blob parser (same layout as model.c; the shard build does not link
 * model.c because its static arena is sized for the whole sequence). */
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

void tm_shard_init(TMShard* st, int node,
                   const float* W, const TMQ12Weights* q12) {
    memset(st, 0, sizeof *st);
    st->node = node;
    st->W = W;
    st->q12 = q12;
}

float* tm_shard_input(TMShard* st)  { return st->x;  }
float* tm_shard_output(TMShard* st) { return st->b1; }

void tm_shard_layer_pre(TMShard* st, int l) {
    const float* W = st->W;
    const TMQ12Weights* q12 = st->q12;


    tm_layernorm(st->x, W + woff(l, TM_W_BLK_N1W), W + woff(l, TM_W_BLK_N1B),
                 st->b1, TMS_SLOC, TM_D);

    /* one shared Q15 quantization of the norm output for all 12 head GEMMs */
    const float sa = tm_gemm_amax(st->b1, TMS_SLOC * TM_D);
    int16_t* aq = tm_gemm_a16();
    tm_gemm_quantA_into(st->b1, TMS_SLOC * TM_D, aq, sa);
    const float sainv = 1.0f / sa;
    st->sa_inv = sainv;
    /* b1 is consumed above; reuse it as the int32 accumulator scratch */
    int32_t* acc = (int32_t*)st->b1;

    for (int h = 0; h < TM_H; h++) {
        const size_t base = (size_t)h * TMS_HEAD_ELEMS;
        st->hdr_own[2 * h] = tm_gemm_head_q15_m(
            aq, sainv, q12->q[l][1] + (size_t)h * TM_HD * TM_D, q12->ws[l][1],
            W + woff(l, TM_W_BLK_KB) + h * TM_HD, acc,
            st->kv_own + base, TMS_SLOC, TM_D);
        st->hdr_own[2 * h + 1] = tm_gemm_head_q15_m(
            aq, sainv, q12->q[l][2] + (size_t)h * TM_HD * TM_D, q12->ws[l][2],
            W + woff(l, TM_W_BLK_VB) + h * TM_HD, acc,
            st->kv_own + base + (size_t)TMS_SLOC * TM_HD, TMS_SLOC, TM_D);
        /* head h's chunk is now complete: hand it to the link immediately */
        if (st->kv_sent) st->kv_sent(st->hook_ctx, h);
    }
}

void tm_shard_layer_post(TMShard* st, int l) {
    const float* W = st->W;
    const TMQ12Weights* q12 = st->q12;

    /* aq still holds this layer's quantized norm1 output (nothing between
     * pre() and here touches it), so each head's queries can be projected
     * just before they are consumed. */
    const int16_t* aq = tm_gemm_a16();
    int32_t* acc = (int32_t*)st->b1;
    for (int h = 0; h < TM_H; h++) {
        const float sq = tm_gemm_head_q15_m(
            aq, st->sa_inv, q12->q[l][0] + (size_t)h * TM_HD * TM_D, q12->ws[l][0],
            W + woff(l, TM_W_BLK_QB) + h * TM_HD, acc, st->qh, TMS_SLOC, TM_D);
        /* the query projection above runs while the peer chunk is in flight */
        if (st->kv_needed) st->kv_needed(st->hook_ctx, h);
        attn_head_shard(st, h, sq);
    }

    tm_gemm_q12(st->b2, q12->q[l][3], q12->ws[l][3],
                W + woff(l, TM_W_BLK_OB), st->b1, TMS_SLOC, TM_D, TM_D, TM_D);
    tm_add_inplace(st->b1, st->x, TMS_SLOC * TM_D);

    tm_layernorm(st->x, W + woff(l, TM_W_BLK_N2W), W + woff(l, TM_W_BLK_N2B),
                 st->b1, TMS_SLOC, TM_D);

    tm_gemm_q12(st->b1, q12->q[l][4], q12->ws[l][4],
                W + woff(l, TM_W_BLK_F1B), st->b2, TMS_SLOC, TM_D, TM_F, TM_F);

    /* fused GELU + f2 quantize (identical to the single-board FAST path) */
    const float sa2 = tm_gemm_amax(st->b2, TMS_SLOC * TM_F);
    int16_t* a2 = tm_gemm_a16();
    tm_gemm_quantA_into(st->b2, TMS_SLOC * TM_F, a2, sa2);
    tm_gelu_q15_lut(a2, TMS_SLOC * TM_F, TM_QACT_MAX / sa2);
    tm_gemm_core4(a2, 1.0f / sa2, q12->q[l][5], q12->ws[l][5],
                  W + woff(l, TM_W_BLK_F2B), st->b1, TMS_SLOC, TM_F, TM_D, TM_D);

    tm_add_inplace(st->b1, st->x, TMS_SLOC * TM_D);
}

void tm_shard_final(TMShard* st) {
    tm_layernorm(st->x, st->W + TM_W_FINALW, st->W + TM_W_FINALB,
                 st->b1, TMS_SLOC, TM_D);
}
