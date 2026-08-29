/*
 * model_shard.c - see model_shard.h for the partition and the wire layout.
 *
 * Everything outside attention is the single-board opt23 FAST path applied to
 * TMS_SLOC rows instead of TM_S, so the kernels are shared verbatim with
 * ../../optimisation/esp32-baseline/src/kernels.c.
 */
#include "model_shard.h"
#include "kernels.h"

#include <math.h>   /* TM_ATTN_SCALE uses sqrtf */
#include <string.h>

/* kernels.c routes its microbenchmark output through this hook, which model.c
 * normally provides; the shard build does not link model.c, so supply the same
 * weak stub (the firmware overrides it to print over serial). */
__attribute__((weak)) void tm_prof_emit(const char* line) { (void)line; }

/* Same flat fp32 layout as model.c (only the norm and bias blocks are read on
 * the FAST path; the projection matrices come from the Q12 blob). */
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

/* Choose (m, sh) with m = round(v * 2^sh) in [2^28, 2^30) so an int32 operand
 * times m stays inside int64. Shared by the logit and the context epilogues. */
static void fixed_mul(float v, int64_t* out_m, int* out_sh) {
    if (!(v > 0.0f)) { *out_m = 0; *out_sh = 1; return; }
    float t = v;
    int sh = 0;
    while (t < 268435456.0f && sh < 62) { t += t; sh++; }
    while (t >= 1073741824.0f && sh > 1) { t *= 0.5f; sh--; }
    *out_m = (int64_t)t;
    *out_sh = sh;
}

/* exp LUT shared with the single-board attention (kernels.c). */
static inline int32_t exp_lut_q15(int64_t diff /* <= 0 */) {
    uint64_t y = (uint64_t)(-diff);
    if (y > 65535u) y = 65535u;
    int32_t idx = (int32_t)(y >> 7);
    int32_t off = (int32_t)(y & 127u);
    return (int32_t)tm_attn_exp_lut[idx] +
           ((((int32_t)tm_attn_exp_lut[idx + 1] - (int32_t)tm_attn_exp_lut[idx])
             * off + 64) >> 7);
}

/* per-row scratch (rows are processed one at a time, node-sequentially) */
static int64_t s_l_own [TMS_SLOC];
static int64_t s_l_peer[TMS_SLOC];
static int32_t s_p_own [TMS_SLOC];
static int32_t s_p_peer[TMS_SLOC];

/* Causal attention for one head over the merged K/V, writing Q15 context at
 * the global ctx scale. Mirrors model.c's attn_head FAST epilogue, with the
 * two sources kept in their own dequant scales throughout. */
static void attn_head_shard(TMShard* st, int h, float sq) {
    const int node = st->node;
    const int peer = TMS_NODES - 1 - node;
    const size_t blk  = (size_t)TMS_SLOC * TM_HD;
    const size_t base = (size_t)h * TMS_HEAD_ELEMS;

    const int16_t* qh = st->qh;
    const int16_t* ko = st->big.kv.own  + base;
    const int16_t* vo = st->big.kv.own  + base + blk;
    const int16_t* kp = st->big.kv.peer + base;
    const int16_t* vp = st->big.kv.peer + base + blk;

    const float svo = st->hdr_own [h];
    const float svp = st->hdr_peer[h];

    /* logit multipliers: score * 6553.5 lands in the exp LUT's y-units */
    int64_t mo, mp; int sho, shp;
    fixed_mul(sq * st->sk_own [h] * TM_ATTN_SCALE * 6553.5f, &mo, &sho);
    fixed_mul(sq * st->sk_peer[h] * TM_ATTN_SCALE * 6553.5f, &mp, &shp);

    /* Context epilogue: ctx_q = (c_own*co + c_peer*cp + rnd) >> csh, so the two
     * sources are combined with a single rounding. Both multipliers are put at
     * the common shift of whichever source has the larger scale (the smaller
     * of the two shifts); the other is shifted down to match, which keeps both
     * under 2^30 and the int64 products well inside range. */
    int64_t co, cp; int csh_o, csh_p;
    fixed_mul(svo * st->ctx_sa / TM_QACT_MAX, &co, &csh_o);
    fixed_mul(svp * st->ctx_sa / TM_QACT_MAX, &cp, &csh_p);
    const int csh = (csh_o < csh_p) ? csh_o : csh_p;
    {
        const int dropo = csh_o - csh, dropp = csh_p - csh;
        co = (dropo >= 63) ? 0 : (co >> dropo);
        cp = (dropp >= 63) ? 0 : (cp >> dropp);
    }
    const int64_t crnd = (csh > 0) ? (1LL << (csh - 1)) : 0;

    for (int a = 0; a < TMS_SLOC; a++) {
        const int gi = a * TMS_NODES + node;   /* global row index */
        const int n_own = a + 1;               /* own j: a'*N+node <= gi */
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

        /* rescale the weights to sum ~32767 so the PV product fits int32 */
        const int32_t QM = (int32_t)TM_QACT_MAX;
        const int32_t f15 = (int32_t)((int64_t)QM * QM / (lsum > 0 ? (int64_t)lsum : 1));
        for (int b = 0; b < n_own; b++)  s_p_own[b]  = (int32_t)(((int64_t)s_p_own[b]  * f15 + 0x4000) >> 15);
        for (int b = 0; b < n_peer; b++) s_p_peer[b] = (int32_t)(((int64_t)s_p_peer[b] * f15 + 0x4000) >> 15);

        int16_t* oq = st->ctxq + (size_t)a * TM_D + h * TM_HD;
        for (int db = 0; db < TM_HD; db += 8) {
            int32_t o0=0,o1=0,o2=0,o3=0,o4=0,o5=0,o6=0,o7=0;
            for (int b = 0; b < n_own; b++) {
                const int16_t* vj = vo + (size_t)b * TM_HD;
                const int32_t p = s_p_own[b];
                o0 += p*(int32_t)vj[db+0]; o1 += p*(int32_t)vj[db+1];
                o2 += p*(int32_t)vj[db+2]; o3 += p*(int32_t)vj[db+3];
                o4 += p*(int32_t)vj[db+4]; o5 += p*(int32_t)vj[db+5];
                o6 += p*(int32_t)vj[db+6]; o7 += p*(int32_t)vj[db+7];
            }
            int32_t q0=0,q1=0,q2=0,q3=0,q4=0,q5=0,q6=0,q7=0;
            for (int b = 0; b < n_peer; b++) {
                const int16_t* vj = vp + (size_t)b * TM_HD;
                const int32_t p = s_p_peer[b];
                q0 += p*(int32_t)vj[db+0]; q1 += p*(int32_t)vj[db+1];
                q2 += p*(int32_t)vj[db+2]; q3 += p*(int32_t)vj[db+3];
                q4 += p*(int32_t)vj[db+4]; q5 += p*(int32_t)vj[db+5];
                q6 += p*(int32_t)vj[db+6]; q7 += p*(int32_t)vj[db+7];
            }
            #define TMS_CTX(k, cown, cpeer) do {                                \
                int64_t v = (int64_t)(cown) * co + (int64_t)(cpeer) * cp + crnd; \
                v >>= csh;                                                       \
                if (v >  32767) v =  32767;                                      \
                if (v < -32767) v = -32767;                                      \
                oq[db + (k)] = (int16_t)v;                                       \
            } while (0)
            TMS_CTX(0, o0, q0); TMS_CTX(1, o1, q1);
            TMS_CTX(2, o2, q2); TMS_CTX(3, o3, q3);
            TMS_CTX(4, o4, q4); TMS_CTX(5, o5, q5);
            TMS_CTX(6, o6, q6); TMS_CTX(7, o7, q7);
            #undef TMS_CTX
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
    /* residual carried as int32 at a fixed span, exactly as model.c does */
    st->res_sa = TM_RES_SPAN / 2147483648.0f;
}

void tm_shard_load(TMShard* st, const float* local_rows) {
    st->res_sa = TM_RES_SPAN / 2147483648.0f;
    tm_quant_res_i32(local_rows, st->x, TMS_SLOC * TM_D, st->res_sa);
}

float* tm_shard_output(TMShard* st) { return st->big.out; }

void tm_shard_layer_pre(TMShard* st, int l) {
    const float* W = st->W;
    const TMQ12Weights* q12 = st->q12;
    const size_t blk = (size_t)TMS_SLOC * TM_HD;

    /* fused LayerNorm -> Q15 straight into the shared activation scratch */
    st->qkv_sa = tm_bn_q15_res(st->x, st->res_sa,
                               W + woff(l, TM_W_BLK_N1W), W + woff(l, TM_W_BLK_N1B),
                               tm_gemm_a16(), TMS_SLOC, TM_D);
    const float sainv = 1.0f / st->qkv_sa;
    const int16_t* aq = tm_gemm_a16();

    /* V first for every head: the global context scale needs all four bounds,
     * and the peer needs them before it can start attending. */
    for (int h = 0; h < TM_H; h++) {
        int16_t* vh = st->big.kv.own + (size_t)h * TMS_HEAD_ELEMS + blk;
        const float sv = tm_gemm_head_q15_m(
            aq, sainv, q12->q[l][2] + (size_t)h * TM_HD * TM_D, q12->ws[l][2],
            W + woff(l, TM_W_BLK_VB) + h * TM_HD, st->acc, vh, TMS_SLOC, TM_D);
        int32_t vmax = 0;
        for (int32_t j = 0; j < TMS_SLOC * TM_HD; j++) {
            int32_t v = vh[j] < 0 ? -vh[j] : vh[j];
            if (v > vmax) vmax = v;
        }
        st->hdr_own[h] = sv;                                  /* dequant scale */
        st->hdr_own[TM_H + h] = sv * (float)(vmax ? vmax : 1);/* |V| bound     */
    }
    if (st->hdr_sent) st->hdr_sent(st->hook_ctx);

    for (int h = 0; h < TM_H; h++) {
        st->sk_own[h] = tm_gemm_head_q15_m(
            aq, sainv, q12->q[l][1] + (size_t)h * TM_HD * TM_D, q12->ws[l][1],
            W + woff(l, TM_W_BLK_KB) + h * TM_HD, st->acc,
            st->big.kv.own + (size_t)h * TMS_HEAD_ELEMS, TMS_SLOC, TM_D);
        /* head h's chunk is complete: hand it to the link immediately */
        if (st->kv_sent) st->kv_sent(st->hook_ctx, h);
    }
}

void tm_shard_layer_post(TMShard* st, int l) {
    const float* W = st->W;
    const TMQ12Weights* q12 = st->q12;

    /* |ctx| is bounded by the largest |V| over any row a token may attend to,
     * which spans both boards - hence the early header exchange. */
    if (st->hdr_needed) st->hdr_needed(st->hook_ctx);
    float ctx_max = 0.0f;
    for (int h = 0; h < TM_H; h++) {
        if (st->hdr_own [TM_H + h] > ctx_max) ctx_max = st->hdr_own [TM_H + h];
        if (st->hdr_peer[TM_H + h] > ctx_max) ctx_max = st->hdr_peer[TM_H + h];
    }
    st->ctx_sa = TM_QACT_MAX / (ctx_max > 0.0f ? ctx_max : 1.0f) * 0.9999f;

    /* aq still holds this layer's quantized norm1 output (nothing between
     * pre() and here touches it), so each head's queries are projected just
     * before they are consumed - which also covers the peer chunk's flight. */
    const int16_t* aq = tm_gemm_a16();
    const float sainv = 1.0f / st->qkv_sa;
    for (int h = 0; h < TM_H; h++) {
        const float sq = tm_gemm_head_q15_m(
            aq, sainv, q12->q[l][0] + (size_t)h * TM_HD * TM_D, q12->ws[l][0],
            W + woff(l, TM_W_BLK_QB) + h * TM_HD, st->acc, st->qh, TMS_SLOC, TM_D);
        if (st->kv_needed) st->kv_needed(st->hook_ctx, h);
        attn_head_shard(st, h, sq);
    }

    /* O projection, folded straight into the fixed-point residual */
    tm_gemm_core5_resid(st->ctxq, 1.0f / st->ctx_sa, q12->q[l][3], q12->ws[l][3],
                        W + woff(l, TM_W_BLK_OB), st->x, st->res_sa,
                        TMS_SLOC, TM_D, TM_D, TM_D);

    const float sa_ffn = tm_bn_q15_res(st->x, st->res_sa,
                                       W + woff(l, TM_W_BLK_N2W),
                                       W + woff(l, TM_W_BLK_N2B),
                                       tm_gemm_a16(), TMS_SLOC, TM_D);

    /* kv is dead from here until the next pre(), so FFN1's int32 scratch
     * borrows it (see the union in model_shard.h). */
    int16_t* a16 = tm_gemm_a16();
    const float sa2 = tm_gemm_core5_q15(a16, 1.0f / sa_ffn,
                                        q12->q[l][4], q12->ws[l][4],
                                        W + woff(l, TM_W_BLK_F1B),
                                        st->big.ffn_scratch, a16,
                                        TMS_SLOC, TM_D, TM_F, TM_F);
    tm_gelu_q15_lut(a16, TMS_SLOC * TM_F, TM_QACT_MAX / sa2);
    tm_gemm_core5_resid(a16, 1.0f / sa2, q12->q[l][5], q12->ws[l][5],
                        W + woff(l, TM_W_BLK_F2B), st->x, st->res_sa,
                        TMS_SLOC, TM_F, TM_D, TM_D);
}

void tm_shard_final(TMShard* st) {
    tm_ln_final_res(st->x, st->res_sa, st->W + TM_W_FINALW, st->W + TM_W_FINALB,
                    st->big.out, TMS_SLOC, TM_D);
}
