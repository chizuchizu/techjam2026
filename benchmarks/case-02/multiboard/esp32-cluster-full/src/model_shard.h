/*
 * model_shard.h - sequence-sharded case-2 transformer forward.
 *
 * Partition: global token row i is owned by node (i % TMS_NODES); its local
 * index is i / TMS_NODES. Every operator in the case-2 body is per-token
 * (LayerNorm, Q/K/V/O projections, residuals, FFN, GELU, final norm) except
 * causal attention, which needs the keys and values of every j <= i. So one
 * layer is:
 *
 *   pre(l)   fused LayerNorm -> Q15, then the local V and K projections
 *   <exchange the per-head K/V chunks with the peer>
 *   post(l)  per head: local Q projection, then causal attention over the
 *            merged K/V; then O proj folded into the residual, LayerNorm 2,
 *            FFN, second residual - all local rows only
 *
 * Interleaving rows by parity (rather than splitting the sequence in half)
 * balances the causal attention triangle: node 0 accumulates sum over even i
 * of (i+1) = 4096 score rows, node 1 sum over odd i of (i+1) = 4160.
 *
 * This tracks the single-board FAST path in ../../optimisation/esp32-baseline
 * (opt23): the residual stream is carried as int32 fixed point, both
 * LayerNorms are fused into the Q15 quantize, attention emits Q15 context at
 * one global scale, and the O/FFN2 projections fold straight into the
 * residual. Only TM_MODE_FAST is implemented here.
 *
 * Two details the split forces that the single-board path does not have:
 *
 *  - The global context scale bounds |ctx| by the largest |V| over every row a
 *    token may attend to, which spans both boards. Each node therefore ships
 *    its per-head V bound in an early header, ahead of the bulk payload, so
 *    the scale is known before the first head's attention runs.
 *  - The two K/V halves keep their own Q15 dequant scales. Rather than
 *    requantizing one side to the other's, scores are converted into a common
 *    logit fixed-point domain per source and the PV product is accumulated
 *    per source before a single combined rounding, so the shard is never less
 *    accurate than the single-board path.
 */
#ifndef TM_MODEL_SHARD_H
#define TM_MODEL_SHARD_H

#include <stddef.h>

#include "tm_config.h"
#include "model.h"

#define TMS_NODES 2
#define TMS_SLOC  (TM_S / TMS_NODES)          /* 64 local rows per node */

/* Wire layout, one layer:
 *   header  [float sv[TM_H]][float vbound[TM_H]]            (early, 32 B)
 *   chunk h [float sk_h][int16 K_h[SLOC][HD]][int16 V_h[SLOC][HD]]
 * so head h is consumable once TMS_HDR_BYTES + (h+1)*TMS_CHUNK_BYTES have
 * arrived, and later heads can still be in flight.
 */
#define TMS_HEAD_ELEMS  (2 * TMS_SLOC * TM_HD)         /* K then V, one head */
#define TMS_KV_ELEMS    (TM_H * TMS_HEAD_ELEMS)
#define TMS_KV_BYTES    (TMS_KV_ELEMS * 2)             /* 32768 B */
#define TMS_HDR_FLOATS  (2 * TM_H)                     /* sv[H] then vbound[H] */
#define TMS_HDR_BYTES   (TMS_HDR_FLOATS * 4)           /* 32 B */
#define TMS_CHUNK_BYTES (4 + TMS_HEAD_ELEMS * 2)       /* 8196 B per head */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int                 node;      /* 0 or 1 */
    const float*        W;         /* flat fp32 weights (biases + norms used) */
    const TMQ12Weights* q12;

    int32_t x   [TMS_SLOC * TM_D];        /* residual stream, fixed point 32 KB */
    int16_t ctxq[TMS_SLOC * TM_D];        /* Q15 attention context       16 KB */
    int16_t qh  [TMS_SLOC * TM_HD];       /* one head's queries           4 KB */
    int32_t acc [TMS_SLOC * TM_HD];       /* head-GEMM accumulator        8 KB */

    /* kv_own/kv_peer are dead from the last attention of a layer until the
     * next pre(), so the FFN1 int32 scratch and the fp32 output share them. */
    union {
        struct {
            int16_t own [TMS_KV_ELEMS];   /*                             32 KB */
            int16_t peer[TMS_KV_ELEMS];   /*                             32 KB */
        } kv;
        int32_t ffn_scratch[TMS_SLOC * TM_F];          /*                32 KB */
        float   out[TMS_SLOC * TM_D];                  /*                32 KB */
    } big;

    float   res_sa;                       /* value per int32 residual unit */
    float   qkv_sa;                       /* Q15 scale of the norm1 output */
    float   ctx_sa;                       /* global Q15 context scale      */
    float   sk_own [TM_H];
    float   hdr_own [TMS_HDR_FLOATS];     /* sv[H] then vbound[H], local   */
    float   sk_peer[TM_H];
    float   hdr_peer[TMS_HDR_FLOATS];     /* sv[H] then vbound[H], peer    */

    /* Streaming hooks (NULL on the host driver). hdr_sent fires once the V
     * bounds are known; kv_sent(h) the moment head h's chunk is complete;
     * hdr_needed / kv_needed must not return until the peer's part landed. */
    void  (*hdr_sent)(void* ctx);
    void  (*kv_sent)(void* ctx, int head);
    void  (*hdr_needed)(void* ctx);
    void  (*kv_needed)(void* ctx, int head);
    void*   hook_ctx;
} TMShard;

void tm_shard_init(TMShard* st, int node,
                   const float* W, const TMQ12Weights* q12);

/* Scatter the node's local input rows in, gather its output rows out. */
void   tm_shard_load(TMShard* st, const float* local_rows);
float* tm_shard_output(TMShard* st);

/* Wire segments for head h (peer selects the receive side). */
static inline void tm_shard_chunk(TMShard* st, int h, int peer,
                                  void** sk, void** kv) {
    *sk = (peer ? st->sk_peer : st->sk_own) + h;
    *kv = (peer ? st->big.kv.peer : st->big.kv.own) + (size_t)h * TMS_HEAD_ELEMS;
}
static inline void* tm_shard_hdr(TMShard* st, int peer) {
    return peer ? st->hdr_peer : st->hdr_own;
}

void tm_shard_layer_pre (TMShard* st, int l);
void tm_shard_layer_post(TMShard* st, int l);
void tm_shard_final(TMShard* st);

#ifdef __cplusplus
}
#endif

#endif /* TM_MODEL_SHARD_H */
