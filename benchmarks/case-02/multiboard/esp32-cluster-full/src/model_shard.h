/*
 * model_shard.h - sequence-sharded case-2 transformer forward.
 *
 * Partition: global token row i is owned by node (i % TMS_NODES); its local
 * index is i / TMS_NODES. Every operator in the case-2 body is per-token
 * (LayerNorm, Q/K/V/O projections, residuals, FFN, GELU, final norm) except
 * causal attention, which needs the keys and values of every j <= i. So one
 * layer is:
 *

 *   pre(l)   local LayerNorm + local K/V projections (Q15)
 *   <exchange kv_own <-> kv_peer, TMS_HDR_BYTES + TMS_KV_BYTES each way>
 *   post(l)  per head: local Q projection then causal attention over the
 *            merged K/V; then O proj, residual, LayerNorm 2, FFN, residual -
 *            all local rows only
 *
 * Q is projected in post() rather than pre() so only one head of queries is
 * ever resident (4 KB instead of 16 KB), which leaves the WiFi stack the heap
 * it needs; the quantized LayerNorm output it reads survives the exchange.
 *
 * Interleaving rows by parity (rather than splitting the sequence in half)
 * balances the causal attention triangle: node 0 accumulates sum over even i
 * of (i+1) = 4096 score rows, node 1 sum over odd i of (i+1) = 4160.
 *
 * The two K/V halves keep their own Q15 dequant scales; scores are converted
 * into a common logit fixed-point domain per source instead of requantizing,
 * so the shard is never less accurate than the single-board path.
 *
 * Only TM_MODE_FAST is implemented here (the EXACT soft-float path exists in
 * the single-board build for reference checks).
 */
#ifndef TM_MODEL_SHARD_H
#define TM_MODEL_SHARD_H

#include <stddef.h>

#include "tm_config.h"
#include "model.h"

#define TMS_NODES 2
#define TMS_SLOC  (TM_S / TMS_NODES)          /* 64 local rows per node */
#define TMS_FD    (TM_F > TM_D ? TM_F : TM_D)

/* Wire layout, one layer: four per-head chunks, each
 *   [float sk][float sv][int16 K[TMS_SLOC][TM_HD]][int16 V[TMS_SLOC][TM_HD]]
 * so head h is complete on the wire after (h+1)*TMS_CHUNK_BYTES bytes and can
 * be consumed while later heads are still in flight. */
#define TMS_HEAD_ELEMS (2 * TMS_SLOC * TM_HD)          /* K then V, one head  */
#define TMS_KV_ELEMS   (TM_H * TMS_HEAD_ELEMS)
#define TMS_KV_BYTES   (TMS_KV_ELEMS * 2)              /* 32768 B */
#define TMS_HDR_FLOATS (2 * TM_H)                      /* [h*2]=sk [h*2+1]=sv */
#define TMS_HDR_BYTES  (TMS_HDR_FLOATS * 4)            /* 32 B                */
#define TMS_CHUNK_BYTES (8 + TMS_HEAD_ELEMS * 2)       /* 8200 B per head     */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int                 node;      /* 0 or 1 */
    const float*        W;         /* flat fp32 weights (biases + norms used) */
    const TMQ12Weights* q12;

    float   x [TMS_SLOC * TM_D];   /* residual stream (local rows)   32 KB */
    float   b1[TMS_SLOC * TM_D];   /* norm / layer output            32 KB */
    float   b2[TMS_SLOC * TMS_FD]; /* attention ctx / FFN hidden     32 KB */

    int16_t qh     [TMS_SLOC * TM_HD];         /* one head's queries  4 KB */
    int16_t kv_own [TMS_KV_ELEMS];             /* local K,V          32 KB */
    int16_t kv_peer[TMS_KV_ELEMS];             /* peer K,V           32 KB */

    float   sa_inv;                            /* Q15 scale of the norm1 out */
    float   hdr_own [TMS_HDR_FLOATS];          /* local  K,V dequant scales  */
    float   hdr_peer[TMS_HDR_FLOATS];          /* peer   K,V dequant scales  */

    /* Streaming hooks (NULL in the host driver, set by the firmware).
     * kv_sent(h) fires the moment head h's chunk is complete in kv_own;
     * kv_needed(h) must not return until head h's chunk has landed in
     * kv_peer. Together they let the transfer hide behind arithmetic. */
    void  (*kv_sent)(void* ctx, int head);
    void  (*kv_needed)(void* ctx, int head);
    void*   hook_ctx;
} TMShard;

void tm_shard_init(TMShard* st, int node,
                   const float* W, const TMQ12Weights* q12);

/* Local input rows [TMS_SLOC, TM_D] and, after tm_shard_final, local output. */
float* tm_shard_input(TMShard* st);
float* tm_shard_output(TMShard* st);

/* Wire buffers for the per-layer exchange. */
/* Segment pair for head h: the two scales, then its K and V blocks. */
static inline void tm_shard_chunk(TMShard* st, int h, int peer,
                                  void** scales, void** kv) {
    *scales = (peer ? st->hdr_peer : st->hdr_own) + 2 * h;
    *kv     = (peer ? st->kv_peer  : st->kv_own)  + (size_t)h * TMS_HEAD_ELEMS;
}

void tm_shard_layer_pre (TMShard* st, int l);
void tm_shard_layer_post(TMShard* st, int l);
void tm_shard_final(TMShard* st);

#ifdef __cplusplus
}
#endif

#endif /* TM_MODEL_SHARD_H */
