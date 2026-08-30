/*
 * transformer.h - runtime-parameterized Transformer forward for the Tech Jam
 * benchmark family.  ONE C function runs any case geometry:
 *
 *     tm_case cfg = {B, S, D, H, F, L, causal=1, mode=TM_FAST};
 *     size_t n  = tm_workspace_size(&cfg);
 *     void *ws  = malloc(n);
 *     int16_t **q = malloc(tm_qmat_count(&cfg) * sizeof(...)); // see below
 *     float *wscl  = malloc(tm_qmat_count(&cfg) * sizeof(float));
 *     tm_scan_q12(&cfg, q12_blob, q, wscl);
 *     tm_run(&cfg, ws, weights_f32, q, wscl, x, y);
 *
 * The module is plain C99 with no dynamic allocation inside tm_run: all
 * scratch lives in the caller-provided workspace, so it can be pre-sized on
 * an MCU (malloc once, run many times) and never fragments the heap.
 *
 * Two numeric modes:
 *   TM_EXACT - fp32 GEMM + fp32 softmax + exact-erf GELU (reference-quality)
 *   TM_FAST  - Q15 activations x Q12 weights integer GEMM, int64 QK/PV
 *              attention, fp32 softmax with a fast exp, fp32 residual.
 * Both are compiled together; choose per call with cfg.mode.
 */
#ifndef TM_PARAM_H
#define TM_PARAM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int B;       /* batch: informational; one forward processes one S*D frame */
    int S;       /* sequence length (tokens per frame)                        */
    int D;       /* model dim; must be divisible by H                          */
    int H;       /* attention heads                                            */
    int F;       /* FFN hidden dim                                             */
    int L;       /* transformer layers                                         */
    int causal;  /* 1 = causal (upper-triangle masked), 0 = bidirectional      */
    int mode;    /* TM_EXACT or TM_FAST                                        */
} tm_case;

enum { TM_EXACT = 0, TM_FAST = 1 };

/* ---- weight blob sizing (float buffer and parsed Q12 matrices) ---- */
size_t tm_w_total(const tm_case* cfg);   /* flat fp32 weight floats            */
size_t tm_w_layer(const tm_case* cfg);   /* floats per layer                   */
int    tm_qmat_count(const tm_case* cfg);/* == L * 6 (q,k,v,o,f1,f2 per layer) */

/* Parse the Q12 blob written by tools/export_case2.py:
 *   [per layer][q,k,v,o,f1,f2] = {u32 count, f32 wscale, int16 data[N*K]}
 * Caller supplies arrays of tm_qmat_count() entries.  q[i] points *inside*
 * `blob`, so keep the blob alive while the forward runs. */
void tm_scan_q12(const tm_case* cfg, const void* blob,
                 const int16_t** q, float* wscale);

/* ---- workspace & run ---- */
size_t tm_workspace_size(const tm_case* cfg);
void   tm_workspace_init(const tm_case* cfg, void* ws);  /* reset profile accs */

/* One forward.  x,y point to S*D floats each; weights=flat fp32 of
 * tm_w_total() floats.  q/wscale may be NULL when cfg.mode==TM_EXACT.
 * Returns 0 on success, negative on invalid parameters. */
int tm_run(const tm_case* cfg, void* ws,
           const float* weights,
           const int16_t* const* q, const float* wscale,
           const float* x, float* y);

/* direct buffer access (device firmware reads input / streams output
 * straight into the workspace to avoid an extra 64 KB copy) */
void*  tm_input_buf(const tm_case* cfg, void* ws);   /* residual buffer (S*D fp32) */
void*  tm_output_buf(const tm_case* cfg, void* ws);  /* after tm_run with yout==input */

/* ---- profiling ---- */
typedef int64_t (*tm_now_us_fn)(void);
void tm_set_clock(tm_now_us_fn fn);   /* NULL restores the host default     */

/* Accumulated per-kernel microseconds for the last run (valid until the next
 * run; reset by tm_workspace_init). */
typedef struct {
    int64_t norm1_us, qkv_us, attn_us, oproj_us, res1_us;
    int64_t norm2_us, ffn1_us, gelu_us, ffn2_us, res2_us, final_us;
    int64_t quant_us;
    int64_t total_us;
} tm_profile;

void tm_profile_get(const tm_case* cfg, const void* ws, tm_profile* out);
void tm_profile_dump(const tm_case* cfg, const void* ws);  /* prints to stdout */

#ifdef __cplusplus
}
#endif
#endif /* TM_PARAM_H */
