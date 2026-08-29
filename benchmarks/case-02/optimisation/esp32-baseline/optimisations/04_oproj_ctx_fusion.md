# opt 10 — oproj Q15-ctx fusion (two-phase V projection + single core4)

Date: 2026-08-29 · Device: ESP32-C3 @160 MHz (RV32IMC, no FPU)

## Result
| build | s/forward | speedup | gate |
|---|---|---|---|
| before (opt 9b, fp32 A-quant oproj) | 4.160 | 10.13× | host 50/50, worst 1.19e-3; device 5/5 |
| **after (opt 10)** | **4.021 (RNE) / 3.969 (trunc)** | **10.62×** | host 50/50 FAST+EXACT, worst 1.12e-3; device 5/5, worst 1.14e-3 |

## Problem
The oproj (attention output projection, 128×128×128) used `tm_gemm_q12`:
it re-read the fp32 ctx that attention wrote, ran an fp32 amax scan (~6 ms),
fp32 quantize to Q15 (~8 ms) and then the core4 GEMM (~98 ms). The extra
passes cost ~52 ms/call × 4 layers ≈ **208 ms/forward** of soft-float and
memory traffic, on top of attention having already spent equal time writing
that ctx.

## Key ideas
1. **Attention writes Q15 ctx directly** using a *per-layer global* scale
   derived from an exact bound: ctx is a convex combination (weighted average)
   of the V rows, so |ctx_i| ≤ sv_h·max|v_q15|_h for head h. With
   `sa = QACT/(max_h sv_h·vmax_h)·0.9999` every head's |ctx| fits Q15 with no
   clamp;
2. **Two-phase attention** so the global scale is known before any head
   attends: (A) project all 4 heads' V into an int16 view of `g_buf1`
   (`v_all`, head-major), folding per-head sv·vmax into one running max;
   (B) per head, project Q/K (existing buffers) + attend, writing interleaved
   Q15 ctx `[i][h·HD+d]` into `g_ctxq` (an int16 alias of `g_buf2`);
3. **One K=128 `tm_gemm_core4`** for oproj reads g_ctxq directly — the A-quant
   and amax scan disappear; the head-GEMM scratch lives in the top half of
   `g_buf1` (`g_acc`, 32 KB offset so it never collides with `v_all`).

No new SRAM: `v_all` aliases `g_buf1` (dead after the last attn, oproj then
overwrites g_buf1 as its output), `g_ctxq` aliases `g_buf2` (free from qkv
until oproj). EXACT path untouched (still per-head fp32 ctx + fp32 oproj).

## Implementation (src/model.c, FAST only)
```
phase A: for h in 0..3:
    g_vs_h[h] = tm_gemm_head_q15(a16, 1/g_qkv_sa, Wv_h, wx, Vb_h, g_acc, v_all+h·S·HD, D)
    vmax_h = int16 abs-scan of v_all[h]          # exact, cheap (4 KB)
    ctx_max = max(ctx_max, g_vs_h[h]·vmax_h)
g_ctx_sa = QACT/ctx_max·0.9999
phase B: for h in h_order:
    g_qs = proj Q(h) -> g_qh;  g_ks = proj K(h) -> g_kh
    attn_head(..., vh=v_all+h·S·HD, sv=g_vs_h[h], h)   # writes g_ctxq Q15
oproj: tm_gemm_core4(g_ctxq, 1/g_ctx_sa, W_o, wso, OB, g_buf1, 128,128,128,128)
```

## Measurements
oproj 153.8 → 101.4 ms/call (pure core4); attn +4.0 ms/call (Q15-ctx epilogue).
Net −139 ms/forward at opt10, then −52 ms more by replacing the RNE rounding
in the ctx epilogue with a truncating cast (`(int16_t)(int32_t)(a·rot)`),
which also *improved* host worst error 1.27e-3 → 1.12e-3 (rounding mode here
is not the dominate error term).

## Gate
host_test 50/50 FAST + 50/50 EXACT PASS; device 5/5, worst max_abs 1.14e-3
(gate 2e-3). Flash: q12 blob unchanged; SRAM: no net change (aliases).
