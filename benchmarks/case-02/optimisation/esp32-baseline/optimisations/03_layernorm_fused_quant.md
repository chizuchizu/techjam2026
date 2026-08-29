# opt 9 — Fused LayerNorm → Q15 (amax-bound) for the QKV path

Date: 2026-08-29 · Device: ESP32-C3 @160 MHz (RV32IMC, soft-float)

## Result
| build | s/forward | speedup | gate |
|---|---|---|---|
| before (j-outer head gemm + integer amax) | 4.862 | 8.67× | host 50/50, worst 9.4e-4 |
| **after (fused LN→Q15)** | **4.784 / 4.782 / 4.779** | **8.82×** | host 50/50 FAST+EXACT, worst §1.14e-3 |

## Problem
norm1 (fp32 LayerNorm) + the QKV a16 quantize were three full soft-float passes
per layer: (a) LN stats, (b) LN normalize writing fp32 to `g_buf1`,
(c) Q15 amax scan + quantize into `a16`. On a core with no FPU every fp32 op is
~40–100 cycles, so norm1 ≈ 471 ms/forward with the separate a16 quant.

## Key idea — tight analytic amax bound makes `sa` computable before normalizing
LayerNorm is `out = (x−μ)·rstd·γ + β`, quantized as `q = round(sa·out)`.
A safe per-buffer bound on max|out|:

    out ≤ (x−μ)·rstd·γ + β  ⇒  bound_k = rstd·(max|x|_row + |μ|)·|γ_k| + |β_k|

Scanning rows only for `max|x|` (already needed for stats) and taking
`amax = max_k( Bmax·|γ_k| + |β_k| )` with `Bmax = max_row rstd·(max|x|+|μ|)`
is **O(S·D) for stats + O(D) for the bound** — no separate output scan.

Measured tightness vs the true max over 9 LN sites × 8 seeds (fp32 torch):
median 1.01×, p90 1.05×, worst 1.12× → ≤0.16 bits of Q15 range lost,
negligible accuracy impact (host FAST worst went 9.4e-4 → 1.14e-3, gate 2e-3).

## Implementation — `tm_bn_q15()` (src/kernels.c)
1. Pass 1: per-row sum, sumsq, max|x| → mean, rstd (stores only `mean_r`,
   `rstd_r`, 512 B stack).
2. O(D) bound → `sa = 32767/amax · 0.9999` (safety factor makes the Q15 clamp
   unreachable → per-element clamp removed).
3. Precompute per-layer `AG_k = sa·γ_k`, `AB_k = sa·β_k` (D fp32 ops).
4. Pass 2 (fused normalize+quant): `q = round( (x−μ)·rstd·AG_k + AB_k )`
   → 3 fp32 ops + 1 fcvt per element, straight into `a16`.

Replaces: fp32 LN output write, the separate `tm_gemm_amax` scan, and the
separate `tm_gemm_quantA_into` pass (3 extra full-buffer soft-float passes
removed — this is where the win comes from, not the LN ops themselves).

`g_qkv_sa` is now the BN scale, so the head GEMMs/dequant need no code change
(self-consistent quant/dequant), and the EXACT path keeps plain `tm_layernorm`.

## Cost/lesson
- Host: 50/50 all pass, both paths. Device: 5/5 gate ok (not re-run this build;
  host worst 1.14e-3 keeps ≥1.75× margin).
- Flash: −1 RODATA block (a16 stays in `a16` static SRAM as before, 32 KB).
- Important negative result: a *first* fused variant that recomputed per-row
  A/B coefficients in two per-row arrays was *slower* (~119 ms/call vs ~113 ms
  split) — on this single-issue in-order core, smaller simpler loops with less
  stack state win over fewer-pass complex loops. The A/B-per-call, array-free
  normalize is the version that shipped.

## Next targets (soft-float counts, see profile)
| pass | ms/fwd | note |
|---|---|---|
| QKV head GEMMs | 1010 | int, ~6 cyc/MAC — near floor |
| attention QK+PV | 864 | int, near floor |
| oproj/f1/f2 int16 GEMMs | 617/615/679 | core4 ~102 ms/call + A-quant |
| norm2 + final LN | 357 + 89 | fp32 soft-float (norm2 would need fusion into f1's Q12 quant) |
| gelu | 271 | LUT already |
