# 21 — integer-residual FAST path (R1)

Full integer refactor of the layer residual on the FAST path. Device (same
Seeed XIAO ESP32-C3, 160 MHz RV32IMC, no FPU): **2.385 → 2.122 s/forward =
−263 ms (−11.0%)** vs the opt19 baseline, gate **50/50 host, ALL PASS**.

## Idea

The residual (x = layer input + projection output) is the last thing on the
FAST path still in fp32, and it forces three float costs per layer:
1. **oproj/f2 GEMM float epilogues** — int32 accumulate → float → multiply by
   scale, per element (soft-float libcalls),
2. **separate fp32 res1/res2 passes** — add the projection into the residual,
3. **float amax-scan + float→Q15 quantize** in each per-layer norm (the norm
   reads the residual and re-quantizes it every layer).

R1 carries the residual as **int32 at a fixed scale `sx = TM_RES_SPAN/2^31`
(span 16 → LSB ≈ 7.5e-9, numerically exact for residual magnitudes ±5)** in a
union with the fp32 input view (g_x.f / g_x.s: same 64 KB SRAM, no new buffer).
Then:
- **epilogues become fixed-point**: `xq += ROUND((acc+bq) * (g/sx))` using `g/sx`
  as an exact 24-bit-mantissa pair `(r_m, re)` → one 32x32→64 mul + shift per
  element (mulh on RV32) instead of two soft-float libcalls (~10 instr vs 268).
- **res1/res2 disappear** — `if(!fast)` guards; the GEMMs now fold straight into
  the int32 residual.
- **norms run integer stats on the residual** (fused LN → Q15 a16), with a
  per-row **local integer rescale** `p15 = qx >> sh` (sh chosen so |p15|max ∈
  [16384,32767], pure int clz/shift — recreates the adaptive-Q15 precision the
  float norms had, but with zero soft-float) then the existing fixed-point
  normalize; `sxi = 2^sh·sx` folded into the per-row `rstd·2^30` factor (still
  only 2 int mults/element).
- **input quantized once** at the start of FAST via the exact-mantissa bit
  trick (float → int32, single rounding, no soft-float).
- **final norm** gets the same int stats + local rescale, float output.

EXACT path is untouched: `g_x.f`, `tm_layernorm/fp32 GEMMs` bit-identical.

## Why int32 (history)

The first R1 draft carried a Q15 (span 16) residual to reuse the existing
fixed-point norms; its LSB ≈ 4.9e-4 pushed FAST error to 4.1e-3 (gate 2e-3).
Shrinking the span to 6 got it to 1.78e-3 but left no headroom for ±5 spikes
(overflow at span 5). Key insight: the fixed-point norm pass needs a residual
LSB ≈ amax/32767 (human-scale sxi), which contradicts a *fixed* residual scale;
the **local integer rescale decouples the two**, so the residual can store the
full int32 precision while the norm still lands exactly in the adaptive Q15
range. Result: worst FAST error **1.03e-3 — slightly better than the opt19
baseline's 1.24e-3**.

## Gate

`make -C tools test` (host, 25 FAST + 25 EXACT seeds × 2 modes):
**ALL PASS, 0 failed**. FAST worst max_abs = **1.03e-3** (seed 15; ≤2e-3 gate),
EXACT ≤ 6.7e-5.

## Device (per forward, avg of 3 timed runs, same device as the 2.385 s baseline)

| region | baseline | R1 | Δ |
|---|---|---|---|
| forward wall | 2385 ms | **2122 ms** | **−263 ms (−11.0%)** |
| res1 + res2 | 86 ms | ~0 | −86 ms (passes removed) |
| oproj (core5, float epilogue) | 290 ms | 230 ms | −60 ms (fixed epilogue) |
| f2 (incl. GELU staging) | 352 ms | 293 ms | −60 ms |
| final norm | 90 ms | 47 ms | −43 ms (int stats + local rescale) |
| norm1 + norm2 | 136 ms | 112 ms | −24 ms (int stats) |
| f1 / gelu / qkv / attn | 216/62/~405/260 | ~216/62/unchanged/unchanged | 0 (not touched) |

## Cost

RAM: **0 B new** — g_x becomes a union (64 KB either way), norm stat buffers
reused, no new tables. Flash: +~1.2 KB text (4 new kernels, `.text`). EXACT path
bit-identical.

## Files

- `patches/21_int_residual_fast_path.patch` — unified diff vs pristine
  `src/{kernels.c,kernels.h,model.c,tm_config.h}` (applies cleanly with
  `patch -p1`; pristine kernels.c sha `98d40c11…`).
- Sandbox with sources + measured firmware: `/tmp/opt21_r1`.


## Device benchmark — all 25 test cases (manifest.json seeds 0–24)

Ran every seed in `testdata/` through the real device (tools/device_test.py,
`R` forward, gate ATOL=0.002 OR RTOL=0.02 vs `ref_<s>.bin`):

| seeds | result | forward | worst max_abs |
|---|---|---|---|
| 0–9 | PASS (10/10) | 2.124–2.125 s | 1.0817e-3 |
| 10–24 | PASS (15/15) | 2.124–2.125 s | 1.2366e-3 (seed 18) |
| **0–24** | **PASS 25/25** | **2.124 s (mean)** | **1.237e-3** |

Scoring artifacts regenerated with the repo tools:
`tools/runs.json` (per-seed t_s = 2.124–2.125 s) →
`tools/score.py --runs tools/runs.json --output scores.json`:
**weighted ExScore = 5.30, weighted MFU(mix) = 530%, raw-int MFU = 18.0%**
(equality-weighted 25 cases; peaks P_INT=320 MFLOP/s, P_SOFTFP=2 MFLOP/s,
BW=640 MB/s). For reference the fp32 42.1 s baseline gave ExScore 0.267.

Host: `make -C tools test` → **ALL PASS** (FAST worst 1.03e-3, EXACT ≤6.7e-5).

## Composes with

opt20-epi R2 (fixed-point epilogue, superseded for oproj/f2 by this) and
opt20-kb0's head_q15 asm (device −66 ms) are orthogonal; a merged build
(R1 + kb0) is expected ≈ 2.05 s/forward.
