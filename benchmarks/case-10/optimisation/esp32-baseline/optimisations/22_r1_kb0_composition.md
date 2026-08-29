# 22 — R1 + KB0 composition (integer residual + head-GEMM asm)

Merges the two independently-shipped optimizations onto one firmware:
**R1 (opt 21)** integer-residual FAST path + **KB0 (opt20-kb0)** hand-asm inner
loop for `tm_gemm_head_q15`. Device (same Seeed XIAO ESP32-C3):
**2.122 → 2.056 s/forward** (kb0's −66 ms adds exactly onto R1; both are
orthogonal — R1 touches residual/norms/epilogues, kb0 touches the head-GEMM
MAC loop only).

## Gate & device
- Host `make -C tools test`: **ALL PASS** (54 seed-runs, 0 failed). FAST worst
  1.03e-3, EXACT ≤ 6.7e-5.
- Device, all 25 manifest seeds (`tools/device_test.py`): **PASS 25/25**,
  worst max_abs **1.24e-3** (seed 18 — identical to R1; kb0 is bit-exact),
  forward **2.057 s**.
- Scoring: `tools/runs.json` (t_s=2.057) → `score.py` → **scores.json:
  weighted ExScore 0.267 (fp32) → 5.30 (R1) → 5.48 (R1+kb0)**; MFU(mix) 547.8%.

## Per-forward region profile (merged build, 6 fwd)
| region | time | % |
|---|---|---|
| KB0 head GEMM (QKV proj) | 578 ms | 28% |
| attention QK + exp + PV | 207 + 44 + 266 = 517 ms | 25% |
| f2 (core5_resid + gelu staging) | 293 ms | 14% |
| oproj (core5_resid) | 230 ms | 11% |
| f1 (core5_q15) | 216 ms | 11% |
| norm1 + norm2 | 112 ms | 5% |
| gelu | 63 ms | 3% |
| final norm | 47 ms | 2% |

Next levers (in order): (1) hand-asm the `tm_gemm_core5_resid` /
`tm_gemm_core5_q15` inner MAC loops (currently pure-C 5.8 cyc/MAC; kb0-style
2-pair + XIP prefetch targets ~3.3 → would cut f1+oproj+f2 ≈ 740 ms by ~1/3);
(2) attention PV/QK further unroll/prefetch (517 ms).

## Files
- Merged sandbox: `/tmp/opt21_merge` (R1 + kb0.patch applied; measured
  build). Patches: `patches/21_int_residual_fast_path.patch` (R1),
  `/tmp/opt20/kb0/kb0.patch` (kb0, applied on top).
- This build = `esp32-baseline/src` at the R1+kb0 state; docs and scoring
  artifacts updated to 2.057 s.
