# opt 11 — integer-only attention PV + integer ctx epilogue (no soft-float in attention)

Date: 2026-08-29 · Device: ESP32-C3 @160 MHz (RV32IMC, no FPU)

## Result
| build | s/forward | gate |
|---|---|---|
| before (opt 10b) | 3.969 (host) | host 50/50, worst 1.12e-3 |
| **after (opt 11)** | **3.688 / 3.706 (device)** | host 50/50 worst 1.20e-3; device 5/5 worst 1.11e-3 |

## Problem
Two soft-float spots remained inside the per-token attention loop:
1. **PV**: for each context row, an fp32 per-row divide of the exp weights
   (`p = e/lsum`) followed by an int64 PV accumulate (64-bit math via
   `mul/mulhu/mulh` chains + carry adds). ~280 ms/forward total.
2. **ctx epilogue**: `oq[x] = sa*rot*(float)c0` per element — 3 soft-float
   calls per element, and the fp32 conversion truncates the int32 acc to 24
   significant bits.

## Key ideas
1. **int32 PV via per-row weight rescale.** ctx is a weighted average, so the
   exp weights only matter up to scale. Rescale each row to sum ≈ 32767:
   `f15 = QM^2 / lsum15` (QM = 32767), `p'[j] = (p[j]*f15 + 0x4000)>>15`
   ⇒ Σp' ≈ 32767, so |Σ p'·v| ≤ 32767² ≈ 2^30 fits int32. PV then uses only
   a single 16×16 `mul` per element — no 64-bit accum, no `mulh`.
2. **ctx scale `rot` hoisted per head.** `rot = sv·g_sctx/QACT` computed once
   per head instead of per element; the per-row fp32 divide disappears.
3. **Integer ctx epilogue**: rewrite `rot` as `m/2^sh` (m ∈ [2^28,2^30), sh
   found once per head in fp32), then
   `oq[x] = (int16_t)(((int64_t)c0*m + (1LL<<(sh-1)))>>sh)` — an exact int64
   multiply, strictly more accurate than `(float)c0*rot` (which rounded c0 to
   24 bits first). `g_p15` (exp weight buffer) is overwritten in place with
   the scaled weights.

EXACT (fp32) path untouched, still bit-equivalent. `g_p15 = static int32_t[TM_S]`
is the only new storage (2 KB); all other buffers alias existing ones.

## Gate
- Host: 50/50 PASS FAST, worst 1.198e-3, EXACT all PASS.
- Device: 5/5 PASS, worst 1.108e-3.

## Cost
Zero new SRAM beyond `g_p15` (2 KB). Flash +0 (no new tables).
