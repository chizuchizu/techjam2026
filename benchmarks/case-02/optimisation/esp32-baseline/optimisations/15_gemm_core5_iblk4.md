# opt15: core5 GEMM — j-tile-2 × IBLK=4 (register-pressure fix)

Date: 2026-08 (session).  Baseline HEAD: 6dc0103 (opt12); opt13/14 uncommitted.

## Problem
`tm_gemm_core4_v2` (oproj/FFN1/FFN2, all 128x128x128) ran at **7.48 cyc/MAC**
(C4CYC avg 15,685,798 cyc/call ≈ 98.0 ms).  Its inner loop is an 8-row ×
2-col j-tile with K-pair: 16 int32 accumulators + 20 hoisted loads ≈ **36+
live registers**, far past the RV32IMC ABI's ~28 — so the in-order core
spilled to the stack across every K-pair iteration.

By contrast `tm_gemm_head_q15` (8-row × 1-col, ~17 live registers) hits
**5.2 cyc/MAC** on the same core/shape family.

## Fix
New `tm_gemm_core5` = j-tile-2 × **IBLK=4** × K-pair: 8 int32 accumulators
(4 rows × 2 cols) + 4 weight loads + 8 activation loads ≈ **20 live
registers**, comfortably within the ABI.  K=128 even so the K-pair fast path
covers the whole inner loop; fp32 epilogue `(float)c*g+bj` unchanged;
odd-N tail (never taken for N=128) kept.

Identical product order and int32 accumulation ⇒ **bit-exact vs core4_v2**
(host worst gate 1.2774e-3 before and after).

## Results (device, 13-fwd profile / device_test 5 seeds)
| metric | core4_v2 | core5 | Δ |
|---|---|---|---|
| C5CYC avg cyc/call | 15,685,798 | 13,024,767 | **−17.0%** |
| cyc/MAC | 7.48 | 6.21 | −1.27 |
| oproj avg µs/call | 97,921 | 81,493 | −16.8% |
| f1 avg µs/call | 98,000 | 81,191 | −17.2% |
| f2 avg µs/call | 120,236 | 103,634 | −13.8% |
| **s/forward** | **3.184** | **2.982** | **−202 ms (−6.3%)** |
| gate | PASS (worst 1.0877e-3) | PASS (worst 1.0877e-3) | — |

## Notes / next
- The fp32 epilogue (floatsisf+mulsf3+addsf3 per element) is ~irreducible
  for exactness: the outputs are O(0.01–2) fp32 values with fractional bits,
  so a Q30 fixed-point epilogue truncates to integers (verified: worst diff
  ~1.0–1.2 vs fp32) and is numerically invalid.  Head gemm avoids it via the
  per-column min/max amax trick + separate Q15 pass.
- Candidate next: core6 = 8-row × 1-col (head shape) with fp32 epilogue;
  then attn (qk/pv ~570 ms/fwd) and qkv pass-1 (KB0 821 ms/fwd @ 5.2 cyc/MAC).
