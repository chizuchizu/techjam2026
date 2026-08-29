# opt 12 — core4_v2 GEMM: j-tile-2 (8-row × 2-col) + K-pair prefetch

Date: 2026-08-29 · Device: ESP32-C3 @160 MHz (RV32IMC, no FPU)

## Result
| build | oproj/f1 avg | f2 avg | s/forward | gate |
|---|---|---|---|---|
| core4 (opt 11) | ~101.4 ms/call | ~169.4 ms/call | 3.706 | device 5/5, worst 1.11e-3 |
| **core4_v2 (opt 12)** | **~97.9 ms/call** | **~165.9 ms/call** | **3.664** | device 5/5, worst ~1.1e-3 |

## Problem
The core4 kernel processed one output column at a time (8 rows × 1 col).
Each A row element was loaded once per column, and the B column stream came
one value ahead of its use (flash-XIP load→use latency stalls on the 4-stage
in-order core hidden imperfectly by only 8 muls between loads).

## Key ideas (per int-GEMM research: CMSIS-style paired operands + j-tile-2)
1. **j-tile-2**: process 2 output columns per i-tile. 16 int32 accumulators
   (8 rows × 2 cols). Each A q15 load is reused across both weight columns,
   halving A-side `lh` per MAC.
2. **K-pair prefetch**: loop `k += 2`, hoisting the 20 loads (8 A, 4 A[k+1],
   2 B, 2 B[k+1], 4 scratch) ahead of all 32 `mul`s, so flash-XIP latency is
   hidden behind independent multiply chains.
3. Odd-N tail column and odd-K remainder handled (not hit in this model:
   all M=K=N=128).

## Gate & correctness
- kernel numerically bit-identical to core4 (verified 128×128×128 random +
  odd shapes vs reference: max |Δ| = 0).
- Host 50/50 PASS FAST worst 1.198e-3 (unchanged).
- Device 5/5 PASS.

## Notes
The compiler spills ~6 accumulators to the stack in the inner loop (18 live
pointers > 16 GPRs), so the true instruction cost (~205 instr / 32 MACs) is
higher than core4's; the win comes purely from latency hiding. Result:
+3.4% on oproj/f1, +2.1% on f2, ≈ −42 ms/forward. Keep if cheap; the bigger
structural lever remains fusing the QKV head GEMMs (K=32→K=128) and SRAM
transpose-B.

## Cost
No new SRAM, no new flash. One new kernel + prototype + 3 call-site swaps.
