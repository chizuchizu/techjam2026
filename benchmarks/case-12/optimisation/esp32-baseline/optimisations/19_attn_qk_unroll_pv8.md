# 19 — attention QK unroll + PV 8-accumulator (opt 19)

Goal: cut the causal attention QK and PV hot loops (`attn_head` in `src/model.c`).
Baseline at opt18: attn_qk ≈ 266 ms/forward (≈ 129.6 µs/call), attn_pv ≈ 270 ms/forward
(≈ 131.8 µs/call), forward total ≈ 2.4438 s.

## What changed
1. **QK j-unroll-4 with shared q loads.** The causal QK dot loop now computes four
   rows `j..j+3` per iteration from one `qi16[d]` element pair, each row with its own
   int32 hi/lo limb accumulator (identical integer math, per-dot add order unchanged
   → bit-exact vs opt18; carry/borrow logic `H += (up<L) + (p<0?-1:0)` preserved).
   The 4 kh pointers stay live across the `d` loop.
2. **PV 8 register accumulators** (`db += 8` instead of `db += 4`): halves the
   `g_p15[j]` loads per unit of V data with 8 independent int32 partial sums.
   Bit-exact (same per-`p` accumulation order).

## RV32IMC instruction budget (full-model objdump)
- QK inner: was 8 instr/MAC effective → now 52 instr / 16 MAC = **3.25 instr/MAC**
  (4 dots share q loads, 8 independent limb-adds per step).
- PV inner: 8 accumulators halve the p-load traffic vs 4 accs.

## Verification
- Host gate `make -C tools test`: **50/50 PASS** (FAST ≤ 9.99e-4, EXACT ≤ 6.8e-5).
- Device gate (5 seeds, real flash weights): **5/5 PASS**, max_abs 1.01e-3 .. 1.29e-3
  (identical quality to opt18).

## Device measurement (XIAO ESP32-C3 @ 160 MHz, TM_PROFILE on, 4 fwd)
| region | opt18 avg us/call | opt19 avg us/call | Δ |
|---|---|---|---|
| attn_qk | 129.6 | **100.7** | **−22.3%** |
| attn_exp | 20.6 | 21.4 | noise |
| attn_pv | 131.8 | 130.1 | −1.3% (noise) |
| KB0 / KB1 / C5CYC | 13433 / 925 / 11.586M cyc | 13433 / 925 / 11.586M cyc | unchanged |
| **forward wall** | **2.4438 s** | **2.3860 s** | **−57.8 ms (−2.4%)** |

QK total for the forward dropped ~266 → ~208 ms. PV stayed flat: its inner loop is
mul/accumulate-latency bound (mul dep 7.04 cyc), not load bound, so the 4→8
accumulator split adds no device time. Both changes were retained because they
reduce instruction count with zero downside.

## Negative results worth recording
- **Flash-XIP weight streaming is NOT the core5 bottleneck** (opt-cc5 worker, device
  microbench): current j2×i4 K-pair core5 inner runs 3.65 cyc/MAC with weights in
  flash vs 3.32 in SRAM (+0.33 only; the 512 B j-tile stays in the 16 KB flash cache).
  j4×i4 (16 accs) regresses to 3.80/4.43 (RV32IMC has 32 regs → stack spills),
  packed 32-bit weight loads regress to 3.78. The 5.8 cyc/MAC real-call gap is the
  soft-float epilogue + in-context cache sharing. **core5 shipped unchanged.**
- **IRAM placement of hot kernels is not viable**: ESP32-C3's `.iram0.text` and
  `.dram0.data/bss` are aliases of one physical 313 KB SRAM pool; this build already
  sits at 99.9% of it (384 B free), so moving even tm_gemm_head_q15 (128 B text) into
  IRAM overflows the DRAM segment. Skipped.

## Cost
Flash 0 B (bit-exact, no new tables). RAM 0 B (no new statics).
