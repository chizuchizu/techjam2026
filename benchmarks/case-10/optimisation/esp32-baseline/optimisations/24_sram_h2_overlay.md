# opt 24 (case-10 / W10): H=2 SRAM shrink — per-head Q/K overlay (MISSION 1)

Date: 2026-08-30. Workspace: benchmarks/case-10/optimisation/esp32-baseline.
Signed: W10 (case-10 kernel engineer)

## Problem
case-10 (B=64 S=128 D=128 H=2 HD=64 F=128 L=4) did not link on the C3:
`dram0_0_seg` overflowed by 23,920 B (used 345,216 / 321,296 B). The case-2 tree
kept per-head Q/K/V as dedicated int16 statics of size S*HD each; at HD=64 the
three are 3*16 KB = 48 KB vs 24 KB at the case-2 HD=32 geometry -> +24 KB, exactly
the overflow.

## Change (model.c only)
Removed the `g_qh` / `g_kh` static arrays (2 * S*HD * 2 = 32 KB saved) and route
them through zero-cost overlays:

- FAST (measured mode): qh/kh live in the UPPER half of `g_buf2`
  (byte offset 2*S*D .. 2*S*D+2*S*HD+2*S*HD). ctx/g_ctxq is the int16 [S,D]
  lower half; the upper half is dead from end-of-qkv till oproj.
- EXACT (reference mode, not device-run for this case): qh/kh live in kernels.c's
  `a16` scratch (int16 [S,D] = 32 KB), which the EXACT path never touches.
  `g_vh` (16 KB) stays as a dedicated EXACT V buffer.

## Result
- `pio run -e esp32-baseline` now LINKS.
- dram0_0_seg: 312,448 / 321,296 B = 97.25% (8,848 B free).
- Flash: 2,644,420 / 3,145,728 B = 84.1%.
- Host gate (host_test, 25 seeds x 2 modes): 50/50 PASS
  (FAST worst max_abs ~1.04e-3, EXACT worst ~7.5e-5).

## Next
- MISSION 2: profile per-phase on board (boardB), then adopt the D=128 flagship
  kernels from KERNELS.md and tune attention for the H=2 head width.
