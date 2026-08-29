# 27 — adopt ROCK 20 fixed-table GELU (cross-case, root-validated)

- Date: 2026-08-29 (case-10 W10, branch linkfast)
- Upstream: KERNELS.md "ROCK 20" (root, validated on case-12: host 5/5, link OK)
- Files: src/kernels.c (tm_gelu_q15_lut body + include), src/gelu_tab_2049.h (new).

## What
Replace the per-layer SOFT-FLOAT erf LUT rebuild (513 erf-poly evals x 4 calls/fwd
~ 50-60 ms/fwd) with the shared FIXED 2049-entry const int16 erf table (flash
.rodata, 0 RAM) + per-layer integer index scale (~7 int ops/element).
Same function signature -> no caller change in model.c.
Also REMOVED the now-unused static int16 g_gelu_lut[513] (1 KB SRAM freed; case-10
dram0_0_seg was 97.25% full).

## Verified
- Host: 5/5 FAST + 5/5 EXACT PASS.  FAST max_abs shifted slightly (8.35e-4..1.01e-3,
  gate 2e-3) as expected (~2 LSB gelu diff).  EXACT max_abs unchanged (~5e-5).
- Firmware: RAM 81.0 % (265,324 B), Flash 84.2 % (2,648,278 B).  Builds.

## Expected on-board
gelu 63 ms -> ~10-15 ms/fwd (est., board pending).
