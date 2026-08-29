# 23 — core5 4×2 hand-asm fix (col1 product-reuse bug) + probe/rebuild

Fixes and re-validates the two `core5` hand-asm inner loops (`tm_gemm_core5_q15`,
`tm_gemm_core5_resid`) added around the 12:00 working session, which had been
regressing `device_test` (exact on col0, wildly wrong on col1).

## Root cause (two bugs)

1. **Product-reuse register bug.** The 4×2 asm body kept the *activation* values in
   `t0..t3` and made products directly back into the same registers:
   `mul t0,t0,a4` (col0 ok) then `mul t0,t0,a5` — so col1 computed
   `(act·w0)·w1` instead of `act·w1`. Fixed by sending products to
   `t4,t5,t6,a0` (acts stay in `t0..t3`, weights `a4`/`a5`), clobber list expanded
   to `"a0","a4".."a7","t0".."t6"`. Both the q15 8-MAC loop and the resid/probe
   16-MAC loop (prefetch `a4..a7`, tail for k=K-2) got the same corrected body.

2. **Dropped `1:` label when the probe was regenerated programmatically.** The probe
   asm was rebuilt from a filtered instruction list that omitted the `1:` loop label;
   its `bne ... 1b` then bound to the nearest *other* `1:` in the same translation
   unit (kbench2's loop, 0x42004430), so the probe jumped into `tm_kbench2` and
   panic'd (`Load access fault`). Restored the `1:` label; all three asm blocks
   (core5_q15, core5_resid, probe) now have their own label before each `1b`.

## Recovery
`kernels.c` had been corrupted by an over-eager span replacement (head_q15 asm
clobbered + core5_q15 deleted). Restored from `kernels.c.base` (11:41), which still
held the proven 8×2 `tm_gemm_head_q15` asm, then re-applied the two corrected core5
asm bodies + the regenerated probe on top.

## Verification (device, Seeed XIAO ESP32-C3)
- New probe `tm_dbg_c5acc` (`C` command): asm vs C reference identical,
  **bad=0 worst=0** (both columns).
- `device_test` all **25 manifest seeds PASS**, worst element |Δ| ≈ **1.08e-3**
  (gate 2e-3 / 2%), forward **≈ 1.996 s** (recorded opt22 2.056 → ~3% faster; the
  previously-broken col1 path is now on-device-correct).

## Files
- `/tmp/opt23/src/kernels.c` — working firmware source (asm + probe, builds clean).
- `kernels.c.base` — clean pre-asm snapshot used as recovery point.
