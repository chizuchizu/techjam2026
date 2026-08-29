# 26 — attention QK inner loop: single base pointer + compile-time row offsets

- Date: 2026-08-29 (case-10 W10, branch linkfast)
- File: src/model.c — attn_head FAST branch, QK 4-way j-unrolled row kernel.

## What
The 4-way j-unrolled QK used 4 separate row pointers k0..k3 advanced by 1 per
d-step (5 `addi` per iter).  Rewrote to a single base pointer `k = kh + j*HD`
and 4 compile-time row offsets (0/HD/2HD/3HD halfwords):

    lh x,0(k)  lh x,2(k)          // row j+0
    lh x,128(k)  lh x,130(k)      // row j+1   (TM_HD=64 => 128 bytes)
    lh x,256(k) ...  lh x,386(k)

## Effect on emitted code
Standalone: ~46 instr / 8 MAC (old) vs ~48-49 instr / 8 MAC (new) — compiler
rebalanced scheduling, roughly neutral in the standalone.  Real on-board delta
unknown until measured; retained because it reduces address-register pressure
and is bit-exact.

## Safety
Same integer limb (L/H) math, same per-dot accumulation order -> bit-exact.
Host gate 5/5 with identical max_abs values per seed.

## Status
Host: ALL PASS seeds 0..4.  Board: pending.  Not committed yet.
