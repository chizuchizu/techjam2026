# 25 — attention PV inner loop: clean advancing-pointer form

- Date: 2026-08-29 (case-10 W10, branch linkfast)
- File: src/model.c — attn_head FAST branch, PV (p15 · v) inner loop.

## What
The PV row kernel iterated `const int16_t* vj = vh + (size_t)j * TM_HD;`
inside the j-loop, forcing gcc to recompute row offsets and spill them to the
stack each outer (db-tile) iteration.  Rewrote the j-loop as a single
advancing pointer:

    const int16_t* vj = vh + db;
    for (int j = 0; j <= i; j++) {
        int32_t p = g_p15[j];
        c0 += p*(int32_t)vj[0]; ... c7 += p*(int32_t)vj[7];
        vj += TM_HD;
    }

## Effect on emitted code (riscv32-esp-elf-gcc -O2 -march=rv32imc)
- Old inner loop: ~60 instr / 8 MAC (offset recompute + 6 stack `lw` of
  constant offsets per iteration).
- New inner loop: **28 instr / 8 MAC ≈ 3.5 cyc/MAC**:
  1 lw (p) + 8 lh (v) + 8 mul + 8 add + 2 addi + 1 bne.
- Expected on-device: PV 266 ms -> ~120-140 ms (est.), bit-exact math.

## Safety
Same integer ops in the same per-c dot accumulation order -> bit-exact vs
previous build.  Host gate 5/5 (max_abs identical values).

## Status
Host: ALL PASS seeds 0..4.  Board: pending.  Not committed yet.
