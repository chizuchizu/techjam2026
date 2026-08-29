# 24 — SRAM fit + first real on-board validation

## Why
Case-09 FAST never linked on a single XIAO ESP32-C3: `dram0_0_seg`
origin 0x3FC80000 length 0x4E770 (321,392 B) overflowed by 73,072 B
(needed 394,424 B). No on-device measurement was possible at baseline.
This change keeps every FAST-path optimisation (opt 1–23 kernels) but
fits 400 KB SRAM, then reports the first real on-board timing.

## What (SRAM, no FAST-path cost)
1. **5-slot floor (262,144 B)**: `g_buf2`/`g_qh` become non-static COMMON;
   `g_vh` deleted — FAST V now aliases `g_buf1` tail (`v_all = (int16_t*)g_buf1`).
   Kernels.c file-scope overlays (`a16 = g_qh`, scratch blocks into
   `g_buf2[0..4K)`) replace the seven local static scratch decls.
2. **EXACT path split**: fp32 V lives in `g_buf2`, ctx in `g_buf1`,
   oproj = `tm_gemm_f32(g_buf1, ...)`, residual `tm_add_inplace(g_buf2, g_x.f)`.
3. **48-bit attention-score pack**: `int64_t[128]` → `int32_t hi` + `int16_t lo`
   (store `(uint32_t)lo >> 16`; reconstruct with explicit masks) — −256 B.
   Carry accumulation reconstructs the full 64-bit dot (200k random trials,
   0 mismatches); exp-LUT error < 2^16·gsc·6553.5 ≈ 0.03 bins.
4. **int16 pre-exponential** `p15` with `32768 → 32767` clamp (EXACT diagonal
   would wrap int16 otherwise; FAST normalise max already = 32767) — −256 B.
5. **line buffer** moved to `g_buf2` tail (192 B, replaces 160 B static) — −160 B.
6. **Binary-patched prebuilt SDK objects** (no IDF rebuild):
   `xIsrStack` 0x830→0x400 (−1,072 B, RELA addend fixed), coredump
   stack/prstatus/fake-frame → 0 (−1,624 B). Reproduced in-repo at
   `patched_sdk_libs/` (`-L${PROJECT_DIR}/patched_sdk_libs` precedes SDK `-L`).
7. **GELU fixed table in flash** (`gelu_tab_2049.h`) — no RAM LUT.

## Result
- `dram0_0_seg` = `.bss` 267,744 + `.data` 7,160 + `.dummy` 46,080
  = **320,984 / 321,392 B → +408 B margin**; `pio run` RC 0 (reproducible).
- Device (XIAO ESP32-C3, board A, FAST mode, TM_PROFILE on):
  `seed 0..4: fails=0 max_abs≤1.14e-3, 2.165 s fwd, ALL PASS`
  (tool: `python3 tools/device_test.py /dev/cu.usbmodem101 --seeds 0 1 2 3 4 --reps 3`).
- First real on-board forward for case-09; benchmark gate |Δ|≤0.002/0.02·ref passes.
