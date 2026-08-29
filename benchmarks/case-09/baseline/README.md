# Case 9 baseline

The shared PyTorch definition is
[`../../../torch_transformer_benchmark.py`](../../../torch_transformer_benchmark.py).
The torch reference is vendored inside the implementation at
[`../optimisation/esp32-baseline/tools/torch_ref.py`](../optimisation/esp32-baseline/tools/torch_ref.py)
(a self-contained copy of the official benchmark reference; same weight
seed 1234, same random-input generator, same fp32 forward). This
directory owns the host-gate evidence and the SRAM-limit record for the
Case 9 (H=1) geometry.

## Physical starting result

Complete forward = all 64 batch inputs on one board (each input is one
`S*D` frame streamed through the firmware).

| Implementation | Board | Complete forward (single-input) | Accuracy |
|---|---:|---|
| Current hybrid C implementation | XIAO ESP32-C3, 160 MHz | **not measurable** — linker `dram0_0_seg` overflowed by 73,072 B; no firmware image | Pass, 25/25 host checks both modes, worst 1.0738e-03 (FAST) / 7.7963e-05 (EXACT) |

## SRAM-limit record (why there is no device number)

The C3 has a single 400 KB SRAM (addressable window 0x3FC80000..), of
which the ESP-IDF v4.4.7 linker grants the application a
`dram0_0_seg` of 321,296 bytes (the upper region is reserved for the
ROM). This baseline keeps `g_x/g_buf1/g_buf2` (64 KB each), the Q15
`a16` workspace (32 KB), weights, and the per-head Q/K/V plus 2x context
buffers. Per-head state scales as `S * HD`, so it shrinks as H grows:

| Case | H | HD | Static+heap needed | dram0_0_seg | Fits? |
|---|---|---:|---:|---:|---|
| case-09 | 1 | 128 | 394,424 B | 321,296 B | **no, over by 73,072 B** |
| case-10 | 2 | 64 | 345,272 B | 321,296 B | **no, over by 23,920 B** |
| case-11 | 16 | 8 | 256,180 B | 321,296 B | yes (78.2%) |

Measured (not estimated) DRAM layouts above: `case-11` is the smallest
head width, fits with 65,116 B free. `H<=2` genuinely cannot run the
canonical workspace on one C3 — this is a hardware SRAM property, not a
host-gate or accuracy failure (host checks are 50/50 PASS).

## Result evidence

- **Host gate (case-09).** From
  `optimisation/esp32-baseline`: `(cd tools && make host_test && ./host_test
  all --both)` — 50/50 seed-runs ALL PASS (25 seeds x FAST + EXACT),
  worst absolute error 1.0738e-03 (FAST) / 7.7963e-05 (EXACT). Full raw
  output and the build-failure capture live in
  [`results/`](results/).
- **Device build (case-09).** `pio run` fails at the link step:
  `region 'dram0_0_seg' overflowed by 73072 bytes`. No firmware image is
  produced, so `device_test.py` cannot be run on PORT B.
- **Weights/vectors.** Regenerated deterministically
  (`tools/export_case2.py`, `--H 1 --D 128 --S 128 --F 128 --L 4`, 25
  seeds); weights.bin 1,594,368 B, weights_q12.bin 786,624 B. Both are
  gitignored, as are testdata/ and the build dir.

## Hardware alternatives (for the case-9/10 geometry)

1. Reduced-memory head path: stream Q/K/V rows per token and fuse the
   context write-back so no 128x128 per-head staging copies exist; on
   C3 this is the only way H=1 fits (est. saves >73 KB in .bss).
2. Multiboard: H=1 (and H=2) are attention-subgraph-trivial for a
   split — one peer runs the full-D projections (the C3 can, as
   case-11 shows), the other carries the 128-wide attention. The
   baseline here is the projector + host-gate-verified attention
   reference the split must match.
