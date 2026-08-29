# Case 10 baseline

The shared PyTorch definition is
[`../../../torch_transformer_benchmark.py`](../../../torch_transformer_benchmark.py).
The torch reference is vendored inside the implementation at
[`../optimisation/esp32-baseline/tools/torch_ref.py`](../optimisation/esp32-baseline/tools/torch_ref.py)
(a self-contained copy of the official benchmark reference; same weight
seed 1234, same random-input generator, same fp32 forward). This
directory owns the SRAM-limit record for the
Case 10 (H=2) geometry.

## Physical starting result

Complete forward = all 64 batch inputs on one board (each input is one
`S*D` frame streamed through the firmware).

| Implementation | Board | Complete forward (single-input) | Accuracy |
|---|---:|---|---|
| Current hybrid C implementation | XIAO ESP32-C3, 160 MHz | **not measurable** — linker `dram0_0_seg` overflowed by 23,920 B; no firmware image | no firmware image |

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
canonical workspace on one C3 — this is a hardware SRAM property, not an
accuracy failure.

## Result evidence

- **Device build (case-10).** `pio run` fails at the link step:
  `region 'dram0_0_seg' overflowed by 23920 bytes`. No firmware image is
  produced, so `device_test.py` cannot be run on PORT B. The
  build-failure capture lives in [`results/`](results/).
- **Weights/vectors.** Regenerated deterministically
  (`tools/export_case2.py`, `--H 2 --D 128 --S 128 --F 128 --L 4`, 25
  seeds); weights.bin 1,594,368 B, weights_q12.bin 786,624 B. Both are
  gitignored, as are testdata/ and the build dir.

## Hardware alternatives (for the case-9/10 geometry)

1. Reduced-memory head path: stream Q/K/V rows per token and fuse the
   context write-back so no 128x128 per-head staging copies exist; on
   C3 this is the only way H=2 fits (est. saves >24 KB in .bss).
2. Multiboard: H=1 (and H=2) are attention-subgraph-trivial for a
   split — one peer runs the full-D projections (the C3 can, as
   case-11 shows), the other carries the 64-wide (2x) attention. The
   baseline here is the projector + attention
   reference the split must match.
