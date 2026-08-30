# Case 12 — optimised ESP32-C3 full-case on-board result (v1, measured)

## What was measured

The maintained `optimisation/esp32-baseline` firmware for case 12 was built and
flashed onto one Seeed XIAO ESP32-C3 (160 MHz, RV32IMC, no FPU). The complete
official test case was streamed on-board: **one full batch of B=64 input
frames**, one complete four-layer transformer forward per frame, for 64 frames
total, under `run_full_case.py`.

- Hardware: Seeed XIAO ESP32-C3, board A (`/dev/cu.usbmodem101`)
- Case geometry: B=64, S=32, D=128, H=4, F=128, L=4, causal
- Numeric mode used for the gate: FAST (integer GEMM path)
- Input batch: B=64 frames generated with the official reference generator
  (`torch.manual_seed(1234)` weights; performance input seed 1234 + 100000),
  one S×D frame per forward, streamed in sequential order 0..63
- Cutoff: 600 s per benchmark — **compliant** (full-case wall time ≪ 600 s)

## Result (full test case = complete B=64 batch)

- Frames: **64/64 PASS** (0 failed elements total, all frames `fails=0`)
- Gate: `abs_err <= 0.002 OR rel_err <= 0.02` per element — **all 64 frames PASS**
- Worst per-element absolute error over all 64 frames: **1.3062e-03**
- **Total on-device time for the full case: 33.8794 s**
  (sum of the firmware `us=` forward counters over all 64 frames)
- Median per-forward time: **0.529366 s** (context; the headline
  number above is the complete-case total)
- **Wall time for the full case (flash/upload/verify excluded):
  73.744 s** (first frame send -> last frame output, includes host
  USB serial upload/download pacing for all 64 frames)

## What the timing includes

The full-case total is the sum of the firmware `us=` counter printed after each
`R` forward. The counter brackets `run_forward()` on-board (weights loaded,
activations computed, output produced) and excludes firmware flash/upload and
host serial round-trips. The wall time additionally includes paced host ->
device input delivery (1 KB chunks, 20 ms gaps to avoid the CDC RX drop bug)
and device -> host output download for all 64 frames.

This is a **measured** full-case result, not a per-forward projection: every
one of the 64 frames was executed on the board and its on-board time summed.

Raw capture: `case-12_xiao_c3_optimised_full_case_v1.log` (64/64 frames with per-frame fails/max_abs/us).
