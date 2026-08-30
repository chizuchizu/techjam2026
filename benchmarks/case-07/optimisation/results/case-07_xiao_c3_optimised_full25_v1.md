# Case 07 — optimised ESP32-C3 full-25-seed on-board result (v1, measured)

## What was measured

The maintained `optimisation/esp32-baseline` firmware for case 07 was built and
flashed onto one Seeed XIAO ESP32-C3 (160 MHz, RV32IMC, no FPU) and executed for
**all 25 official device seeds** (`input_0` .. `input_24`) with **3 repetitions
per seed** under `tools/device_test.py`.

- Hardware: Seeed XIAO ESP32-C3, board A (`/dev/cu.usbmodem101`)
- Case geometry: B=64, S=128, D=32, H=4, F=32, L=4, causal
- Numeric mode used for the gate: FAST (integer GEMM path)
- Cutoff: 600 s per benchmark — **compliant** (wall of one case ≪ 600 s)

## Result

- Device seeds: **25/25 PASS** (0 failed elements total, all seeds `fails=0`)
- Worst per-element absolute error over all 25 seeds: **1.4625e-03**
- Gate: `abs_err <= 0.002 OR rel_err <= 0.02` per element — **all seeds PASS**
- Per-forward time (firmware `TM` counter, 3 timed forwards):
  475099 / 474979 / 475172 us -> median **0.4751 s/forward**
- Per-seed forward time reported by the device: median **0.475 s**
- Wall time for the full 25-seed run (flash/upload/verify excluded): **43 s (00:33:21 -> 00:34:04 +08, 2026-08-30)**

## What the timing includes

The forward time is the firmware `TM` counter over one complete four-layer
transformer body on-board (weights loaded, activations computed, result
serialised). It excludes firmware flash/upload, host serial round-trip, and
`device_test.py` per-seed data upload/download. Per-seed device time reported
here (`fwd_s`) is the on-board forward time printed by the firmware after the
FWD command completes, not a host-side projection.

This is a **measured** result, not a projection. A complete-batch total would be
a derived projection (B=64 streamed inputs x per-forward time) and is
intentionally not reported as a measurement.

Raw capture: `case-07_xiao_c3_optimised_full25_v1.log` (25/25 seeds
with per-seed fails/max_abs/forward time, plus the 3-forward `TM` sweep).
