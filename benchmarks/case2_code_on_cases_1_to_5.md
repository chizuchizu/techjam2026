# Case-2 optimised code applied to cases 1–5 (ESP32-C3, measured)

The case-2 complete-body firmware (FAST mode, opt23) was run against the
case 1–5 data on two Seeed XIAO ESP32-C3 boards. Weights and test data for
cases 1–5 are byte-identical to case 2; case 1/3/4/5 differ only in batch
size `B` and stream `B` independent inputs through one board.

> Measured 2026-08-29. Devices: `/dev/cu.usbmodem101` + `/dev/cu.usbmodem1101`.

## Host gate (case-2 code vs each case reference)

Every case: 25/25 FAST + 25/25 EXACT = **50/50 PASS**.
Worst FAST `max_abs = 1.032e-3` (seed 8), worst EXACT `max_abs = 7.82e-5` (seed 16).

## On-device timing (complete body, FAST mode)

| Case | B | Per-input forward | Batch total (device) | Wall incl. serial | Unoptimised baseline (est.) | Baseline vs 5-min cutoff |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 64 | 1.990 s | 127.36 s | 131.0 s | 2,697.6 s (44.96 min) | time limit exceeded |
| 2 | 1 | 1.990 s | 1.990 s | 5.6 s | 42.15 s (0.70 min) | OK |
| 3 | 4 | 1.990 s | 7.96 s | 11.8 s | 168.6 s (2.81 min) | OK |
| 4 | 16 | 1.990 s | 31.84 s | 35.5 s | 674.4 s (11.24 min) | time limit exceeded |
| 5 | 128 | 1.990 s | 254.72 s | 258.4 s | 5,395.2 s (89.92 min) | time limit exceeded |

- **Optimised runs**: all cases finish under the 5-minute per-case cutoff.
- **Optimised speedup**: ~21.2× per forward vs the unoptimized baseline.
- **Unoptimised baseline** estimated at 42.15 s/input (initial fp32/hybrid
  forward from the case-2 log); cases 1, 4, and 5 would exceed the cutoff.

## Method

- Firmware: case-02 `optimisation/esp32-baseline` source (opt23, latest),
  FAST mode; only the timing command's forward cap was raised (9→255 in
  `src/main.cpp`) so a full batch runs as one timed command. Model/kernels
  untouched.
- Timing: firmware `T <B>` command — one warmup (excluded) + `B` timed
  complete-body forwards; device-reported microsecond counts summed per case.
- Per-forward compute is input-independent, so `T` re-running inputs gives
  the same batch total as `B` distinct inputs (verified per-seed: 1.990–1.992 s).
- On-device output validation: gate PASS (max_abs ≈ 1.03e-3 FAST) on the
  checked seeds, matching the host result.
