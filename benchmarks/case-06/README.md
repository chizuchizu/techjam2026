# Case 06

Configuration: `B=10000, S=128, D=128, H=4, F=128, L=4`, causal.

The per-input geometry and weights are identical to case 02; only the batch
differs (10,000 input frames, streamed one forward per frame on the board).

## Status

Implemented as a port of the case-02 ESP32 baseline under
[`optimisation/esp32-baseline/`](optimisation/esp32-baseline/).

- **Host validation: PASS** — `make && ./host_test all --both --reps 5` gives
  50/50 seed-runs pass (25 EXACT + 25 FAST); FAST worst `max_abs = 1.0320e-3`,
  EXACT worst `max_abs = 7.8201e-5`. See `HOST_TEST.log`.
- **Device build: links** — `pio run -e esp32-baseline` succeeds (arduino-esp32
  3.0.7); RAM 83.2%, flash 84.9%. See `BUILD.log`.
- **Physical timing: not measured.** No firmware upload or on-device run was
  performed for case 06, so no case-06 forward time or throughput is claimed.

A full `B=10000` physical batch is a long-running execution (~hours) of 10,000
board forwards plus 2 x 10,000 x 64 KiB of serial transfer; it is **not run and
not claimed as measured**. `optimisation/esp32-baseline/tools/batch_stream.py`
is the host-side stream driver for that batch: it reads per-frame bins (or
synthesizes frames), streams them over the firmware serial protocol, can gate
each frame against the torch reference, and prints the transport/estimation
structure without fabricating physical timing.

## Focus

Bounded-memory batch streaming, persistent weights in flash, and fleet data
parallelism. Report sustained throughput, total completion time, transfer
volume, failures, and retries; a full batch cannot be resident in C3 SRAM.
