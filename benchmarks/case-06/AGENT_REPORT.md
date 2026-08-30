# Case 06 agent report

Date: 2026-08-29 (session-local).

## What was done

- Created `benchmarks/case-06/optimisation/esp32-baseline/` as a port of the
  case-02 `esp32-baseline` reference (same per-input geometry and weights:
  S=128, D=128, H=4, F=128, L=4, causal).
- `src/tm_config.h`: `TM_B` set to `10000`; comments updated to case 6. The
  firmware still runs one forward per input frame (streamed batch).
- Copied `weights.bin`, `weights_q12.bin`, `manifest.json`, and `testdata/`
  verbatim from case 02 (hashes verified identical; no B=10000 torch batch was
  regenerated).
- `tools/host_test.c`: case-specific comments now say `TM_S*TM_D` floats
  instead of `16384`; gate math untouched (atol=0.002, rtol=0.02 OR rule).
- Added `tools/batch_stream.py`: host-side stream driver with dry-run and live
  serial modes; reads bins or synthesizes frames; validates vs torch refs when
  requested; prints transport/estimation structure and never fabricates
  physical timing. Verified `--help` and `--dry-run`.
- Updated `optimisation/esp32-baseline/README.md` and `case-06/README.md`.

## Host test result

Command:
`cd benchmarks/case-06/optimisation/esp32-baseline/tools && make && ./host_test all --both --reps 5`

- EXACT: 25/25 PASS (worst max_abs = 7.8201e-5)
- FAST:  25/25 PASS (worst max_abs = 1.0320e-3)
- Total: 50 seed-runs, 0 failed => **ALL PASS**

Full output: `benchmarks/case-06/HOST_TEST.log`.

## Device build result

Command:
`cd benchmarks/case-06/optimisation/esp32-baseline && timeout 600 pio run -e esp32-baseline`

- Result: **LINKED SUCCESSFULLY** (PlatformIO SUCCESS, no errors)
- Platform: Espressif 32 (51.3.6), arduino-esp32 3.0.7
- RAM 272,700 / 327,680 bytes (83.2%), Flash 2,671,272 / 3,145,728 bytes (84.9%)

Full output: `benchmarks/case-06/BUILD.log`.
No upload / no on-device run was performed, so no case-06 physical timing is
claimed.

## Not claimed / not measured

- Full `B=10000` physical batch: a long-running execution (~hours) of 10,000
  board forwards plus 2 x 10,000 x 64 KiB of serial transfer. Not run, not
  claimed as measured.
- Any per-forward number quoted from case 02 (~1.996 s) is a reference only.
