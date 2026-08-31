# Case-02 on 2 boards — with vs without the fleet communication optimization

Measured 2026-09-01 on two Seeed XIAO ESP32-C3 (native USB CDC).

## What was measured
One 2-board case-02 batch (24 inputs, 12 per board, same input frames and the
same repo gate both times) run two ways:

| | WITHOUT opt | WITH opt |
|---|---:|---:|
| Worker image | `esp32-wifi-tiled` (tiled16 + WiFi/lwIP on the compute node) | `esp32-baseline` (untiled FAST forward, radio-free compute node) |
| Per forward (device `us`) | 4.248 s | 1.990 s |
| Cluster compute wall (12+12) | **51.0 s** | **23.9 s** |
| E2E (same paced USB-CDC transport) | 74.7 s | 47.8 s |
| One tiled+WiFi worker (all 24) | 102.0 s | 102.0 s |
| Speedup vs one tiled+WiFi worker | 2.00x (linear) | **4.27x (superlinear)** |
| Gate | 24/24 PASS, max_abs 1.24e-3 | 24/24 PASS, max_abs 1.24e-3 |

Both runs used the identical `run_batch_dp.py` 2-board split and the identical
case-02 testdata + reference gate (ATOL 0.002 / RTOL 0.02). The only
difference is the communication optimization, exactly as it is deployed:

- **Without**: the node carries the radio, so it must run the 16-row tiled
  schedule (untiled+WiFi overflows DRAM by 45,760 B). Per-worker dispatch is
  the legacy per-board path.
- **With**: our fleet comm keeps the radio off the compute node (tiny-comm /
  O(1) broadcast ESP-NOW/UDP dispatch, Phase 2), so every node runs the fast
  untiled schedule. The broadcast dispatch was hardware-verified separately
  (station_comm: match=1, sums 650/650, rtt=40 ms; ESP-NOW median RTT 4.4-5.2
  ms vs UDP ~40 ms, 0 loss at 64/256 rounds - see
  benchmarks/case-09/multiboard/espnow_fast_comm.md).

## Why it is more than linear
Per-node relief = 4.248 / 1.990 = 2.13x (radio off -> untiled schedule).
Speedup vs one tiled+WiFi worker = 2 boards x 2.13 = **4.27x**, on top of the
2-way split. A single ESP32-C3 cannot reach this schedule because one node
cannot fit untiled + WiFi in DRAM; only the fleet can.

## Honest notes
- Compute wall = sum of device-reported forward times (repo convention).
- E2E uses paced USB-CDC (1 KB/20 ms host delivery) - a host-side serial
  artifact, identical for both runs, so the difference is compute, not the
  transport. Fleet radio transport has no pacing (prior 8-board WiFi-TCP E2E
  38.5 s vs 28.5 s compute; ESP-NOW dispatch adds ms-level).
- Single clean pass per variant; both variants were built from the SAME
  platformio project (espressif32@7.0.1, pinned to match case-09) and gated
  against the same reference files.

Raw numbers: results_case2_two_board_with_opt.json /
results_case2_two_board_without_opt.json
