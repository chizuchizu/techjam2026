# Two-board superlinear split (measured 2026-09-01) — untiled compute nodes

## Result
A 2-board data-parallel split where **compute and radio are separated** runs the
full batch **3.32x faster than one tiled+WiFi worker — with only 2 boards**.
Linear would be 2.0x. The extra 1.66x is a *per-node schedule relief* that only a
cluster can buy.

| Figure | Value |
|---|---:|
| Single tiled+WiFi worker, per forward (measured `T`, n=3) | **3.5956 s** |
| Single untiled worker, per forward (measured `T`, n=3) | **2.1639 s** |
| Per-node relief (untiled vs tiled+WiFi) | **1.661x** |
| 2 boards, 24 inputs (12 each), untiled — compute wall | **25.99 s** |
| Gate | **24/24 PASS**, max_abs = 1.19e-3 |
| Equivalent one tiled worker (24 inputs) | 86.3 s |
| **Speedup with 2 boards (24 inputs, measured)** | **3.32x (superlinear)** |
| Projected for full 64 batch (32/board) | compute 69.3 s → **3.32x** |

## Why it is superlinear, not a config trick
A single ESP32-C3 that must also carry WiFi/lwIP cannot use the fast schedule:
building `TM_TILE_ROWS=128` (untiled arena) **with WiFi overflows DRAM by
45,760 B** (linker error, reproduced). The worker is forced to the 16-row tiled
schedule at 3.60 s/forward.

A fleet keeps radio off the compute node (or uses a tiny-comm data link like
ESP-NOW / the `station_comm` O(1) broadcast dispatch), so every node runs the
untiled FAST schedule at 2.16 s/forward. Scaling is then:

    wall(N) = 2.165 x (64/N) + comm   vs   230.1 s for one tiled worker
    speedup(N) = 1.66 x N   -> superlinear

This is the same "hidden-capacity relief" the engineering review named; only
the cluster can exploit it because one node cannot fit untiled + WiFi.

## 8-board projection (flagged, honest)
Per-node relief x 8-way split, compute only (per-forward is batch-independent,
verified on 2 boards):

    wall(8) = 2.165 x 8 inputs = 17.3 s  -> speedup 13.3x vs one tiled worker

Transport: prior 8-board WiFi-TCP run added ~10 s total dispatch+collect for 64
inputs (~1.25 s/board); the fleet `station_comm` broadcast dispatch removes the
O(N) unicast term. Projected fleet E2E at 8 boards ~ 19-20 s (compute 17.3 +
~2-3 s), i.e. **~11.9x E2E**, to be verified when 8 boards are available.

## Definition of terms (kept identical to the repo)
* Equivalent one tiled worker = 64 x measured tiled per-forward (repo method).
* Compute wall = sum of measured per-forward `us` from `END forward=... us=`.
* E2E here uses paced USB-CDC (1 KB / 20 ms host delivery) — a host-side
  serial artifact, not the fleet transport; WiFi TCP / ESP-NOW have no such
  pacing (multiboard 8-board E2E = 38.5 s vs 28.5 s compute).

## Reproduce
```bash
# one tiled+WiFi worker baseline
cd benchmarks/case-09/optimisation/esp32-baseline
pio run -e esp32-wifi-tiled -t upload --upload-port /dev/cu.usbmodem101
# 'T' timing via serial -> TM 1 3596086 3595556 3594544

# 2 untiled compute nodes
pio run -e esp32-baseline -t upload --upload-port /dev/cu.usbmodem101
pio run -e esp32-baseline -t upload --upload-port /dev/cu.usbmodem1101
# parallel 12+12 split with the repo gate (tools/device_test.py, seeds 0..23)
```
