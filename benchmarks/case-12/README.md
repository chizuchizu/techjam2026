# Case 12 — short-sequence ESP32 Transformer

Case 12 is `B=64, S=32, D=128, H=4, F=128, L=4`, causal. The board streams the
official batch of 64 inputs and performs one complete forward per input frame.
A complete batch of 64 inputs has now been streamed and timed end-to-end
on-board; the measured full-case total is **33.8794 s** on the device
(**73.744 s** wall including host USB pacing) — see the result section below.

## Configuration

| Dim | Value | Meaning |
|---|---:|---|
| B | 64 | batch size (streamed; one forward per input frame) |
| S | 32 | sequence length |
| D | 128 | model dimension |
| H | 4 | attention heads (head dim 32) |
| F | 128 | FFN hidden dimension |
| L | 4 | transformer layers |

## Directory ownership

| Directory | Contents |
|---|---|
| [`baseline/`](baseline/) | First physical C3 capture and independent review |
| [`optimisation/`](optimisation/) | Maintained complete single-board implementation and optimisation log |
| [`multiboard/`](multiboard/) | Two-node WiFi data-parallel firmware, raw results, and reproduction steps |

## Comparable complete-forward result

These rows execute the same complete four-layer Transformer body on one XIAO
ESP32-C3 at 160 MHz, one forward per input frame.

| Build | Time/forward | Speedup | Validation |
|---|---:|---:|---|
| Current implementation (first physical capture) | **0.493 s** | 1.00x | Pass, 5/5 device seeds |
| Optimised firmware, full 25-seed run (v1) | **0.529 s** | 0.93x vs first capture | Pass, 25/25 device seeds |

The v1 row was measured on board A (`/dev/cu.usbmodem101`), all 25 official
device seeds PASS (worst `abs_err` 1.14e-03), firmware `TM` sweep 529,015 /
528,970 / 528,764 us per forward. The maintained v1 build is ~7% slower than
the very first physical capture (0.493 s); it is the number to cite for the
maintained optimisation path and is the full 25-seed result, not a projection.
The complete B=64 batch is now measured directly as **33.8794 s** of on-device
compute (sum of the firmware `us=` counters over all 64 streamed frames) and
**73.744 s** wall time including host USB pacing. Raw captures and summaries:
[`optimisation/results/`](optimisation/results/).

## Two-board WiFi data parallelism

The official-shape host gate was reconfirmed first (50/50 FAST + EXACT
seed-runs), followed by a fresh optimized one-board batch (33.928 s, 64/64
PASS). Two direct-WiFi ESP32-C3 replicas then ran 32 inputs each:

| Active boards | Compute wall | End-to-end wall | Scaling | Validation |
|---:|---:|---:|---:|---|
| 1 optimized USB | 33.928 s | 206.0 s * | 1.00x | 64/64 PASS |
| 2 WiFi replicas | **17.091 s** | **29.4 s** | **2.00x vs one WiFi worker** | 64/64 PASS |

`*` The fresh USB run recovered one short output frame, inflating only its
transport-inclusive wall. The earlier clean optimized capture is 33.879 s
compute / 73.744 s USB-inclusive wall.

The short sequence needs no tiling and the WiFi image uses only 104,956 bytes
of static RAM. Flash is tight at 3,125,954 / 3,145,728 bytes (99.4%). Complete
method, raw JSON, and reproduction commands:
[`multiboard/README.md`](multiboard/README.md).

## Likely next step

Short sequences shrink attention work, so setup and dispatch costs matter more
than in case 2. Batch parallelism is now physically validated at 2.00x; the
next optimisation is projection/normalization fusion, or scaling the same
replica method to four/eight nodes.
