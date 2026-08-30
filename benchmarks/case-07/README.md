# Case 07 — narrow-dimension ESP32 Transformer

Case 07 is `B=64, S=128, D=32, H=4, F=32, L=4`, causal. The board streams the
official batch of 64 inputs and performs one complete forward per input frame.
A complete batch of 64 inputs has now been streamed and timed end-to-end
on-board; the measured full-case total is **30.4272 s** on the device
(**70.227 s** wall including host USB pacing) — see the result section below.

## Configuration

| Dim | Value | Meaning |
|---|---:|---|
| B | 64 | batch size (streamed; one forward per input frame) |
| S | 128 | sequence length |
| D | 32 | model dimension |
| H | 4 | attention heads (head dim 8) |
| F | 32 | FFN hidden dimension |
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
| Current implementation (first physical capture) | **0.491 s** | 1.00x | Pass, 5/5 device seeds |
| Optimised firmware, full 25-seed run (v1) | **0.475 s** | 1.03x | Pass, 25/25 device seeds |

The 0.475 s/forward v1 row was measured on board A (`/dev/cu.usbmodem101`)
with all 25 official device seeds passing the benchmark gate (worst
`abs_err` 1.46e-03), plus a firmware `TM` sweep of 475,099 / 474,979 /
475,172 us per forward. Raw per-seed capture and summary live in
[`optimisation/results/`](optimisation/results/). The complete-batch total is
now a measured number as well: streaming all B=64 frames on-board took
**30.4272 s** of device compute (sum of the firmware `us=` counters) and
**70.227 s** of wall time including host USB pacing — see
`case-07_xiao_c3_optimised_full_case_v1.md` in the same directory.

## Two-board WiFi data parallelism

After reconfirming the official-shape host gate (50/50 FAST + EXACT seed-runs)
and a fresh optimized one-board physical batch (31.006 s, 64/64 PASS), two
WiFi ESP32-C3 workers ran 32 inputs each:

| Active boards | Compute wall | End-to-end wall | Scaling | Validation |
|---:|---:|---:|---:|---|
| 1 optimized USB | 31.006 s | 203.7 s * | 1.00x | 64/64 PASS |
| 2 WiFi replicas | **15.822 s** | **28.6 s** | **2.00x vs one WiFi worker** | 64/64 PASS |

`*` The fresh USB run recovered one short output frame, inflating only its
transport-inclusive wall. No input was omitted; device compute and all 64
accuracy checks completed. The earlier clean optimized capture is 30.427 s
compute / 70.227 s USB-inclusive wall.

The WiFi build avoids the SRAM weight cache and links at only 108,300 bytes
static RAM, so this narrow case does not need sequence tiling. Complete method,
raw JSON, and reproduction commands:
[`multiboard/README.md`](multiboard/README.md).

## Likely next step

D and F of 32 make the projection and FFN matrices tiny, so dispatch, loop,
and serialization overhead dominate the small GEMMs. Fuse the narrow
projections and normalization. Data parallelism is already validated at 2.00x;
the next hardware step is four/eight-node scaling after the same two-board
evidence ladder is completed for case 12.
