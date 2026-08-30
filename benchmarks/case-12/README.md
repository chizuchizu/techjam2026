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

## Likely next step

Short sequences shrink attention work, so setup, dispatch, and transport costs
matter more than in case 2 (attention is only ~17 MFLOP of the ~27 MFLOP
body). Fuse projections and normalization, then compare the single-board
streamed batch path with head or batch parallelism including measured link
cost.

