# Case 12 — short-sequence ESP32 Transformer

Case 12 is `B=64, S=32, D=128, H=4, F=128, L=4`, causal. The board streams the
official batch of 64 inputs and performs one complete forward per input frame.
No complete-batch total is stated here because it would be a derived projection.

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
| Current implementation (first physical capture) | **0.493 s** | 1.00x | Pass, 5/5 device seeds + 25/25 host checks |

The 0.493 s/forward (per-seed host-measured time) was measured on board A
(`/dev/cu.usbmodem101`): 5/5 device seeds pass the benchmark gate and the
firmware's own counter reported 492,020 us and 492,259 us per forward. The host gate passes all 25 seeds in both FAST and
EXACT modes (50/50 seed-runs, 0 failed). A complete-batch total would
be a derived projection and is not reported; only real on-board
measurements appear here.

## Likely next step

Short sequences shrink attention work, so setup, dispatch, and transport costs
matter more than in case 2 (attention is only ~17 MFLOP of the ~27 MFLOP
body). Fuse projections and normalization, then compare the single-board
streamed batch path with head or batch parallelism including measured link
cost.

