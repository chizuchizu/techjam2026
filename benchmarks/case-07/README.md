# Case 07 — narrow-dimension ESP32 Transformer

Case 07 is `B=64, S=128, D=32, H=4, F=32, L=4`, causal. The board streams the
official batch of 64 inputs and performs one complete forward per input frame.
No complete-batch total is stated here because it would be a derived projection.

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

## Comparable complete-forward result

These rows execute the same complete four-layer Transformer body on one XIAO
ESP32-C3 at 160 MHz, one forward per input frame.

| Build | Time/forward | Speedup | Validation |
|---|---:|---:|---|
| Current implementation (first physical capture) | **0.491 s** | 1.00x | Pass, 5/5 device seeds |

The 0.491 s/forward was measured on board A (`/dev/cu.usbmodem101`) with 5/5
device seeds passing the benchmark gate and is reused here per the case spec
(no reflash). A complete-batch total would be a derived projection and is
not reported; only real on-board measurements appear here.

## Likely next step

D and F of 32 make the projection and FFN matrices tiny, so dispatch, loop,
and serialization overhead dominate the small GEMMs. Fuse the narrow
projections and normalization, then compare that single-board path with batch
parallelism. Do not assume head distribution amortises network cost when each
forward is under 0.5 s.
