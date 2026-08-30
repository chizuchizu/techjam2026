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

## Likely next step

D and F of 32 make the projection and FFN matrices tiny, so dispatch, loop,
and serialization overhead dominate the small GEMMs. Fuse the narrow
projections and normalization, then compare that single-board path with batch
parallelism. Do not assume head distribution amortises network cost when each
forward is under 0.5 s.
