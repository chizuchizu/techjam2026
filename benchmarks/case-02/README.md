# Case 02 — latency-focused ESP32 Transformer

Case 2 is the only official case currently implemented on ESP32. Its shape is
`B=1, S=128, D=128, H=4, F=128, L=4`, causal. Batch 1 makes single-input
latency the primary objective and exposes four independent attention heads for
multiboard experiments.

## Directory ownership

| Directory | Contents |
|---|---|
| [`baseline/`](baseline/) | First physical C3 capture and independent review |
| [`optimisation/`](optimisation/) | Maintained complete single-board implementation and optimisation log |
| [`multiboard/`](multiboard/) | Worker firmware, coordinators, link benchmark, and distributed results |

## Comparable complete-forward result

These rows execute the same complete four-layer Transformer body, on Seeed
XIAO ESP32-C3 boards at 160 MHz.

| Build | Boards | Time/forward | Speedup | Validation |
|---|---:|---:|---:|---|
| Initial hybrid baseline | 1 | 42.15 s | 1.00x | Pass, 5/5 device seeds |
| Optimised firmware (opt23) | 1 | 1.996 s | 21.1x | Pass, 25/25 device seeds and 54/54 host checks |
| [`esp32-cluster-full`](multiboard/esp32-cluster-full/) | 2 | **1.276 s** | **33.0x** | Pass, 25/25 host checks, 0 failing elements |

The two-board row splits one forward by token row; against the single-board
figure measured in the same session (1.990 s) that is 1.56x. Host serial
transfer is excluded from every row.

The case-2 optimised firmware was also run against cases 1, 3, 4, and 5
(same S/D/H/F/L geometry, batch variants). Measured batch totals and the
unoptimized-baseline comparison are in
[`../case2_code_on_cases_1_to_5.md`](../case2_code_on_cases_1_to_5.md).

## Complete distributed result

The split is by token row (`i % 2`): every case-2 operator is per-token except
causal attention, so the only inter-board traffic is one K/V exchange per layer
— 131,200 bytes each way per forward, streamed per head over UDP and overlapped
with arithmetic down to 5–88 ms of measured waiting. Details in
[`multiboard/results/CASE2_FULL_E2E_RESULTS.md`](multiboard/results/CASE2_FULL_E2E_RESULTS.md).

## Earlier partial-scope multiboard results

These results use the case-2 shape but cover only the stated partial scope.

| Scope | Boards | Wall time | Speedup | Validation |
|---|---:|---:|---:|---|
| Layer-0 LayerNorm + Q/K/V + causal attention | 1 C3 | 9.693 s | 1.00x | Pass |
| Layer-0 LayerNorm + Q/K/V + causal attention | 2 C3s | **4.850 s** | **2.00x** | Pass, 5/5 seeds |
| Four independent causal attention heads | 1 C3 | 3.000 s average | 1.00x | Pass |
| Four independent causal attention heads | 4 C3s | **0.766 s** | **3.92x** | Pass, zero failed elements |

The partial-layer path excludes output projection, both residual paths, the
second LayerNorm, FFN, final norm, and the other three layers. The four-board
row additionally excludes the first LayerNorm and Q/K/V projections. Neither
is a complete distributed case-2 time.

## Next case-specific step

Extend the row partition from two boards to four (`i % N`), and cut the
replicated per-board weight streaming that stands between 1.56x and 2x.
