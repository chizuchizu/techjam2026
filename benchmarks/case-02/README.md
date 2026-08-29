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

These rows execute the same complete four-layer Transformer body on one XIAO
ESP32-C3 at 160 MHz.

| Build | Time/forward | Speedup | Validation |
|---|---:|---:|---|
| Initial hybrid baseline | 42.15 s | 1.00x | Pass, 5/5 device seeds |
| Current optimised firmware | **5.27 s** | **8.0x** | Pass, 5/5 device seeds and 50/50 host checks |

## Multiboard results

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

Complete one distributed layer by adding output projection, both residuals,
the second LayerNorm, and FFN to the verified multiboard path. Only then extend
the same partition across all four layers and report an end-to-end distributed
case-2 result.
