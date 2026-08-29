# Case 4 — batch-16 ESP32 Transformer

Case 4 is a batch variant of the case-2 ESP32 Transformer body:
`B=16, S=128, D=128, H=4, F=128, L=4`, causal. The 16 independent
inputs stream through one board; the per-input geometry (S/D/H/F/L) is
identical to case 2, so the case-2 device-verified single-input forward
applies to each input. Only the per-input forward is measured on-board
(1.990 s); a complete batch total would be a derived projection and is
deliberately not reported.

## Configuration

| Parameter | Value |
|---|---|
| Batch size `B` | 16 |
| Sequence length `S` | 128 |
| Hidden size `D` | 128 |
| Heads `H` | 4 |
| FFN width `F` | 128 |
| Layers `L` | 4 |
| Causal | yes |
| Board | Seeed XIAO ESP32-C3, 160 MHz (RV32IMC, no FPU) |
| Numeric modes | EXACT (reference-quality hybrid) and FAST (Q15xQ12 integer GEMM) |

## Directory ownership

| Directory | Contents |
|---|---|
| [`baseline/`](baseline/) | Physical starting capture and independent review |
| [`optimisation/`](optimisation/) | Maintained complete single-board implementation and optimisation log |

## Comparable complete-forward result

Each row executes the complete four-layer Transformer body once for a
single input on one XIAO ESP32-C3 at 160 MHz (one forward per input
frame; the batch of 16 inputs streams as 16 sequential
single-input forwards).

| Build | Per-input forward | Speedup | Validation |
|---|---:|---:|---|
| Current implementation (first physical capture) | 1.990 s | 1.00x | Pass, 5/5 device seeds |

Per-input forward measured on-board (board A): 1.990 s at the same
S/D/H/F/L geometry, 5/5 device seeds
(see [`baseline/results/case-04_esp32_baseline_seed0_v1.log`](baseline/results/case-04_esp32_baseline_seed0_v1.log)).
A complete batch-16 total would be a derived
projection and is deliberately not reported.

## Next case-specific step

Tile the batch-16 input stream so resident weights stay in SRAM and
compare single-board streaming with batch-parallel multiboard dispatch.
Record per-batch latency and steady-state throughput; do not present
pipeline fill time as single-sample latency.

## Multiboard result

Data parallel across two physical XIAO ESP32-C3 boards: the 16 inputs are independent forwards over the same weights, so input `i` runs on board `i % N` and the boards exchange nothing.

| Boards | Batch time | Speedup | Gate |
|---:|---:|---:|---|
| 1 | 31.8 s | 1.00x | Pass |
| 2 | **15.9 s** | **2.00x** | Pass, 0 failing elements |

Implementation and method: [`../batch-dp/`](../batch-dp/); results: [`../batch-dp/RESULTS_TWO_C3.md`](../batch-dp/RESULTS_TWO_C3.md).
