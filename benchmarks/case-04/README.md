# Case 4 — batch-16 ESP32 Transformer

Case 4 is a batch variant of the case-2 ESP32 Transformer body:
`B=16, S=128, D=128, H=4, F=128, L=4`, causal. The 16 independent
inputs stream through one board; the per-input geometry (S/D/H/F/L) is
identical to case 2, so the case-2 device-verified single-input forward
applies to each input (1.990 s measured on-board in FAST mode). The full
batch-16 total is measured on-device and reported below.

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

Each row executes the complete four-layer Transformer body on one XIAO
ESP32-C3 at 160 MHz in FAST mode. The batch-16 total is 16 sequential
single-input forwards timed end-to-end on-device with the case-2 optimised
firmware (measured 2026-08-29).

| Build | Per-input forward | Batch-16 total | Speedup | vs 5-min cutoff | Validation |
|---|---:|---:|---:|---:|---|
| Unoptimized fp32 baseline (estimated) | 42.15 s | 674.4 s (11.24 min) | 1.00x | time limit exceeded | — |
| **Case-2 optimised firmware (measured)** | **1.990 s** | **31.84 s** | **21.2x** | OK | Pass, 5/5 device seeds; host 50/50 (worst FAST 1.03e-3) |

Baseline total = 16 x 42.15 s/input (initial unoptimized fp32/hybrid forward
from the case-2 log). Optimised total measured on-board: 31.84 s
device time (35.5 s wall incl. serial transfer). Per-input forward log:
[`baseline/results/case-04_esp32_baseline_seed0_v1.log`](baseline/results/case-04_esp32_baseline_seed0_v1.log).
Full method and cross-case table:
[`../case2_code_on_cases_1_to_5.md`](../case2_code_on_cases_1_to_5.md).

## Next case-specific step

Tile the batch-16 input stream so resident weights stay in SRAM and
compare single-board streaming with batch-parallel multiboard dispatch.
Record per-batch latency and steady-state throughput; do not present
pipeline fill time as single-sample latency.

## Results

Whole-case time: all 16 inputs of the batch, device measurement of the
complete four-layer body, host serial transfer excluded.

| Build | Boards | Batch of 16 | Speedup | Validation |
|---|---:|---:|---:|---|
| Unoptimised baseline | 1 | 674.4 s * | 1.00x | estimated, see note |
| Optimised firmware (opt23) | 1 | 31.84 s | 21.2x | Pass, 50/50 host checks |
| Data parallel | 2 | **15.9 s** | **42.4x** | Pass, 16/16 forwards, 0 failing elements |

`*` This case was never run on the pre-optimisation firmware, so its baseline
is estimated as `16 x 42.15 s` from case 2's measured starting point. The
other two rows are measured.

The 16 inputs are independent forwards over the same weights, so input `i`
runs on board `i % N` and the boards exchange nothing - which is why two boards
give exactly 2.00x over the optimised single board.

Implementation and method: [`../batch-dp/`](../batch-dp/); results:
[`../batch-dp/RESULTS_TWO_C3.md`](../batch-dp/RESULTS_TWO_C3.md).
