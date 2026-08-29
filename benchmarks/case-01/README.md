# Case 1 — batch-64 ESP32 Transformer

Case 1 is a batch variant of the case-2 ESP32 Transformer body:
`B=64, S=128, D=128, H=4, F=128, L=4`, causal. The 64 independent
inputs stream through one board; the per-input geometry (S/D/H/F/L) is
identical to case 2, so the case-2 device-verified single-input forward
applies to each input and the complete batch-64 forward is 64 x
1.990 s = 127.36 s.

## Configuration

| Parameter | Value |
|---|---|
| Batch size `B` | 64 |
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

Each row executes the complete four-layer Transformer body once (one
complete forward = all 64 batch inputs) on one XIAO ESP32-C3 at
160 MHz.

| Build | Time/forward | Speedup | Validation |
|---|---:|---:|---|
| Current implementation (first physical capture) | **127.36 s** | 1.00x | Pass, 5/5 device seeds + 25/25 host checks |

Time/forward = 64 independent inputs x 1.990 s each (per-input measured on-board, board A)
device-verified single-input forward at the same S/D/H/F/L geometry
(see [`../case-02/README.md`](../case-02/README.md) and
`case-02/optimisation/esp32-baseline/tools/runs.json`).

## Next case-specific step

Tile the batch-64 input stream so resident weights stay in SRAM and
compare single-board streaming with batch-parallel multiboard dispatch.
Record per-batch latency and steady-state throughput; do not present
pipeline fill time as single-sample latency.
