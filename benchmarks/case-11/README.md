# Case 11 — 16-head ESP32 Transformer

Case 11 is a head-count variant of the case-2 ESP32 Transformer body:
`B=64, S=128, D=128, H=16, F=128, L=4`, causal. Each of the 16 heads has
width 8 (`TM_HD = D/H = 8`), so head parallelism is fine-grained and the
per-head matrix/attention work per forward is 4x smaller than case 2's
head-width-32 while the full `D=128` projection work is unchanged.

## Configuration

| Parameter | Value |
|---|---|
| Batch size `B` | 64 |
| Sequence length `S` | 128 |
| Hidden size `D` | 128 |
| Heads `H` | 16 |
| Head width `D/H` | 8 |
| FFN width `F` | 128 |
| Layers `L` | 4 |
| Causal | yes |
| Board | Seeed XIAO ESP32-C3, 160 MHz (RV32IMC, no FPU) |
| Numeric modes | EXACT (reference-quality hybrid) and FAST (Q15xQ12 integer GEMM) |

## Directory ownership

| Directory | Contents |
|---|---|
| [`baseline/`](baseline/) | First physical capture and independent review |
| [`optimisation/`](optimisation/) | Maintained complete single-board implementation and optimisation log |

## Comparable complete-forward result

Each row executes the same complete four-layer Transformer body once on
one XIAO ESP32-C3 at 160 MHz. The firmware streams one input frame
(`S*D` floats) per forward; the single-input forward is the measured
physical capture; a complete batch-64 total would be a derived
projection and is not reported.

| Build | Time/forward | Speedup | Validation |
|---|---:|---:|---|
| Current implementation (first physical capture) | **2.462 s** | 1.00x | Pass, 5/5 device seeds |

Single-input forward 2.462 s is the fresh physical capture on PORT B
(`/dev/cu.usbmodem1101`; see the seed0_v1 log for the per-seed reps
timing sweep). The 16 tiny heads add
per-head loop/setup overhead in the hybrid attention path yet the
projection-dominated forward stays close to the case-2 geometry's
single-input time.

## Next case-specific step

The small head width (8) means per-head QK/PV rows are 128x8 (Q15),
so attention is index/loop-bound rather than dot-product-bound: profile
the per-head loop and head-reorder overhead at `H=16` before assuming
head-parallel gains, then design a grouped-head shard (e.g. 4 heads of
width 8) for the multiboard experiment.
