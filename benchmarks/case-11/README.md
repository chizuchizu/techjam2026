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
| [`multiboard/`](multiboard/) | Two- and four-node WiFi data-parallel results and reproduction steps |

## Comparable complete-forward result

Each row executes the same complete four-layer Transformer body once on
one XIAO ESP32-C3 at 160 MHz. The firmware streams one input frame
(`S*D` floats) per forward; the single-input forward is the measured
physical capture; the complete batch-64 total is now measured directly
on-board as **138.6104 s** of device compute (sum of the firmware `us=`
counters over all 64 streamed frames) with **262.754 s** wall time including
host USB pacing.

| Build | Time/forward | Speedup | Validation |
|---|---:|---:|---|
| Current implementation (first physical capture) | **2.462 s** | 1.00x | Pass, 5/5 device seeds |
| Optimised firmware, full 25-seed run (v1) | **2.166 s** | 1.14x | Pass, 25/25 device seeds |

The v1 row was measured on board B (`/dev/cu.usbmodem1101`), all 25 official
device seeds PASS (worst `abs_err` 1.24e-03), firmware `TM` sweep 2,163,993 /
2,164,242 / 2,164,130 us per forward. The 16 tiny heads add per-head loop/setup
overhead yet the projection-dominated forward lands at ~2.17 s, and the full B=64 case
lands at 138.6104 s of device compute. Raw captures and summaries:
[`optimisation/results/`](optimisation/results/).

## WiFi data parallelism

The opt-in WiFi worker uses a 16-row sequential tile schedule. Its activation
arena scales with tile height rather than the complete `S=128` sequence,
reducing the credential-enabled build to **158,964 / 327,680 B static RAM**.
The default optimized USB build remains unchanged at 256,180 B.

The tiled host gate passes 25/25 seeds (worst `max_abs=1.1135e-3`). Two
physical workers passed a seed-0 TCP forward at 6.452 / 6.451 s. Complete
B=64 runs on two and four physical workers produced:

| Build | Compute wall | End-to-end wall | Validation |
|---|---:|---:|---|
| 1 optimized USB worker | **138.610 s** | 262.754 s | 25/25 device seeds PASS |
| 1 tiled WiFi worker (equivalent) | **412.707 s** | - | Derived from measured replica work |
| 2 tiled WiFi replicas | **206.354 s** | **238.0 s** | **64/64 PASS**, zero failing elements |
| 4 tiled WiFi replicas | **103.169 s** | **119.3 s** | **64/64 PASS**, zero failing elements |

Two replicas scale exactly **2.00x** against one tiled worker, but remain
**1.49x slower** than the best optimized USB compute total. H=16 performs
sixteen causal softmaxes per layer, so the memory-saving tile schedule has
more overhead here than in cases 9 and 10. Four replicas preserve **4.00x**
compute scaling and cross over: their 103.169 s compute wall is **1.34x
faster** than the optimized single-board USB result. Raw evidence and commands
are in [`multiboard/README.md`](multiboard/README.md).
