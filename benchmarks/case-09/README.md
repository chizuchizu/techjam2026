# Case 9 — 1-head ESP32 Transformer

Case 9 is a head-count variant of the case-2 ESP32 Transformer body:
`B=64, S=128, D=128, H=1, F=128, L=4`, causal. A single head of width
128 (`TM_HD = D/H = 128`) means all attention work is in one head; the
per-head buffers are at their largest and the mixed Q15xQ12
attention/projection path has no head loop at all.

## Configuration

| Parameter | Value |
|---|---|
| Batch size `B` | 64 |
| Sequence length `S` | 128 |
| Hidden size `D` | 128 |
| Heads `H` | 1 |
| Head width `D/H` | 128 |
| FFN width `F` | 128 |
| Layers `L` | 4 |
| Causal | yes |
| Board | Seeed XIAO ESP32-C3, 160 MHz (RV32IMC, no FPU) |
| Numeric modes | EXACT (reference-quality hybrid) and FAST (Q15xQ12 integer GEMM) |

## Directory ownership

| Directory | Contents |
|---|---|
| [`baseline/`](baseline/) | SRAM-limit record and review |
| [`optimisation/`](optimisation/) | Maintained complete single-board implementation and optimisation log |
| [`multiboard/`](multiboard/) | Four-node WiFi data-parallel result and reproduction steps |

## Comparable complete-forward result

The earlier baseline report stated the firmware could not link on one XIAO
ESP32-C3 (`dram0_0_seg` overflowed at `H=1`). The maintained
`optimisation/esp32-baseline` firmware fixed that: the `H==1` `g_kh` alias was
reverted to a real buffer, and current framework libs were minimised
(`xIsrStack`, coredump stack, `prstatus`) in `patched_sdk_libs_current/`. The
firmware now links (RAM 273,180 / 327,680 B, 83.4%) and runs on the board.

| Build | Time/forward | Speedup | Validation |
|---|---:|---:|---|
| Optimised firmware, full 25-seed run (v1) | **2.157 s** | 1.00x | Pass, 25/25 device seeds |

Measured on board B (`/dev/cu.usbmodem1101`), all 25 official device seeds
PASS (worst `abs_err` 1.24e-03), firmware `TM` sweep 2,154,695 / 2,155,128 /
2,155,441 us per forward. The complete B=64 batch is now measured directly:
**138.0273 s** of on-device compute for the whole case (sum of the firmware
`us=` counters over all 64 streamed frames) and **262.073 s** wall time
including host USB pacing. Raw captures and summaries:
[`optimisation/results/`](optimisation/results/).

## WiFi data parallelism

The reduced-memory path is now implemented as an opt-in 16-row sequential
tile schedule with a persistent WiFi/TCP endpoint. It links at **224,244 B**
static RAM instead of the default build's 273,180 B. The tiled host gate passes
25/25 seeds (worst `max_abs=1.1038e-3`), and the first two physical workers
both passed a seed-0 TCP forward at 3.563 s device compute with zero failing
elements.

The complete B=64 batch was then distributed evenly across four physical
workers:

| Active boards | Compute wall | End-to-end wall | Scaling | Validation |
|---:|---:|---:|---:|---|
| 1 optimized USB | **138.027 s** | 262.073 s | 1.00x | 64/64 PASS |
| 4 tiled WiFi replicas | **57.005 s** | **75.4 s** | **4.00x vs one tiled worker** | 64/64 PASS |

Tiling makes one WiFi worker slower than the best SRAM-cached USB build, so
the measured cluster gain against that best single-board compute total is
**2.42x**, not 4.00x. The four-node run had no missing inputs or failing
elements; worst `max_abs=1.2649e-3`. Raw evidence and commands are in
[`multiboard/README.md`](multiboard/README.md). The next hardware step is the
same complete gate on eight replicas.
