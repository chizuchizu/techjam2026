# Case 10 — 2-head ESP32 Transformer

Case 10 is a head-count variant of the case-2 ESP32 Transformer body:
`B=64, S=128, D=128, H=2, F=128, L=4`, causal. Two heads of width 64
(`TM_HD = D/H = 64`) keep the full-D projection work unchanged while
halving the per-head attention rows relative to case 2.

## Configuration

| Parameter | Value |
|---|---|
| Batch size `B` | 64 |
| Sequence length `S` | 128 |
| Hidden size `D` | 128 |
| Heads `H` | 2 |
| Head width `D/H` | 64 |
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
| [`multiboard/`](multiboard/) | Two-node WiFi data-parallel result and reproduction steps |

## Comparable complete-forward result

The earlier baseline report stated the firmware could not link on one XIAO
ESP32-C3 (`dram0_0_seg` overflowed at `H=2`). The maintained
`optimisation/esp32-baseline` firmware links and runs on the board with a
reduced working set, so this table now reports the physical measurement.

| Build | Time/forward | Speedup | Validation |
|---|---:|---:|---|
| Optimised firmware, full 25-seed run (v1) | **2.165 s** | 1.00x | Pass, 25/25 device seeds |

Measured on board A (`/dev/cu.usbmodem101`), all 25 official device seeds
PASS (worst `abs_err` 1.24e-03), firmware `TM` sweep 2,162,329 / 2,162,943 /
2,162,547 us per forward. The complete B=64 batch is now measured directly:
**138.5358 s** of on-device compute for the whole case (sum of the firmware
`us=` counters over all 64 streamed frames) and **262.778 s** wall time
including host USB pacing. Raw captures and summaries:
[`optimisation/results/`](optimisation/results/).

An independent repeat measured **139.264 s** (+0.526% versus v1) and passed
64/64 with zero failing elements. See
[`../SINGLE_BOARD_REPEAT_CASES_09_11.md`](../SINGLE_BOARD_REPEAT_CASES_09_11.md).

## WiFi data parallelism

The opt-in WiFi worker uses a 16-row sequential tile schedule. Its activation
arena scales with the tile height instead of all `S=128` rows, reducing the
complete-forward build to **189,428 / 327,680 B static RAM including
WiFi/lwIP**. The default optimized USB build remains unchanged and uses
265,324 B.

The tiled host gate passes 25/25 seeds (worst `max_abs=1.1496e-3`). Both
physical workers passed the seed-0 TCP smoke test at 3.722 / 3.719 s with zero
failing elements. The complete official B=64 batch then produced:

| Build | Compute wall | End-to-end wall | Validation |
|---|---:|---:|---|
| 1 optimized USB worker | **138.536 s** | 262.778 s | 25/25 device seeds PASS |
| 1 tiled WiFi worker (equivalent) | **238.2 s** | - | Derived from the replicas' measured work |
| 2 tiled WiFi replicas | **119.101 s** | **146.7 s** | **64/64 PASS**, zero failing elements |
| 4 tiled WiFi replicas | **59.563 s** | **74.2 s** | **64/64 PASS**, zero failing elements |
| 8 tiled WiFi replicas | **29.793 s** | **39.2 s** | **64/64 PASS**, zero failing elements |

The two-, four-, and eight-node runs scale **2.00x**, **4.00x**, and **7.999x**
against one tiled worker. Against the best optimized single-board compute
total, they are **1.16x**, **2.33x**, and **4.65x** faster. Tiling is the
memory enabler, not a per-board speed optimization: one tiled forward is
3.722 s versus 2.165 s on the optimized USB build. Raw evidence and commands are in
[`multiboard/README.md`](multiboard/README.md).
