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

## Next case-specific step

`H=1` is the single-board dead end of whole-head parallelism on the
ESP32-C3. The relevant follow-ups are (a) a reduced-memory head path
(streaming Q/K/V rows, 16-bit residual, no full 128x128 context copies)
that stays bit-compatible with the baseline's host output and would
bring H=1/H=2 within the 321,296-byte dram segment, or (b) the
multiboard split, where H=1 is trivially the single-board
attention-subgraph primitives (per-head shards of width 128, all one
node) — the C3 can carry the full-D projections while a peer carries
attn. Worth doing only if (a) is impossible.
