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
2,162,547 us per forward. Raw capture and summary:
[`optimisation/results/`](optimisation/results/).

## Next case-specific step

`H=2` is (with `H=1`) the SRAM dead end of whole-head parallelism on
the ESP32-C3: head width 64 needs larger per-head staging than the
32-width case-2 reference. Follow-ups are (a) a reduced-memory head
path (streaming Q/K/V rows, fused context write-back, no 128x128
per-head copies) that stays bit-compatible with the baseline's host
output and would bring H=2 within the 321,296-byte dram segment, or
(b) the multiboard split, where H=2 is a two-head shard of width 64 —
one peer runs the full-D projections (the C3 can, as case-11 shows),
the other carries the 128x64 attention. Worth doing only if (a) is
impossible.
