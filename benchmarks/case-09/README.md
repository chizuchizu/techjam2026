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

> **SRAM anomaly — not measurable on a single XIAO ESP32-C3.**
> **The firmware does not link on the board**: the linker reports that
> region `dram0_0_seg` is overfull by
> **73,072 bytes** at `H=1`. The per-head Q/K/V/context buffers scale as
> `S * HD` (inverse of `H`), so the single-head configuration has the
> largest working set of all eight head-count cases. No physical device
> measurement is possible without firmware changes outside the
> mechanical baseline mandate. See [`baseline/`](baseline/) for the
> full record, region budget, and the three hardware alternatives.

| Build | Time/forward | Speedup | Validation |
|---|---:|---:|---|
| Current implementation (first physical capture) | not measurable — linker `dram0_0_seg` overflowed by 73,072 B | — | not run (no firmware image) |

The device measurement is physically impossible on this board with the
current workspace.

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
