# Review of `techjam2026/case-09/optimisation/esp32-baseline` (H=1)

Reviewed commit: `9ceac8a` from `chizuchizu/techjam2026`
(esp32-baseline-multi-case), on 2026-08-29.

## What is confirmed

This is the benchmark-shaped Transformer body at
`B=64, S=128, D=128, H=1, F=128, L=4`, causal, with 398,592
parameters (single head of width 128). The device side fails to link
and is documented as an SRAM anomaly.

| Check | Independent result |
|---|---:|
| XIAO build | FAIL — `dram0_0_seg` overflowed by 73,072 B (needs 394,424 B, have 321,296 B) |
| Physical XIAO seeds 0-4 | not run — no firmware image links |

## Review findings

1. **The H=1 single-board measurement is physically impossible with the
   canonical workspace.** The per-head Q/K/V plus context buffers scale
   as `S * HD = 128 * 128`, and combined with the fixed 192 KB of
   `g_x/g_buf1/g_buf2`/`a16` staging this configuration needs
   394,424 B of static+heap DRAM against a 321,296 B linker budget. I
   confirmed the deficit twice (fresh `pio run` on this branch).
2. This is a DRAM-capacity failure of the single-board workspace, not
   an accuracy failure. Regenerating the weights/vectors is
   deterministic.
3. Region shrinking is exhausted: IRAM/data split is SDK-frozen, and
   `-fdata-sections -Wl,--gc-sections` changes nothing, because all
   `.bss` buffers are referenced statics. The only levers are firmware
   layout changes, which are outside the mechanical baseline mandate.
4. The single-head configuration is the attention-subgraph-primitive
   reference for a multiboard split: full-D projections fit and run on
   a single C3 (measured on case-11), and the 128-wide causal attention
   here is exactly one per-head shard of the case-2 geometry.
5. `weights.bin`/`weights_q12.bin`/`testdata/` and `.pio/` are
   gitignored (regenerable). Steps, manifest and exporter are
   committed, so any collaborator can rebuild and regenerate.

## Recommended direction for this case shape

The H=1/H=2 class is a DRAM-capacity problem on one C3, not a
numerical or code-quality one. Priority order: (1) reduced-memory head
path (stream per-token Q/K/V rows + fused context write-back, keeping
the host output bit-compatible) if a team member can change firmware;
(2) otherwise treat H=1 as the projector + attention-subgraph primitive
for the multiboard split and measure on the split topology. Do not
claim a single-board physical number for case-09 until (1) exists.
