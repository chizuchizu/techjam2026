# Review of `techjam2026/case-10/optimisation/esp32-baseline` (H=2)

Reviewed commit: `9ceac8a` from `chizuchizu/techjam2026`
(esp32-baseline-multi-case), on 2026-08-29.

## What is confirmed

This is the benchmark-shaped Transformer body at
`B=64, S=128, D=128, H=2, F=128, L=4`, causal, with 398,592
parameters (two heads of width 64). The device side fails to link and
is documented as an SRAM anomaly.

| Check | Independent result |
|---|---:|
| XIAO build | FAIL — `dram0_0_seg` overflowed by 23,920 B (needs 345,272 B, have 321,296 B) |
| Physical XIAO seeds 0-4 | not run — no firmware image links |

## Review findings

1. **The H=2 single-board measurement is physically impossible with the
   canonical workspace.** Per-head Q/K/V plus context buffers scale as
   `S * HD = 128 * 64`; together with the fixed 192 KB staging this
   configuration needs 345,272 B against a 321,296 B linker budget. I
   confirmed the deficit twice (fresh `pio run` on this branch).
2. This is a DRAM-capacity failure of the single-board workspace, not
   an accuracy failure. Regeneration and rerun are deterministic.
3. Region shrinking is exhausted: IRAM/data split is SDK-frozen, and
   `-fdata-sections -Wl,--gc-sections` changes nothing because all
   `.bss` buffers are referenced statics. Only firmware layout changes
   (outside the mechanical mandate) can close a 23,920 B gap.
4. H=2 = two head-width-64 shards: the natural multiboard mapping is
   one peer for the full-D projections (C3-proven via case-11) plus one
   peer carrying both 128x64 attention shards.
5. `weights.bin`/`weights_q12.bin`/`testdata/` and `.pio/` are
   gitignored (regenerable). Any collaborator can rebuild and
   regenerate with the committed steps.

## Recommended direction for this case shape

Same class as case-09: DRAM capacity, not numerics. Priority order:
(1) reduced-memory head path (per-token streaming Q/K/V + fused
context write-back, bit-compatible host output) if firmware can change;
(2) otherwise measure H=2 on the multiboard split (two 128x64 shards
behind the full-D projector). No single-board physical number should be
claimed for case-10 until (1) exists.
