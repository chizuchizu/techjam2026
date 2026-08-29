# Review of `techjam2026/case-11/optimisation/esp32-baseline`

Reviewed commit: `dd13a14` from `chizuchizu/techjam2026` (the case-11
landing commit, esp32-baseline-multi-case), on 2026-08-29.

## What is confirmed

The teammate implementation is the benchmark-shaped Transformer body at
`B=64, S=128, D=128, H=16, F=128, L=4`, causal, with 398,592
parameters (the weight layout is independent of B). Each of the 16
heads has width 8. It includes LayerNorm, streaming causal attention,
residuals, GELU FFNs, and final LayerNorm. FAST mode uses Q15
activations and Q12 weights for the six projection matrices per block.

I regenerated the omitted deterministic weights/test vectors, built
the PlatformIO firmware, flashed it to the physical XIAO ESP32-C3 on
PORT B (`/dev/cu.usbmodem1101`), and ran device seeds 0-4 end to end.
The concise physical-device capture is
[`case-11_esp32_baseline_seed0_v1.log`](case-11_esp32_baseline_seed0_v1.log).

| Check | Independent result |
|---|---:|
| XIAO build RAM | 256,180 / 327,680 B (78.2%) |
| XIAO build enlarged app partition | 2,646,542 / 3,145,728 B (84.1%) |
| Physical XIAO seeds 0-4 | PASS, 5/5 seeds, worst max_abs 1.0893e-03 |
| Physical XIAO FAST single-input forward | 2.462 s/forward |

The single-input forward 2.462 s is a fresh physical capture (the
firmware streams one `S*D` input frame per forward); a complete batch-64
total would be a derived projection and is not reported.

## Review findings

1. `EXACT` is not an all-fp32 or bit-for-bit mode. Both modes quantize
   per-head Q/K/V to int16 before attention. It passes the tolerance
   gate, but the description should say hybrid reference mode rather
   than exact.
2. `weights.bin`, `weights_q12.bin`, and `testdata/` are intentionally
   omitted from Git. A clean checkout must regenerate them
   (`python3 tools/export_case2.py --outdir . --seeds 25 --B 64 --S 128 --D 128 --H 16 --F 128 --L 4`)
   before firmware build. The exporter runs with
   the system python3 (torch 2.10.0) and reproduces the manifests
   deterministically.
3. The firmware streams one input frame per forward; a batch of 64
   inputs is 64 sequential forwards with resident weights. Batch time
   would be additive (B x single forward), but only the measured
   single-input forward is reported; there is no pipelining that turns
   batch parallelism into per-input latency reduction on one board.
4. At `H=16` head width is 8, so per-head QK/PV rows are tiny (128x8)
   and the attention loop is index/setup-bound rather than
   dot-product-bound. The 2.462 s forward is dominated by the full-D
   projections (identical to case-2 per input) plus per-head loop
   overhead.
5. This is the official benchmark's Transformer body, not yet a trained
   text generator: it accepts float embeddings and returns float hidden
   states, with no token/position embeddings or vocabulary head.

## Recommended direction for this case shape

Keep this large baseline as the H=16 correctness target. Head
parallelism at width 8 is fine-grained and communication-bound, so
whole-head sharding of 16 heads is unlikely to scale; the better
experiment is grouped-head shards (e.g. 4 workers x 4 heads, or 8
workers x 2 heads) with the per-head 128x8 Q15 context slices as the
message unit. Profile the per-head loop overhead first — at 16 heads it
may rival the dot-product work. The measured 2.462 s single-input is
the number any distributed claim must beat.
