# Review of `techjam2026/case-04/optimisation/esp32-baseline`

Reviewed commit: `79f6cf666e6594b8d6cd5469055f2ad4479b5bc8` from `chizuchizu/techjam2026`
(the canonical parametric base this case copy is derived from), on
2026-08-29. The case-04 baseline directory is committed with this
review; see `git log` for the case-04 commit.

## What is confirmed

The teammate implementation is the benchmark-shaped Transformer body at
`B=16, S=128, D=128, H=4, F=128, L=4`, causal, with 398,592
parameters (the weight layout is independent of B). It includes
LayerNorm, streaming causal attention, residuals, GELU FFNs, and final
LayerNorm. FAST mode uses Q15 activations and Q12 weights for the six
projection matrices per block.

I regenerated the omitted deterministic weights/test vectors, built
the PlatformIO firmware for the batch shape, flashed it to the board,
and ran seeds 0–4 end to end. The concise capture is
[`case-04_esp32_baseline_seed0_v1.log`](case-04_esp32_baseline_seed0_v1.log).

| Check | Independent result |
|---|---:|
| XIAO build RAM | 274,564 / 327,680 B (83.8%) |
| XIAO build enlarged app partition | 2,644,470 / 3,145,728 B (84.1%) |
| Physical XIAO seeds 0-4 (fresh on-board capture, board A) | PASS 5/5, 1.990 s/forward |

The timed per-input forward is confirmed on the physical board: the
per-input 1.990 s is a fresh on-board capture on board A (5/5 device
seeds, see the seed0_v1 log). A
complete-batch total would be a derived projection and is
deliberately not reported.

## Review findings

1. `EXACT` is not an all-fp32 or bit-for-bit mode. Both modes quantize
   per-head Q/K/V to int16 before attention. It passes the tolerance
   gate, but the description should say hybrid reference mode rather
   than exact.
2. `weights.bin`, `weights_q12.bin`, and `testdata/` are intentionally
   omitted from Git. A clean checkout must regenerate them
   (`python3 tools/export_case2.py --outdir . --seeds 25 --B 16 --S 128 --D 128 --H 4 --F 128 --L 4`)
   before firmware build. The exporter runs with
   the system python3 (torch 2.10.0) and reproduces the manifests
   deterministically.
3. The firmware streams one input frame per forward; a batch of 16
   inputs is 16 sequential forwards with resident weights. Batch time
   would be additive (B x single forward), but this baseline reports
   only the measured per-input forward; there is no pipelining that
   could turn batch parallelism into per-input latency reduction on one
   board.
4. This is the official benchmark's Transformer body, not yet a trained
   text generator: it accepts float embeddings and returns float hidden
   states, with no token/position embeddings or vocabulary head.

## Recommended direction for this case shape

Keep this large baseline as the batch-16 correctness target. The
highest-value next step for a single board is to measure the full
16-input stream on the physical device (one flash of the batch loop,
seeds 0-4) and to tile the stream so weight fetches stay resident in
flash cache. For aggregate throughput, batch-parallel
multiboard dispatch of the 16 independent inputs is the better
experiment: communication must be included in the result rather than
inferred from per-input compute time.
