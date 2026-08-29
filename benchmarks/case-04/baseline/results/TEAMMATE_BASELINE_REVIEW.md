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

I regenerated the omitted deterministic weights/test vectors, compiled
the C host implementation, reran the full 25-seed x 2-mode gate,
independently recomputed seed 0 through the vendored torch_ref, and
built the PlatformIO firmware for the batch shape. The concise capture
is
[`case-04_esp32_baseline_seed0_v1.log`](case-04_esp32_baseline_seed0_v1.log).

| Check | Independent result |
|---|---:|
| C FAST, 25 host seeds | 0 / 409,600 failed outputs |
| Worst C FAST absolute error | 0.0010 |
| C EXACT, 25 host seeds | 0 / 409,600 failed outputs |
| Worst C EXACT absolute error | 0.0001 |
| Vendored torch_ref recompute, seed 0 vs committed ref_0.bin | 0 / 16,384 mismatched (max_abs 0.0) |
| XIAO build RAM | 274,564 / 327,680 B (83.8%) |
| XIAO build enlarged app partition | 2,644,470 / 3,145,728 B (84.1%) |
| Physical XIAO seed 0 (reused from case-02, B=1) | PASS, 1.996 s |
| Case 4 complete forward (16 x 1.996 s) | 31.936 s |

The numerical result is confirmed on host. The device timing for this
case is not a fresh physical capture: case-04 belongs to the batch
group whose port policy is NONE (no reflash of the shared boards), so
the per-input 1.996 s is reused from the case-02 device run at the
identical B=1 geometry (25/25 device seeds verified at
1.996 s/forward). The complete-forward time is therefore B x 1.996 s.

## Review findings

1. `EXACT` is not an all-fp32 or bit-for-bit mode. Both modes quantize
   per-head Q/K/V to int16 before attention. It passes the tolerance
   gate, but the description should say hybrid reference mode rather
   than exact.
2. `weights.bin`, `weights_q12.bin`, and `testdata/` are intentionally
   omitted from Git. A clean checkout must regenerate them
   (`python3 tools/export_case2.py --outdir . --seeds 25 --B 16 --S 128 --D 128 --H 4 --F 128 --L 4`)
   before host validation or firmware build. The exporter runs with
   the system python3 (torch 2.10.0) and reproduces the manifests
   deterministically.
3. Host gate commands: from `optimisation/esp32-baseline`,
   `(cd tools && make host_test && ./host_test all --both)` prints 50
   per-seed lines and `done: 50 seed-runs, 0 failed ==> ALL PASS`.
4. The firmware streams one input frame per forward; a batch of 16
   inputs is 16 sequential forwards with resident weights. Batch time
   is additive (B x single forward), which is exactly what this
   baseline reports; there is no pipelining that could turn batch
   parallelism into per-input latency reduction on one board.
5. This is the official benchmark's Transformer body, not yet a trained
   text generator: it accepts float embeddings and returns float hidden
   states, with no token/position embeddings or vocabulary head.

## Recommended direction for this case shape

Keep this large baseline as the batch-16 correctness target. The
highest-value next step for a single board is to confirm the
assumed 1.996 s/input hold across the batch on the physical device
(one flash, seeds 0-4) and to tile the stream so weight fetches stay
resident in flash cache. For aggregate throughput, batch-parallel
multiboard dispatch of the 16 independent inputs is the better
experiment: communication must be included in the result rather than
inferred from per-input compute time.
