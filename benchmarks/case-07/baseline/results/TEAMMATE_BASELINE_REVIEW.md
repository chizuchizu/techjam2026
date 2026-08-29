# Review of `techjam2026/esp32-baseline` — case 07

Reviewed commit: `bf8388d1fb` from `chizuchizu/techjam2026`, on 2026-08-29.

## What is confirmed

The implementation is the shared parametric benchmark-shaped Transformer body
configured to the case-07 shape: `S=128`, `D=32`, four heads (head dim 8),
four blocks, FFN width 32, `B=64` inputs streamed one forward per frame,
causal. It includes LayerNorm, streaming causal attention, residuals, GELU
FFNs, and final LayerNorm. FAST mode uses Q15 activations and Q12 weights for
the six projection matrices per block.

I regenerated the deterministic weights/test vectors (25 seeds from seed 1234),
compiled the C host implementation, ran both numeric modes against the vendored
torch reference, built the PlatformIO firmware, and confirmed the physical
board-A capture reused for this case (0.491 s/forward, 5/5 device seeds; no
reflash was permitted per the case spec).

| Check | Independent result |
|---|---:|
| C FAST, 25 host seeds | 0 / 102,400 failed outputs |
| Worst C FAST absolute error | 1.3875e-03 |
| C EXACT, 25 host seeds | 0 / 102,400 failed outputs |
| Worst C EXACT absolute error | 6.7055e-05 |
| XIAO build RAM | 83,332 / 327,680 B (25.4%) |
| XIAO build enlarged app partition | 418,610 / 3,145,728 B (13.3%) |
| Physical XIAO seeds (board A, pre-measured) | 5/5 PASS |
| Physical XIAO FAST forward | 0.491 s |

The numerical result is confirmed on the host for all 25 seeds in both modes.
The physical timing is the pre-measured board-A value (0.491 s/forward), which
also matches the freshly measured case-12 value on the same board (0.493 s).

## Review findings

1. `EXACT` is not truly an all-fp32 or bit-for-bit mode. Both modes quantize
   per-head Q/K/V to int16 before attention. It passes the tolerance gate, but
   the description should say hybrid reference mode rather than exact.
2. The physical-device capture in
   [`case-07_esp32_baseline_seed0_v1.log`](case-07_esp32_baseline_seed0_v1.log)
   is a genuine on-board capture on board A (`/dev/cu.usbmodem101`): the
   `0.491 s/forward` and per-seed device `max_abs` were measured in that run
   (5/5 device PASS). Reproduce with `tools/device_test.py
   /dev/cu.usbmodem101 --root . --seeds 0 1 2 3 4 --reps 2`. The per-input
   forward is reused across the 64 streamed inputs per the case spec (no
   reflash); no derived batch total is reported.
3. `weights.bin`, `weights_q12.bin`, and `testdata/` are intentionally omitted
   from Git, so a clean checkout must regenerate the artifacts before host
   validation or firmware build (see commands below). The case-specific
   runner is `python3 tools/export_case2.py --outdir . --seeds 25 --B 64
   --S 128 --D 32 --H 4 --F 32 --L 4`.
4. Host gate reproduction: `(cd tools && make host_test && ./host_test all
   --both)` prints 50 seed-runs and `done: 50 seed-runs, 0 failed ==> ALL PASS`.
5. This is the official benchmark's Transformer body, not yet a trained text
   generator: it accepts float embeddings and returns float hidden states, with
   no token/position embeddings or vocabulary head.

## Recommended direction for case 07

D and F of 32 make every projection and FFN matrix tiny, so dispatch, loop,
and serialization overhead dominate the GEMMs. Fuse the narrow projections and
normalization into fewer kernels first, then compare single-board batch
streaming the 64 inputs against head or batch parallelism.
With per-forward time under 0.5 s, include communication in any distributed
latency result rather than inferring it from compute time.
