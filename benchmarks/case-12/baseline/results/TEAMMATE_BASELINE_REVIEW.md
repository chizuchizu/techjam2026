# Review of `techjam2026/esp32-baseline` — case 12

Reviewed commit: `f3b48ab` from `chizuchizu/techjam2026`, on 2026-08-29.

## What is confirmed

The implementation is the shared parametric benchmark-shaped Transformer body
configured to the case-12 shape: `S=32`, `D=128`, four heads (head dim 32),
four blocks, FFN width 128, `B=64` inputs streamed one forward per frame,
causal. It includes LayerNorm, streaming causal attention, residuals, GELU
FFNs, and final LayerNorm. FAST mode uses Q15 activations and Q12 weights for
the six projection matrices per block.

I regenerated the deterministic weights/test vectors (25 seeds from seed 1234),
compiled the C host implementation, ran both numeric modes against the vendored
torch reference, built the PlatformIO firmware, flashed it to the physical XIAO
ESP32-C3 (board A, `/dev/cu.usbmodem101`), and ran seeds 0–4 end to end. The
physical-device capture is
[`case-12_esp32_baseline_seed0_v1.log`](case-12_esp32_baseline_seed0_v1.log).

| Check | Independent result |
|---|---:|
| C FAST, 25 host seeds | 0 / 102,400 failed outputs |
| Worst C FAST absolute error | 1.0319e-03 |
| C EXACT, 25 host seeds | 0 / 102,400 failed outputs |
| Worst C EXACT absolute error | 7.5936e-05 |
| XIAO build RAM | 81,028 / 327,680 B (24.7%) |
| XIAO build enlarged app partition | 2,644,516 / 3,145,728 B (84.1%) |
| Physical XIAO seeds (board A) | 5/5 PASS |
| Physical XIAO FAST forward | 0.492 s (492,020 / 492,259 us) |

The numerical result is confirmed on the host for all 25 seeds in both modes
and on the board for seeds 0–4. The measured 0.492 s/forward matches the
pre-measured case-07 value on the same board (0.491 s at D=32/F=32), so the
short-sequence/projection overheads dominate at these shapes.

## Review findings

1. `EXACT` is not truly an all-fp32 or bit-for-bit mode. Both modes quantize
   per-head Q/K/V to int16 before attention. It passes the tolerance gate, but
   the description should say hybrid reference mode rather than exact.
2. `weights.bin`, `weights_q12.bin`, and `testdata/` are intentionally omitted
   from Git, so a clean checkout must regenerate the artifacts before host
   validation or firmware build (see commands below). The case-specific
   exporter is `python3 tools/export_case2.py --outdir . --seeds 25 --B 64
   --S 32 --D 128 --H 4 --F 128 --L 4`.
3. Host gate reproduction: `(cd tools && make host_test && ./host_test all
   --both)` prints 50 seed-runs and `done: 50 seed-runs, 0 failed ==> ALL PASS`.
4. Device reproduction on board A: `pio run` then
   `pio run -t upload --upload-port /dev/cu.usbmodem101` then
   `python3 tools/device_test.py /dev/cu.usbmodem101 --root . --seeds 0 1 2 3 4
   --reps 2`.
5. This is the official benchmark's Transformer body, not yet a trained text
   generator: it accepts float embeddings and returns float hidden states, with
   no token/position embeddings or vocabulary head.

## Recommended direction for case 12

Short sequences make attention cheap, so the GEMM-heavy projections and FFN
plus their dispatch/loop overhead dominate. Kernel fusion of the projection
GEMMs (K==128 asm path is active at this shape) and the normalization is the
first lever. Then compare single-board batch streaming (64 inputs × 0.492 s =
31.5 s) against head or batch parallelism. At ~0.49 s per forward, any
distributed latency claim must include measured communication time, not just
compute.
