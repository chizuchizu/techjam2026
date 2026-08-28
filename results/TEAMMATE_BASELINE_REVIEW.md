# Review of `techjam2026/esp32-baseline`

Reviewed commit: `34f8c8a9228ead88743823923e28d29eb5a50a0c` from
`chizuchizu/techjam2026`, on 2026-08-28.

## What is confirmed

The teammate implementation is a substantially larger benchmark-shaped
Transformer body: `S=128`, `D=128`, four heads, four blocks, FFN width 128, and
398,592 parameters. It includes LayerNorm, streaming causal attention,
residuals, GELU FFNs, and final LayerNorm. FAST mode uses Q15 activations and
Q12 weights for the six projection matrices per block.

I regenerated its omitted deterministic weights/test vectors, compiled the C
host implementation, reran the independent NumPy review, built the PlatformIO
firmware, flashed it to the physical XIAO ESP32-C3, and ran seed 0 end to end.
The concise physical-device capture is
[`teammate_esp32_baseline_seed0_v1.log`](teammate_esp32_baseline_seed0_v1.log).

| Check | Independent result |
|---|---:|
| C FAST, 25 host seeds | 0 / 409,600 failed outputs |
| Worst C FAST absolute error | 0.0009970 |
| Independent NumPy FAST | 0 / 409,600 failed outputs |
| XIAO build RAM | 267,804 / 327,680 B (81.7%) |
| XIAO build enlarged app partition | 2,621,594 / 3,145,728 B (83.3%) |
| Physical XIAO seed 0 | PASS, max abs 0.0008124 |
| Physical XIAO FAST forward | 42.152 s |

The numerical result is confirmed. The current speed claim is not: comments
and documentation say a 2–4 second target, while the physical board measures
42.152 seconds. The test driver itself also mentions approximately 42 seconds.

## Review findings

1. `EXACT` is not truly an all-fp32 or bit-for-bit mode. Both modes quantize
   per-head Q/K/V to int16 before attention. It passes the tolerance gate, but
   the description should say hybrid reference mode rather than exact.
2. Reproduction commands are inconsistent. `make -C tools` uses paths that do
   not resolve, and its printed `./tools/host_test` path does not match where
   the Makefile writes the binary. `make -f tools/Makefile host_test` followed
   by running `../host_test` from `tools/` works.
3. `weights.bin`, `weights_q12.bin`, and `testdata/` are intentionally omitted
   from Git, so a clean checkout must regenerate about 5 MB of artifacts before
   host validation or firmware build. The exporter works, but dependencies and
   a small committed golden capture should be pinned for easier reproduction.
4. FAST firmware embeds 1.59 MB of fp32 weights as well as 0.79 MB of Q12
   matrices. Most fp32 matrices are unused in FAST mode; packing only norms and
   biases would recover roughly 1.57 MB of flash.
5. This is the official benchmark's Transformer body, not yet a trained text
   generator: it accepts float embeddings and returns float hidden states, with
   no token/position embeddings or vocabulary head.
6. At review time, remote `main` and `esp32-baseline` both pointed at the same
   commit, so the baseline work is not isolated from main despite the branch
   name.

## Recommended combined direction

Keep this large baseline as the official-shape correctness target and keep the
small trained model as the end-to-end language-model demonstration. The highest
value next optimization is to replace the baseline attention's inner fp32
dequantized QK multiply loop with the passing int8/int32 QK dot product already
used in this repository. Then remove unused fp32 matrices from the FAST image,
remeasure, and only after that generalize the existing head-parallel protocol
to `S=128, D=128` for two-board tests.

For multiple boards, whole-layer pipeline parallelism can improve throughput
over a stream of inputs, but not single-input latency. Head parallelism is the
better latency experiment: keep each worker's head weights resident, broadcast
the normalized activation once, and return one 16 KB context slice per worker.
The measured LAN rate means communication must be included in the result rather
than inferred from compute time.
