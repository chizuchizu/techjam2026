# Narrow project plan

## Project question

How far can stable scaled dot-product attention be pushed on a no-PSRAM,
single-core ESP32-C3, and when do four or ten such devices beat one after real
wireless communication overhead is included?

## Success criteria

Every optimization must satisfy all of these before it is called successful:

1. Every output element passes `abs_error <= 0.002 OR rel_error <= 0.02` against
   the float reference.
2. Median latency is measured on the board after a warm-up and excludes input
   generation/allocation.
3. Workspace and total kernel working set are reported separately.
4. Causal and non-causal attention are both tested.
5. Failed and out-of-memory shapes remain in the results rather than being
   silently removed.

## Milestones

### M0 — Reproducible float baseline (complete)

- Materialize `QK^T`, stable row softmax, and multiply by V.
- Test `N=8..128`, head dimensions `8..64`, and a causal case.
- Capture CSV output from the physical ESP32-C3.

### M1 — Single-node memory and arithmetic optimization (complete)

- Implement exact block-online softmax with `O(tile + d)` workspace.
- Test a fast exponential approximation; retain it only if it wins.
- Quantize Q/K/V, diagnose accuracy failures, and choose a passing mixed format.
- Selected format: int8 Q/K and int16 V.

### M2 — End-to-end attention layer (complete)

- Added Q/K/V linear projections and output projection.
- Added deterministic padding masks, causal masking, and four attention heads.
- The host validator independently reconstructs all inputs and weights and
  checks all 512 board outputs without using the firmware kernel code.
- End-to-end latency includes projection weight reads, activation quantization,
  tiled attention, fused dequantization, and output projection.
- Both validation cases pass. Mixed precision is 1.015x faster non-causally and
  0.999x for causal attention, showing that projection work dominated.
- Follow-up complete: int16 activations and per-output-channel int8 projection
  weights raise end-to-end speedup to 3.05x non-causal and 3.87x causal, with
  every output still passing.

### M2.5 — Complete trained Transformer (complete)

- Trained a deterministic two-block causal character model with token and
  position embeddings, RMSNorm, residual paths, ReLU FFNs, and a tied LM head.
- Exported per-output int8 linear weights; linear activations use dynamic int16
  and attention uses the accuracy-qualified int8 Q/K plus int16 V format.
- The independent NumPy deployment model passes all 126 corpus windows.
- Physical-board validation passes all 24 logits and 48 generated tokens. The
  median complete forward is 106.614 ms with 24,800 B of model data and a
  22,688 B runtime working set.

### M3 — Four-node cluster (head-parallel physical run complete)

- TCP and UDP payload throughput and round-trip latency are measured against one
  physical worker; concurrent TCP is measured on both matched and heterogeneous
  two-board pairs, and ESP-NOW remains.
- Head-parallel execution is implemented: the coordinator sends four 534-byte
  tasks, reconstructs the returned heads, applies the output projection, and
  passes independent validation. It dispatches concurrently when given multiple
  worker IPs; both one-worker and two-worker paths are physically measured.
- Exact key/value sharding is implemented and passes: the worker returns local
  `(max, denominator, numerator)` statistics and the coordinator merges four
  shards per head. The one-worker result moves 3.43x more payload than head
  parallelism, supporting head-first execution for the current shape.
- One, two, and four matched C3 nodes have now been compared at identical
  shapes and accuracy. The four-node round-robin run reaches 3.98x non-causal
  and 3.92x causal speedup, including Wi-Fi communication.
- The head worker and coordinator now support the official `N=128,
  d_head=32` shape over persistent TCP. All 16,384 output elements per mode
  pass on every physical worker. Concurrent round-robin and profiled scheduling
  pass on both the heterogeneous pair and two matched C3 boards.
- Workers report chip model, cores, clock, heap, shape limits, and transport
  capabilities so heterogeneous ESP32 models can be scheduled by measurement.
- Measured per-head compute differs by roughly 14x between the C3 and the
  dual-core ESP32. The coordinator now supports round-robin, calibrated-all,
  and unconstrained calibrated scheduling; the last policy may idle a node when
  using it would increase latency.
- The matched pair achieves about 2x speedup. Four matched boards achieve
  3.92–3.98x, with every measured response passing the accuracy gate.
- The official-weight path now starts from case-2 `X`: every worker performs
  layer-0 LayerNorm, its assigned Q/K/V projection rows, and one causal head.
  Five physical seeds pass; one C3 takes 9.693 s and two take 4.850 s. Output
  projection, residuals, second LayerNorm, and FFN remain to complete a layer.

### M4 — Ten-node study

- Add node discovery, sequence numbers, timeouts, and straggler reporting.
- Compare static head assignment with profiled load balancing.
- Report the compute/communication crossover; do not assume more nodes are
  faster.

## Immediate experiment backlog

1. Port the integer Q/K dot-product and compact weight storage to the official
   `S=128, D=128, L=4` teammate baseline, then remeasure on the C3.
2. Finish the official-size distributed layer by adding output projection,
   residuals, second LayerNorm, and FFN to the verified Norm/Q/K/V/head path.
3. Test int8 Q/K + int16 V on independent adversarial inputs with wider score
   ranges.
4. Sweep online tile sizes 4, 8, 16, and 32.
5. Replace remaining float softmax arithmetic with a validated fixed-point/LUT
   version.
6. Compile the stable kernel under ESP-IDF with `-O3` and compare assembly and
   timing against Arduino's build.
7. Add energy measurement with an external current monitor.
