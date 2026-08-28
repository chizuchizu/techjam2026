# Pocket Attention Cluster

Pocket Attention Cluster is a TechJam experiment in running and distributing
Transformers across inexpensive ESP32 microcontrollers. The current milestone
includes both measured attention kernels and a trained character Transformer
running end to end on a Seeed Studio XIAO ESP32-C3.

The project deliberately targets ESP32 hardware only: memory-efficient attention,
mixed-precision arithmetic, and communication-aware execution across a cluster
of small microcontrollers. The supplied GPU benchmark remains background
material, not an implementation target.

## What works now

- A complete trained causal language model: token and position embeddings, two
  pre-norm Transformer blocks, four-head attention, residuals, feed-forward
  networks, final RMSNorm, and a tied 24-token language-model head.
- Greedy generation on the physical board at a median 106.614 ms per token;
  all 24 validation logits and all 48 generated tokens match the independent
  NumPy deployment reference.
- Float32 materialized scaled dot-product attention.
- Exact block-online softmax that never stores the `N x N` attention matrix.
- Mixed precision with int8 Q/K and int16 V.
- Causal and non-causal cases.
- A complete 4-head attention layer with Q/K/V and output projections, padding
  masks, causal masks, and timed activation quantization/dequantization.
- Projection optimization using int16 activations, per-output-channel int8
  weights, and native int32 dot products.
- Independent host validation that reconstructs the fixture without calling the
  ESP32 implementation.
- A binary UDP/TCP cluster transport and physical LAN benchmark from WSL to the
  ESP32.
- A working binary head-task protocol and coordinator that reassembles four
  remotely computed heads and validates the complete output.
- Exact distributed key/value sharding using mergeable online-softmax
  statistics, also validated over the real LAN.
- Per-element validation using the hackathon rule: absolute error <= 0.002 or
  relative error <= 0.02.
- Reproducible latency, workspace, working-set, and accuracy output over USB.

On the largest tested shape (`N=128`, head dimension `32`):

| Kernel | Median | Speedup | Working set | Accuracy |
|---|---:|---:|---:|---:|
| Float materialized reference | 1.866 s | 1.00x | 131,072 B | reference |
| Int8 Q/K + int16 V, materialized | 1.336 s | 1.40x | 98,304 B | pass |
| Int8 Q/K + int16 V, tiled online | 1.483 s | 1.26x | 32,928 B | pass |

The tiled mixed-precision kernel reduces modeled working memory by 74.9% and
attention workspace from 65,536 bytes to 160 bytes while still outperforming
the float reference. Full results are in
[`results/esp32c3_attention_v3.csv`](results/esp32c3_attention_v3.csv).

The end-to-end `N=16, d_model=32, 4 heads` benchmark also passes every output
for causal and non-causal padding-mask cases. The first float-projection version
was only 1.015x faster non-causally and 0.999x for causal attention. Replacing
the four float projection matrices with per-output-channel int8 weights and
int16 activations improves that to 3.05x and 3.87x, while reducing their modeled
working set from 31,328 B to 21,600 B. See
[`results/esp32c3_end_to_end_v2.csv`](results/esp32c3_end_to_end_v2.csv).

## Repository map

- [`esp32_tiny_transformer/`](esp32_tiny_transformer/) — trained complete
  Transformer inference and serial benchmark firmware.
- [`tools/train_tiny_transformer.py`](tools/train_tiny_transformer.py) —
  deterministic PyTorch training and int8/int16 deployment export.
- [`tools/tiny_transformer_reference.py`](tools/tiny_transformer_reference.py)
  — independent NumPy reference and physical-board validator.
- [`results/TINY_TRANSFORMER_RESULTS.md`](results/TINY_TRANSFORMER_RESULTS.md) —
  complete-model accuracy, latency, and memory results.
- [`esp32_attention_benchmark/`](esp32_attention_benchmark/) — firmware and C++
  attention kernels, including the complete attention-layer benchmark.
- [`tools/validate_e2e.py`](tools/validate_e2e.py) — independent host reference
  and serial validator.
- [`esp32_cluster_transport/`](esp32_cluster_transport/) — LAN worker discovery
  and validated binary echo firmware.
- [`results/TRANSPORT_RESULTS.md`](results/TRANSPORT_RESULTS.md) — measured UDP
  and TCP latency, throughput, and loss.
- [`results/HEAD_PARALLEL_RESULTS.md`](results/HEAD_PARALLEL_RESULTS.md) — real
  one-worker head-task timings and accuracy.
- [`results/KV_SHARDED_RESULTS.md`](results/KV_SHARDED_RESULTS.md) — exact
  distributed-softmax validation and its communication cost.
- [`results/`](results/) — raw board measurements and interpretation.
- [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) — narrow milestones and gates.
- [`docs/PROBLEMS_AND_SOLUTIONS.md`](docs/PROBLEMS_AND_SOLUTIONS.md) — pitch-ready
  engineering problems and responses.
- [`docs/MULTI_ESP32_DESIGN.md`](docs/MULTI_ESP32_DESIGN.md) — four/ten-node
  execution design.
- [`docs/PRIOR_ART.md`](docs/PRIOR_ART.md) — novelty check and positioning.
- [`statement.md`](statement.md) — supplied hackathon statement.

## Reproduce on WSL

Requirements: Arduino CLI, the Espressif `esp32:esp32` core, `usbipd-win`, and a
XIAO ESP32-C3 attached to WSL as `/dev/ttyACM0`.

```bash
arduino-cli compile --fqbn esp32:esp32:XIAO_ESP32C3 \
  esp32_attention_benchmark
arduino-cli upload --fqbn esp32:esp32:XIAO_ESP32C3 \
  --port /dev/ttyACM0 esp32_attention_benchmark
arduino-cli monitor --port /dev/ttyACM0 --config baudrate=115200
```

Send `r` over serial to run the primitive suite, or `e` for the complete
attention-layer suite. Validate the latter directly from WSL with:

```bash
python3 tools/validate_e2e.py --port /dev/ttyACM0 \
  --capture results/esp32c3_end_to_end_v2.log
```

Build, flash, and independently validate the trained Transformer with:

```bash
arduino-cli compile --fqbn esp32:esp32:XIAO_ESP32C3 esp32_tiny_transformer
arduino-cli upload --fqbn esp32:esp32:XIAO_ESP32C3 \
  --port /dev/ttyACM0 esp32_tiny_transformer
.venv/bin/python tools/tiny_transformer_reference.py --port /dev/ttyACM0 \
  --capture results/esp32c3_tiny_transformer_v1.log
```

The first build used Arduino CLI 1.5.1, Arduino-ESP32 3.3.11, and esptool 5.3.1.

## Current limitations

- The complete model is deliberately tiny (`context=16`, `d_model=32`, two
  blocks, 17,824 trained parameters). Its 100% corpus-window accuracy measures
  memorization of a 126-character training corpus, not generalization.
- The larger attention-layer fixture remains synthetic. The quantized formats
  now pass a trained small model, but still need validation on the official
  benchmark-sized Transformer.
- Activation quantization is included in end-to-end timing. Offline model-weight
  quantization is excluded, as weights are stored in their inference format.
- Latency is measured; energy has not yet been instrumented.
- Only one ESP32 is currently visible to WSL, so multi-board cluster speedup is
  not yet a measured claim.
- LAN transport is measured with one worker at strong signal; contention and
  straggler behavior require the four-board setup.
- No world-first claim is justified; related MCU and distributed Transformer
  work already exists.
