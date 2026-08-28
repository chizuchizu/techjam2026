# Pocket Attention Cluster

Pocket Attention Cluster is a TechJam experiment in running and eventually
distributing Transformer attention across inexpensive ESP32 microcontrollers.
The current milestone is a measured single-node implementation on a Seeed
Studio XIAO ESP32-C3; the multi-node stage begins when the additional boards are
available.

The project deliberately targets ESP32 hardware only: memory-efficient attention,
mixed-precision arithmetic, and communication-aware execution across a cluster
of small microcontrollers. The supplied GPU benchmark remains background
material, not an implementation target.

## What works now

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

- [`esp32_attention_benchmark/`](esp32_attention_benchmark/) — firmware and C++
  attention kernels, including the complete attention-layer benchmark.
- [`tools/validate_e2e.py`](tools/validate_e2e.py) — independent host reference
  and serial validator.
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

The first build used Arduino CLI 1.5.1, Arduino-ESP32 3.3.11, and esptool 5.3.1.

## Current limitations

- This now benchmarks a complete attention layer, not layer normalization, the
  feed-forward network, residual connections, or a complete Transformer block.
- Batch is 1 and the end-to-end fixture is `N=16, d_model=32, 4 heads`.
- The tested fixture is synthetic and small; the projection format must be
  recalibrated and validated on weights and activations from a trained model.
- Activation quantization is included in end-to-end timing. Offline model-weight
  quantization is excluded, as weights are stored in their inference format.
- Latency is measured; energy has not yet been instrumented.
- Only one ESP32 is currently available, so cluster speedup remains a design,
  not a measured claim.
- No world-first claim is justified; related MCU and distributed Transformer
  work already exists.
