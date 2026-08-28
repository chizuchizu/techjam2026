# TechJam 2026 — ESP32 Transformer

This repository now focuses on running and distributing Transformer workloads
on ESP32 microcontrollers. The previous NVIDIA H200 experiments were retired;
the GPU baseline is the official `torch_transformer_benchmark.py` at the repo root
(see `COMPETITION_RULES.MD`).

## Start here

| Area | Purpose | Best first file |
|---|---|---|
| [`esp32-baseline/`](esp32-baseline/) | Official-size four-layer numerical benchmark | [`README.md`](esp32-baseline/README.md) |
| [`esp32_cluster_transport/`](esp32_cluster_transport/) | Wi-Fi worker for multi-ESP32 attention | [`README.md`](esp32_cluster_transport/README.md) |
| [`tools/`](tools/) | Host coordinators, validation, training, and discovery | [`run_large_head_parallel.py`](tools/run_large_head_parallel.py) |
| [`esp32_attention_benchmark/`](esp32_attention_benchmark/) | Single-board attention kernels | [`README.md`](esp32_attention_benchmark/README.md) |
| [`esp32_tiny_transformer/`](esp32_tiny_transformer/) | Complete small trained character Transformer | [`README.md`](esp32_tiny_transformer/README.md) |
| [`results/`](results/) | Raw measurements and short reports | [`FOUR_C3_PARALLEL_RESULTS.md`](results/FOUR_C3_PARALLEL_RESULTS.md) |
| [`docs/`](docs/) | Design decisions, plans, and prior art | [`PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) |

The benchmark-sized baseline does not generate text; it validates Transformer
body calculations on random hidden states. The tiny model generates character
tokens, but is intentionally small. The cluster work currently distributes
independent attention heads, not the complete four-layer model.

## Quick setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
make check
```

Arduino sketches use the Espressif Arduino core. The benchmark-sized baseline
uses PlatformIO. PyTorch is optional and only needed to regenerate reference
artifacts or retrain the tiny model.

To configure a Wi-Fi worker, copy the ignored secrets template and edit the
local copy:

```bash
cp esp32_cluster_transport/secrets.example.h \
   esp32_cluster_transport/secrets.h
arduino-cli compile --fqbn esp32:esp32:XIAO_ESP32C3 \
   esp32_cluster_transport
```

Never commit `secrets.h`, Wi-Fi credentials, private IP addresses, or generated
build directories.

## Current verified results

- The complete tiny model matches an independent NumPy implementation and
  generates at 9.38 tokens/s on a physical XIAO ESP32-C3.
- The official-size baseline passes 25 host seeds and five device seeds.
- Four matched XIAO ESP32-C3 boards achieve 3.92–3.98x attention-head speedup,
  or 98.0–99.4% four-node efficiency including Wi-Fi communication.
- The earlier matched two-C3 experiment achieves about 2x speedup.
- A separate heterogeneous C3 plus dual-core ESP32 experiment demonstrates why
  unequal boards require measured assignment. All returned elements pass the
  accuracy gate in both experiments.

See [`TODO.md`](TODO.md) for the shared priorities and
[`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a pull request.
