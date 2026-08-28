# TechJam 2026 — ESP32 Transformer

This repository now focuses on running and distributing Transformer workloads
on ESP32 microcontrollers. The previous NVIDIA H200 experiments are preserved
under [`h200/`](h200/) for reference, but they are not an active workstream.

## Start here

| Area | Purpose | Best first file |
|---|---|---|
| [`esp32-baseline/`](esp32-baseline/) | Official-size four-layer numerical benchmark | [`README.md`](esp32-baseline/README.md) |
| [`esp32_cluster_transport/`](esp32_cluster_transport/) | Wi-Fi worker for multi-ESP32 attention | [`README.md`](esp32_cluster_transport/README.md) |
| [`tools/`](tools/) | Host coordinators, validation, training, and discovery | [`run_large_head_parallel.py`](tools/run_large_head_parallel.py) |
| [`esp32_attention_benchmark/`](esp32_attention_benchmark/) | Single-board attention kernels | [`README.md`](esp32_attention_benchmark/README.md) |
| [`esp32_tiny_transformer/`](esp32_tiny_transformer/) | Complete small trained character Transformer | [`README.md`](esp32_tiny_transformer/README.md) |
| [`results/`](results/) | Raw measurements and short reports | [`LARGE_HEAD_PARALLEL_RESULTS.md`](results/LARGE_HEAD_PARALLEL_RESULTS.md) |
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
- Two physical ESP32s execute attention heads concurrently. Equal assignment
  gives 1.94–1.95x over the XIAO-only baseline; a measured 1+3 split gives
  3.93–3.96x. All returned elements pass the accuracy gate.

See [`TODO.md`](TODO.md) for the shared priorities and
[`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a pull request.
