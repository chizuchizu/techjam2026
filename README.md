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

## Official test-case coverage on ESP32

All official cases are causal and contain four Transformer layers except case
14, which contains two. A running time is shown only when the complete official
Transformer body has run and passed on physical ESP32 hardware; attention-only
experiments are not counted as implemented cases.

| Case | Batch | Sequence | Model dim | Heads | FFN dim | ESP32 status | Physical running time |
|---:|---:|---:|---:|---:|---:|---|---:|
| 1 | 64 | 128 | 128 | 4 | 128 | Not implemented | — |
| 2 | 1 | 128 | 128 | 4 | 128 | **Implemented and verified** | **42.09 s** on one C3 |
| 3 | 4 | 128 | 128 | 4 | 128 | Not implemented | — |
| 4 | 16 | 128 | 128 | 4 | 128 | Not implemented | — |
| 5 | 128 | 128 | 128 | 4 | 128 | Not implemented | — |
| 6 | 10,000 | 128 | 128 | 4 | 128 | Not implemented | — |
| 7 | 64 | 128 | 32 | 4 | 32 | Not implemented | — |
| 8 | 64 | 128 | 1,024 | 4 | 1,024 | Not implemented | — |
| 9 | 64 | 128 | 128 | 1 | 128 | Not implemented | — |
| 10 | 64 | 128 | 128 | 2 | 128 | Not implemented | — |
| 11 | 64 | 128 | 128 | 16 | 128 | Not implemented | — |
| 12 | 64 | 32 | 128 | 4 | 128 | Not implemented | — |
| 13 | 64 | 1,024 | 128 | 4 | 128 | Not implemented | — |
| 14 | 32 | 100,000 | 1,024 | 16 | 1,024 | Not implemented | — |

Case 2's full single-board measurement is documented in
[`esp32-baseline/README.md`](esp32-baseline/README.md). The four-C3 result of
0.766 s is only one causal four-head attention operation with the same
`S=128, H=4, d_head=32` shape. It excludes projections, residuals,
normalization, FFN, and the remaining layers, so it is not a case-2 end-to-end
running time.

See [`TODO.md`](TODO.md) for the shared priorities and
[`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a pull request.
