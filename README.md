# TechJam 2026 — ESP32 Transformer

This repository explores a different ESP32 execution strategy for each of the
14 official Transformer benchmark cases. Every case owns its documentation,
baseline evidence, single-board optimisations, multiboard implementation, and
results under [`benchmarks/case-NN/`](benchmarks/).

## Case layout

[`benchmarks/README.md`](benchmarks/README.md) is the full case index. The
current implementation is concentrated in
[`case-02/`](benchmarks/case-02/):

```text
benchmarks/case-02/
├── README.md
├── baseline/
│   └── results/
├── optimisation/
│   └── esp32-baseline/
└── multiboard/
    ├── esp32_cluster_transport/
    ├── esp32-linkbench/
    ├── tools/
    └── results/
```

Shared competition material is stored once in
[`benchmarks/reference/`](benchmarks/reference/). Smaller experiments that are
not official cases are isolated in
[`benchmarks/experiments/`](benchmarks/experiments/).

## Latest verified results

Case 2 is `B=1, S=128, D=128, H=4, F=128, L=4`, causal. The first two rows
measure the same complete Transformer body. The multiboard rows have narrower
scopes and are not complete distributed forwards.

| Stage | Measured scope | Boards | Median wall time | Speedup | Accuracy |
|---|---|---:|---:|---:|---|
| Baseline | Complete case-2 Transformer body | 1 C3 | 42.15 s | 1.00x | Pass, 5/5 device seeds |
| Optimised single board | Complete case-2 Transformer body | 1 C3 | **5.27 s** | **8.0x** | Pass, 5/5 device seeds and 50/50 host checks |
| Multiboard | Layer-0 LayerNorm + Q/K/V + causal attention | 2 C3s | **4.850 s** | **2.00x** vs 9.693 s on one C3 | Pass, 5/5 seeds |
| Multiboard | Four independent causal attention heads | 4 C3s | **0.766 s** | **3.92x** vs 3.000 s average on one C3 | Pass, zero failed elements |

See the [`case-02 README`](benchmarks/case-02/README.md) for scope boundaries,
implementation links, and raw measurements.

## Official case status

| Case | Shape `(B,S,D,H,F,L)` | Status | Case notes |
|---:|---|---|---|
| [1](benchmarks/case-01/) | `(64,128,128,4,128,4)` | Not implemented | Batch-parallel candidate |
| [2](benchmarks/case-02/) | `(1,128,128,4,128,4)` | **Single-board verified** | Full body at 5.27 s; partial multiboard paths verified |
| [3](benchmarks/case-03/) | `(4,128,128,4,128,4)` | Not implemented | Small-batch scheduling |
| [4](benchmarks/case-04/) | `(16,128,128,4,128,4)` | Not implemented | Batch tiling and dispatch |
| [5](benchmarks/case-05/) | `(128,128,128,4,128,4)` | Not implemented | Throughput-oriented batch sharding |
| [6](benchmarks/case-06/) | `(10000,128,128,4,128,4)` | Not implemented | Streaming batch execution |
| [7](benchmarks/case-07/) | `(64,128,32,4,32,4)` | Not implemented | Narrow-kernel overhead and fusion |
| [8](benchmarks/case-08/) | `(64,128,1024,4,1024,4)` | Not implemented | Weight and feature sharding |
| [9](benchmarks/case-09/) | `(64,128,128,1,128,4)` | Not implemented | Sequence/model sharding; no head parallelism |
| [10](benchmarks/case-10/) | `(64,128,128,2,128,4)` | Not implemented | Two head shards plus batch parallelism |
| [11](benchmarks/case-11/) | `(64,128,128,16,128,4)` | Not implemented | Fine-grained head parallelism |
| [12](benchmarks/case-12/) | `(64,32,128,4,128,4)` | Not implemented | Short-sequence launch overhead |
| [13](benchmarks/case-13/) | `(64,1024,128,4,128,4)` | Not implemented | Online attention and KV sharding |
| [14](benchmarks/case-14/) | `(32,100000,1024,16,1024,2)` | Not implemented | Extreme sequence streaming |

The approach notes for unimplemented cases are design hypotheses, not measured
claims. Each case README records what must be validated before its status can
change.

## Quick setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
make check
```

Build the current case-2 single-board implementation:

The exporter also requires PyTorch in the selected Python environment.

```bash
cd benchmarks/case-02/optimisation/esp32-baseline
python3 tools/export_case2.py --outdir . --seeds 25
pio run -e esp32-baseline
```

Configure a local, Git-ignored Wi-Fi secrets file before compiling the case-2
cluster worker:

```bash
cp benchmarks/case-02/multiboard/esp32_cluster_transport/secrets.example.h \
   benchmarks/case-02/multiboard/esp32_cluster_transport/secrets.h
arduino-cli compile --fqbn esp32:esp32:XIAO_ESP32C3 \
  benchmarks/case-02/multiboard/esp32_cluster_transport
```

Never commit credentials, private addresses, generated model artifacts, or
build directories.

## Project documentation

- [`TODO.md`](TODO.md) — shared priorities.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — validation and result conventions.
- [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) — milestones and acceptance gates.
- [`docs/MULTI_ESP32_DESIGN.md`](docs/MULTI_ESP32_DESIGN.md) — cluster decomposition and protocol design.
- [`docs/PRIOR_ART.md`](docs/PRIOR_ART.md) — prior-art review and positioning.
