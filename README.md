# NotGPU Attention

> A complete Transformer on an S$7 microcontroller—optimized from one no-FPU
> ESP32-C3 to an eight-board wireless cluster.

**TinyCluster:** Karthik Gangula, Mingchen Yang, and Yuma Ochi

[Devpost technical report](docs/DEVPOST_PROJECT_DESCRIPTION.md) ·
[benchmark evidence](benchmarks/README.md) ·
[competition specification](COMPETITION_RULES.MD)

The competition asks teams to accelerate a Transformer while preserving the output
of the supplied PyTorch reference. Most solutions start with a GPU. We asked a more
extreme question: **what if there is no GPU at all?**

Our target is the Seeed XIAO ESP32-C3: a 160 MHz single-core RISC-V
microcontroller with no floating-point unit, no PSRAM, about 321 KB of usable
application SRAM, and 4 MB of flash. We kept the complete four-layer Transformer
body—including causal multi-head attention, LayerNorm, residuals, GELU, and the
feed-forward network—and redesigned how it computes, stores, and distributes data.

## Headline results

| Result | Improvement | Validation |
|---|---:|---|
| One C3: `42.15 s -> 1.996 s` | **21.1x** | Complete case-2 forward; physical device |
| Two C3s: `1.996 s -> 1.276 s` | **1.56x**, or **33.0x** vs original | Complete token-row forward; gate passes |
| Eight Wi-Fi workers | **8.00x** vs one identical tiled worker | Physical batch-data-parallel run |
| Cases 1–5 eight-board sweep | **213/213** forwards pass | No missing inputs or failing elements |

The eight-board workers use a memory-saving tiled schedule so that the model and
Wi-Fi stack fit together. Against the fastest untiled single board, the fair cluster
gain for fully utilized cases 1, 4, and 5 is approximately **3.78x**. We keep that
comparison separate from the 8.00x node-scaling result.

## How it works

1. **Fixed-point attention:** Q15 activations, Q12 weights, integer QK/PV, and a
   lookup-table softmax remove software floating point from the hottest loops.
2. **Register-aware GEMM:** a `4 x 2` output tile fits the real RISC-V register
   budget and avoids stack spills caused by a larger `8 x 2` tile.
3. **Fusion and assembly:** fixed-point residual updates remove full memory passes;
   hand-scheduled independent accumulators hide multiply latency.
4. **SRAM-aware execution:** 16-row, head-sequential tiling lets the Transformer and
   Wi-Fi/lwIP coexist without PSRAM.
5. **Shape-aware parallelism:** independent batch inputs use data parallelism;
   batch-one case 2 uses an alternating token-row split with overlapped K/V exchange.
6. **Evidence-first optimization:** every accepted change is checked against the
   official numerical gate and stored with reproducible measurement artifacts.

## Benchmark coverage

The complete measurement table, shape definitions, raw-artifact links, and
measured-versus-projected notes live in [`benchmarks/README.md`](benchmarks/README.md).
This README keeps only the physical-device summary needed by a reviewer.

| Scope | Physical result | Correctness |
|---|---:|---|
| Cases 1–5, eight-board Wi-Fi sweep | 213 complete forwards | **213/213 PASS** |
| Case 7, eight boards | 3.963 s for the full B=64 case | **64/64 PASS** |
| Case 9, eight boards | 28.508 s for the full B=64 case | **64/64 PASS** |
| Case 10, eight boards | 29.793 s for the full B=64 case | **64/64 PASS** |
| Case 11, eight boards | 51.604 s for the full B=64 case | **64/64 PASS** |
| Case 12, eight boards | 4.282 s for the full B=64 case | **64/64 PASS** |

All reported compute times cover the complete four-layer body and exclude host
input/output consistently. Wi-Fi-inclusive wall times are recorded separately.
Only case 2 has a physically measured pre-optimization baseline. Cross-shape
baseline estimates remain marked as projections and are not presented as device
measurements.

The official gate is applied per output element:

```text
absolute error <= 0.002  OR  relative error <= 0.02
```

The supplied benchmark uses deterministic random inputs and weights. Passing this
gate demonstrates numerical equivalence to the PyTorch reference, not trained-task
accuracy.

## Parallelization

### Batch data parallelism

Each worker stores a complete model and receives independent inputs from the host over
persistent TCP. Workers exchange no tensors during a forward, so useful parallelism
is `min(batch size, board count)`. When at least eight inputs are available, the eight
workers maintain essentially linear scaling against one identical tiled worker.

### Token-row parallelism

Official case 2 has batch size one, so ordinary data parallelism cannot use multiple
boards. We divide alternating sequence rows between two C3s. This balances the uneven
causal-attention workload. Most of the Transformer runs locally; the boards exchange
only attention K/V data, and a dedicated task overlaps that communication with
computation. The complete forward improves from 1.990 to 1.276 seconds.

Parity-interleaved attention is related to published Striped Attention. We do not
claim invention of the general algorithm. Our contribution is its fixed-point,
memory-bounded, communication-overlapped realization as a complete Transformer on
two no-FPU microcontrollers.

## Reproducible engineering

[`tinyprof/`](tinyprof/) is our operator-level ESP32-C3 profiler. It captures cycle
timing, call counts, instrumentation overhead, heap and stack watermarks, ELF-derived
static memory, embedded weight sizes, and traffic derived from measured calls. It
produces machine-readable artifacts and a self-contained comparison report.

[`esp32-linkbench/`](esp32-linkbench/) measures the communication layer independently
from model compute, including TCP, UDP, and ESP-NOW experiments. Keeping compute and
transport evidence separate prevents radio results from being mistaken for complete
Transformer performance.

Every official case keeps its implementation, validation evidence, and raw results
under [`benchmarks/case-NN/`](benchmarks/):

```text
benchmarks/case-02/
├── README.md
├── baseline/
│   └── results/
├── optimisation/
│   └── esp32-baseline/
└── multiboard/
    ├── esp32_cluster_transport/
    ├── tools/
    └── results/
```

Smaller experiments that are not complete official cases remain isolated under
[`benchmarks/experiments/`](benchmarks/experiments/).

## Development stack

- **Embedded:** C/C++, RISC-V assembly, Arduino-ESP32, ESP-IDF components, FreeRTOS,
  lwIP, TCP, UDP, and ESP-NOW.
- **Build and analysis:** PlatformIO, Arduino CLI, GNU Make, GCC, RISC-V GCC/binutils,
  ELF files, and linker maps.
- **Reference and validation:** Python 3, PyTorch, NumPy, and pySerial.
- **Visualization:** Matplotlib-generated scientific charts and original project
  illustrations.
- **Collaboration:** Git and GitHub.

### AI-assisted development

- **OpenAI Codex** supported repository analysis, implementation candidates,
  debugging, experiment design, review, and technical documentation.
- **Anthropic Claude Code** supported alternative design review and refinement of
  technical explanations and presentation material.

Neither tool is a runtime dependency. The ESP32 firmware calls no external AI or
cloud inference API. AI-assisted changes were retained only after compilation,
validation against the official reference, and physical-device measurement where a
hardware result is claimed.

## Data and assets

No external dataset is used. The organizers supplied
[`torch_transformer_benchmark.py`](torch_transformer_benchmark.py), which generates
deterministic random weights and inputs. We export those tensors into binary test
fixtures and quantized weight blobs for the firmware.

The repository also contains project-generated serial logs, JSON measurements,
linker evidence, Matplotlib figures, and photographs of the physical boards. Prior
work and third-party projects are cited in [`docs/PRIOR_ART.md`](docs/PRIOR_ART.md);
their results are not represented as our measurements.

## Limitations

- This benchmark demonstrates numerical equivalence on seeded random weights, not the
  accuracy of a trained application model.
- Complete official-case coverage remains unfinished for cases 6, 8, 13, and 14.
  Bounded streaming and ring-attention experiments are documented separately and are
  not presented as complete official results.
- The Wi-Fi-capable tiled worker is slower than the fastest radio-free worker; node
  scaling and fair speedup versus the best single board are reported separately.
- Device-compute timing excludes host transfer by benchmark convention. End-to-end
  transport results are reported separately.
- Energy per forward has not yet been measured with an external power instrument.
- This is a reproducible benchmark prototype, not a production-hardened network.

## Quick start

Create a Python environment and run all host-side checks:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
make check
```

Build the optimized case-2 firmware:

```bash
cd benchmarks/case-02/optimisation/esp32-baseline
python3 tools/export_case2.py --outdir . --seeds 25
pio run -e esp32-baseline
```

Build the complete two-board case-2 worker:

```bash
cd benchmarks/case-02/multiboard/esp32-cluster-full
pio run
```

The exporter requires PyTorch. Local Wi-Fi credentials belong in the provided
Git-ignored `secrets.h` workflow and must never be committed.

## Documentation

- [`docs/DEVPOST_PROJECT_DESCRIPTION.md`](docs/DEVPOST_PROJECT_DESCRIPTION.md) —
  copy-ready Devpost description and full technical report.
- [`benchmarks/README.md`](benchmarks/README.md) — authoritative case table and
  measured-versus-projected definitions.
- [`docs/report/index.html`](docs/report/index.html) — self-contained engineering
  report with figures.
- [`docs/WIFI_ON_A_COMPUTE_NODE.md`](docs/WIFI_ON_A_COMPUTE_NODE.md) — Wi-Fi SRAM
  collision, tiled solution, and measured trade-off.
- [`docs/MULTI_ESP32_DESIGN.md`](docs/MULTI_ESP32_DESIGN.md) — distributed execution
  architecture.
- [`tinyprof/README.md`](tinyprof/README.md) — profiler design and reproduction.
- [`docs/PRIOR_ART.md`](docs/PRIOR_ART.md) — novelty boundary and related work.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — validation and result-reporting rules.

## Submission assets

This repository contains the implementation and measurement evidence for the
TechJam 2026 submission. Add the public demo-video link to the Devpost entry after the
final upload; no video URL is fabricated here.
