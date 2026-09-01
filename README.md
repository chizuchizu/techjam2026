# NotGPU Attention

> A complete Transformer on an S$7 ESP32-C3—optimized from one no-FPU
> microcontroller to an eight-board wireless cluster.

**TinyCluster:** Karthik Gangula, Mingchen Yang, and Yuma Ochi

[Devpost report](docs/DEVPOST_PROJECT_DESCRIPTION.md) ·
[full benchmark table](benchmarks/README.md) ·
[competition specification](COMPETITION_RULES.MD)

![Eight Seeed XIAO ESP32-C3 boards used for the wireless cluster](docs/assets/esp32-eight-board-cluster.jpg)

*The physical eight-board ESP32-C3 cluster used for our measurements.*

## What we built

The competition asks teams to accelerate a Transformer while matching a PyTorch
reference. Instead of using a GPU, we ran the complete four-layer Transformer body on
a 160 MHz ESP32-C3 with no FPU, no PSRAM, and about 321 KB of usable SRAM.

The implementation includes causal multi-head attention, LayerNorm, residuals, GELU,
and the feed-forward network. Every reported result is checked against the official
numerical gate.

## Results

| Case | Baseline, 1 C3 | Optimized, 1 C3 | Cluster result | Total improvement | Gate |
|---:|---:|---:|---:|---:|---:|
| [01](benchmarks/case-01/) | 2,697.6 s `*` | 127.36 s | **33.713 s** (8 C3s) | **80.0x** `*` | 64/64 pass |
| [02](benchmarks/case-02/) | 42.15 s | 1.990 s | **1.276 s** (2 C3s) | **33.0x** | pass |
| [03](benchmarks/case-03/) | 168.6 s `*` | 7.96 s | **4.218 s** (4/8 active) | **40.0x** `*` | 4/4 pass |
| [04](benchmarks/case-04/) | 674.4 s `*` | 31.84 s | **8.438 s** (8 C3s) | **79.9x** `*` | 16/16 pass |
| [05](benchmarks/case-05/) | 5,395.2 s `*` | 254.72 s | **67.451 s** (8 C3s) | **80.0x** `*` | 128/128 pass |
| [07](benchmarks/case-07/) | 295.05 s `†` | 30.427 s | **3.963 s** (8 C3s) | **74.5x** `†` | 64/64 pass |
| [09](benchmarks/case-09/) | 2,697.6 s `†` | 138.027 s | **28.508 s** (8 C3s) | **94.6x** `†` | 64/64 pass |
| [10](benchmarks/case-10/) | 2,697.6 s `†` | 138.536 s | **29.793 s** (8 C3s) | **90.5x** `†` | 64/64 pass |
| [11](benchmarks/case-11/) | 2,697.6 s `†` | 138.610 s | **51.604 s** (8 C3s) | **52.3x** `†` | 64/64 pass |
| [12](benchmarks/case-12/) | 547.95 s `†` | 33.879 s | **4.282 s** (8 C3s) | **128.0x** `†` | 64/64 pass |

`*` Direct batch projection from the measured Case 2 baseline. `†` FLOP-normalized
baseline estimate. All optimized and cluster times are physical measurements of the
complete four-layer body; host transfer is excluded consistently.

![Case 2 single-board optimization from 42.15 seconds to 1.996 seconds](docs/assets/case-2-single-board-optimisation.png)

*Case 2 improved cumulatively from 42.15 s to 1.996 s on one board.*

The eight Wi-Fi workers use a slower memory-saving schedule so that Wi-Fi and the
model fit together. Against the fastest untiled single board, the fair cluster gain
for fully utilized cases is approximately **3.78x**.

Device-compute timings exclude host transfer consistently. Physical measurements,
derived values, and projections are labelled separately in the
[benchmark evidence](benchmarks/README.md).

![Single-board optimization and eight-board scaling across official benchmark cases](docs/assets/optimisation-and-eight-board-scaling.png)

*Cross-case results. `*` marks direct batch projections and `†` marks
FLOP-normalized baseline estimates; optimized and cluster times are physical runs.*

## Main optimizations

- **Fixed-point attention:** Q15 activations, Q12 weights, integer QK/PV, and a
  lookup-table softmax replace expensive software floating point.
- **Register-aware GEMM:** a `4 x 2` output tile avoids register spills on RV32IMC.
- **Fusion:** fixed-point residual updates remove conversions and complete memory
  passes.
- **RISC-V assembly:** independent accumulators hide multiplication latency.
- **SRAM-aware tiling:** 16-row, head-sequential execution lets the model and Wi-Fi
  stack coexist without PSRAM.

## Parallel execution

- **Batch data parallelism:** each board holds a full model and processes independent
  inputs. No tensors move between workers during a forward.
- **Token-row parallelism:** case 2 has only one input, so two boards process
  alternating token rows and exchange only attention K/V data. Communication is
  overlapped with computation.

## Correctness

Each finite output element must satisfy:

```text
absolute error <= 0.002  OR  relative error <= 0.02
```

The supplied benchmark uses deterministic random inputs and weights. Passing means
numerical equivalence to the reference, not trained-task accuracy.

## Tools

- C/C++, RISC-V assembly, Arduino-ESP32, FreeRTOS, lwIP, TCP/UDP, and ESP-NOW
- PlatformIO, Arduino CLI, GNU Make, GCC, and RISC-V binutils
- Python, PyTorch, NumPy, pySerial, and Matplotlib
- Git and GitHub
- OpenAI Codex and Anthropic Claude Code as development and review assistants

No cloud inference API or external dataset is used at runtime. AI-assisted changes
were retained only after compilation, validation, and physical measurement where
claimed.

## Quick start

Run the host-side checks:

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

## Current limits

- Complete official-case coverage is unfinished for cases 6, 8, 13, and 14.
- The benchmark uses random weights rather than a trained application model.
- Energy per forward has not yet been measured externally.
- The cluster is a reproducible prototype, not a production-hardened network.

## More detail

- [Devpost project description](docs/DEVPOST_PROJECT_DESCRIPTION.md)
- [Authoritative results](benchmarks/README.md)
- [Engineering report](docs/report/index.html)
- [Wi-Fi memory design](docs/WIFI_ON_A_COMPUTE_NODE.md)
- [Profiler](tinyprof/README.md)
- [Prior art and novelty boundary](docs/PRIOR_ART.md)
