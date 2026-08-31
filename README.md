# NotGPU Attention

> A complete Transformer on an S$7 ESP32-C3—optimized from one no-FPU
> microcontroller to an eight-board wireless cluster.

**TinyCluster:** Karthik Gangula, Mingchen Yang, and Yuma Ochi

[Devpost report](docs/DEVPOST_PROJECT_DESCRIPTION.md) ·
[full benchmark table](benchmarks/README.md) ·
[competition specification](COMPETITION_RULES.MD)

## What we built

The competition asks teams to accelerate a Transformer while matching a PyTorch
reference. Instead of using a GPU, we ran the complete four-layer Transformer body on
a 160 MHz ESP32-C3 with no FPU, no PSRAM, and about 321 KB of usable SRAM.

The implementation includes causal multi-head attention, LayerNorm, residuals, GELU,
and the feed-forward network. Every reported result is checked against the official
numerical gate.

## Results

| Configuration | Result |
|---|---:|
| One C3 | `42.15 s -> 1.996 s` — **21.1x faster** |
| Two C3s, one complete input | `1.996 s -> 1.276 s` — **1.56x faster** |
| Eight Wi-Fi workers | **8.00x scaling** vs one identical tiled worker |
| Cases 1–5 eight-board sweep | **213/213 forwards passed** |

The eight Wi-Fi workers use a slower memory-saving schedule so that Wi-Fi and the
model fit together. Against the fastest untiled single board, the fair cluster gain
for fully utilized cases is approximately **3.78x**.

Device-compute timings exclude host transfer consistently. Physical measurements,
derived values, and projections are labelled separately in the
[benchmark evidence](benchmarks/README.md).

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
