## Inspiration 💡

What if **every tiny IoT device could run AI locally**—without a GPU, cloud
connection, or expensive hardware?

We challenged ourselves to run a complete Transformer on a **S$7 ESP32-C3
microcontroller**—a chip designed for small connected sensors, not matrix-heavy AI.

We genuinely love working **close to the hardware**: counting cycles, reading
assembly, studying linker maps, and discovering how much useful computation can be
squeezed out of a tiny chip. For us, those constraints make the engineering more
exciting—not less.

If a Transformer could run here, it could open new possibilities for **private,
offline, and low-cost edge AI**.

## What it does 🤖

**NotGPUAttention runs a complete four-layer pre-LayerNorm Transformer body on an
ESP32-C3.**

It includes attention, LayerNorm, GELU, residual connections, and feed-forward
layers. It runs locally on **one C3**, splits one input across **two C3s**, or uses
**eight wireless workers** for larger batches.

Every result is compared with the official PyTorch output:

$$
\text{absolute error} \leq 0.002
\quad \text{or} \quad
\text{relative error} \leq 2\%
$$

So we are not only making it faster—we are making sure it is still **correct**.

## Hardware overview 🔬

| Hardware feature | Our XIAO ESP32-C3 | Why it matters for a Transformer |
|---|---|---|
| **CPU** | Single-core, 32-bit RISC-V RV32IMC; four-stage pipeline | Only one instruction stream—no second core for compute |
| **Clock** | Up to **160 MHz** | A tight cycle budget for millions of Transformer multiply-accumulates |
| **Arithmetic** | Integer multiply/divide, but **no FPU, SIMD, vector unit, or fused MAC** | FP32 attention becomes slow software routines; one MAC needs separate multiply and add instructions |
| **Registers** | 32 architectural 32-bit integer registers; `x0` is fixed at zero, with about **28 practical registers** for our inner kernel | Large GEMM tiles spill to the stack; this is why our $4\times2$ tile beats $8\times2$ |
| **On-chip SRAM** | 400 KB physical; about **321 KB usable application region** in our build | Activations, scratch space, stack, heap, and Wi-Fi must share this memory |
| **Flash** | **4 MB** onboard; about 3.1 MB available to the app partition | Weights fit in flash, but flash cannot act like fast working memory |
| **PSRAM** | **None** | Large tensors must be tiled, reused, or never materialized |
| **Wireless** | 2.4 GHz Wi-Fi and Bluetooth LE 5 | Enables clustering, but lwIP/FreeRTOS consume SRAM and communication costs time |
| **Measured model throughput** | **67.45 model MFLOP/s; 42.2% MFU** on optimized Case 2 | Shows how much of our derived scalar arithmetic bound became useful Transformer work |
| **Cost** | About **S$7 per board** | Makes an eight-node physical AI cluster inexpensive and reproducible |

Chip specifications come from the [Espressif ESP32-C3 datasheet](https://documentation.espressif.com/esp32-c3_datasheet_en.html)
and [Seeed XIAO ESP32-C3 documentation](https://wiki.seeedstudio.com/XIAO_ESP32C3_Getting_Started/).
The usable-memory and model-throughput rows are our measured or derived project values.

## How we built it 🛠️

We redesigned the execution path around the ESP32-C3 instead of simply shrinking a
desktop implementation:

- 🔢 **Fixed-point attention:** Q15 activations, Q12 weights, integer QK/PV kernels,
  and LUT softmax removed software floating point from the hot loops.
- 🧮 **Register-aware GEMM:** A $4\times2$ tile fits the RV32IMC register budget;
  hand-scheduled assembly hides multiply latency with independent accumulators.
- ⚡ **Operator fusion:** GEMM epilogues combine bias, scaling, quantization, and
  residual updates, eliminating extra activation-buffer passes.
- 🧠 **SRAM-aware execution:** A 16-row, head-sequential schedule lets the model and
  Wi-Fi/lwIP stack coexist without PSRAM.
- 📡 **Shape-aware parallelism:** We use batch data parallelism for independent
  inputs and alternating token-row parallelism for batch-size-one attention.
- 🔄 **Communication overlap:** A FreeRTOS task moves K/V data while computation
  continues.
- 📊 **`tinyprof`:** Our profiler records operator cycles, heap, stack, static
  memory, and traffic.

We used C/C++, RISC-V assembly, Arduino-ESP32, FreeRTOS, PlatformIO, Python,
PyTorch, NumPy, Codex, and Claude Code.

## Challenges we ran into 😵‍💫

- 🐢 **No FPU:** Ordinary `float` operations are software routines on the C3, so
  attention dominated the original runtime.
- 💥 **Wi-Fi caused an SRAM clash:** Model activations, FreeRTOS, and lwIP all
  competed for the same few hundred kilobytes.
- 🧮 **Register pressure:** An $8\times2$ GEMM tile spilled values to the stack; the
  smaller $4\times2$ tile was faster.
- ⚖️ **Causal attention is unbalanced:** A contiguous token split gives one board
  much more work. Alternating token rows makes the triangular workload nearly even.
- 📡 **Communication could erase the gain:** K/V exchange had to overlap with useful
  computation, while every kernel still passed the numerical gate.

This was not one optimization—it was a constant balancing act between **speed,
memory, communication, and correctness**.

## Accomplishments that we're proud of 🏆

Before parallelism, we reduced Case 2 from **42.15 s to 1.996 s on one C3**—a
**21.1× single-board speedup**.

| Case | Baseline, one board | Optimized, one board | Parallel boards | **Total speedup** |
|---:|---:|---:|---:|---:|
| 01 | 2,697.6 s `*` | 127.360 s | **33.713 s** (8 C3s) | **80.0×** `*` |
| 02 | 42.15 s | 1.996 s | **1.276 s** (2 C3s) | **33.0×** |
| 03 | 168.6 s `*` | 7.960 s | **4.218 s** (4 active C3s) | **40.0×** `*` |
| 04 | 674.4 s `*` | 31.840 s | **8.438 s** (8 C3s) | **79.9×** `*` |
| 05 | 5,395.2 s `*` | 254.720 s | **67.451 s** (8 C3s) | **80.0×** `*` |
| 07 | 295.05 s `†` | 30.427 s | **3.963 s** (8 C3s) | **74.5×** `†` |
| 09 | 2,697.6 s `†` | 138.027 s | **28.508 s** (8 C3s) | **94.6×** `†` |
| 10 | 2,697.6 s `†` | 138.536 s | **29.793 s** (8 C3s) | **90.5×** `†` |
| 11 | 2,697.6 s `†` | 138.610 s | **51.604 s** (8 C3s) | **52.3×** `†` |
| 12 | 547.95 s `†` | 33.879 s | **4.282 s** (8 C3s) | **128.0×** `†` |

Only the Case 2 baseline is a physical measurement. `*` is a direct batch projection
from that baseline; `†` is a FLOP-normalized estimate for a different model shape.
Every optimized and parallel time is a physical measurement of the complete
four-layer body, with host transfer excluded consistently.

Case 2 gains **21.1×** from single-board optimization and **1.56×** more from its
two-board token-row split. Eight Wi-Fi workers achieve **8.00× node scaling** against
one identical tiled worker. Our Cases 1–5 cluster sweep passed **213/213 forwards**.

The complete Transformer and Wi-Fi stack fit **without PSRAM**, and every headline
result is backed by reproducible profiling and validation artifacts.

## What's next for NotGPUAttention 🔮

Next, we want to:

- Run a **trained sensor or time-series model**
- Support larger cases and extend token parallelism beyond two boards
- Measure **energy consumption per forward**
- Improve reliability, security, and automatic board discovery

Our long-term goal is to make capable local AI possible on hardware that is
**small, affordable, private, and available everywhere**. We want to keep exploring
the boundary between AI systems and low-level hardware—one byte, register, and clock
cycle at a time. ⚡
