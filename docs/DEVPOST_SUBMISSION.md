## Inspiration 💡

What if **every tiny IoT device could run AI locally**—without a GPU, cloud
connection, or expensive hardware?

We challenged ourselves to run a complete Transformer on a **S$7 ESP32-C3
microcontroller**. It has only **321 KB of usable memory**, runs at **160 MHz**,
and does not even have hardware support for floating-point math.

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

| Milestone | Measured result | **Improvement** |
|---|---:|---:|
| Original Case 2, one C3 | 42.15 s | 1.00× |
| Optimized Case 2, one C3 | **1.996 s** | **21.1× faster** |
| Token-row split, two C3s | **1.276 s** | **33.0× vs baseline; 1.56× vs optimized** |
| Batch data parallel, eight C3s | Eight active workers | **8.00× node scaling** |
| Eight C3s vs fastest untiled C3 | Fully utilized batch cases | **3.78× fair cluster gain** |
| Cases 1–5 cluster sweep | **213/213 forwards passed** | **100% pass rate** |

These are physical device measurements. The **8.00×** result compares eight workers
with one identical memory-saving Wi-Fi worker; **3.78×** compares the cluster with
our fastest untiled single-board firmware.

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
