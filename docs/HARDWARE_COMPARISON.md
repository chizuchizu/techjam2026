# Where an ESP32-C3 sits next to a laptop, a consumer GPU, and an H200

This note puts the hardware this project actually runs on into context against
the machines the competition assumes. It exists because the brief says
"optimize a Transformer **on a given GPU**" and we answered with a $5
microcontroller; the honest framing of that choice needs numbers on both sides.

**Provenance is marked on every figure.** `[M]` measured in this repository,
`[D]` vendor datasheet or specification, `[E]` derived estimate. No `[E]` cell
should be quoted as a result.

---

## 1. The four platforms

| | Seeed XIAO ESP32-C3 | Dev laptop (this host) | MacBook Pro (M4 Pro) | Consumer GPU (RTX 4090) | Datacenter GPU (H200 SXM) |
|---|---|---|---|---|---|
| Part | ESP32-C3 (RV32IMC) `[D]` | Ryzen AI 9 HX 370 `[M]` | Apple M4 Pro `[D]` | AD102 `[D]` | GH100, 141 GB `[D]` |
| Cores | 1 core @ 160 MHz `[D]` | 12C/24T, ~5.0 GHz `[M]` | 12C CPU + 20C GPU `[D]` | 128 SM / 16,384 CUDA `[D]` | 132 SM / 16,896 CUDA `[D]` |
| FP unit | **none** — soft-float `[D]` | AVX-512 (Zen 5) `[D]` | NEON + GPU + ANE `[D]` | tensor cores `[D]` | 4th-gen tensor cores `[D]` |
| Process | 40 nm `[D]` | TSMC N4 `[D]` | TSMC N3E `[D]` | TSMC 4N `[D]` | TSMC 4N `[D]` |

The single most consequential row is the third one. The C3 has **no hardware
FPU**: every `float` multiply is a compiler-emitted software routine. That one
fact is why the whole optimisation log in
[`benchmarks/case-02/optimisation/`](../benchmarks/case-02/optimisation/) is
about getting off floating point and onto int8/int16 fixed point, while a GPU
report would be about tensor-core occupancy.

---

## 2. Computational capability

Peak arithmetic throughput, best usable numeric format:

| Platform | Peak throughput | Format |
|---|---:|---|
| ESP32-C3 | ~0.16 GOP/s `[E]` | int32 scalar, 1 op/cycle |
| Ryzen AI 9 HX 370 (CPU) | ~1.9 TFLOP/s `[E]` | FP32 AVX-512 |
| Ryzen AI 9 HX 370 (NPU) | 50 TOPS `[D]` | int8, XDNA 2 |
| Radeon 890M (iGPU) | ~5.9 TFLOP/s `[E]` | FP32 |
| M4 Pro GPU | ~9.2 TFLOP/s `[D]` | FP32 |
| M4 Pro Neural Engine | 38 TOPS `[D]` | int8 |
| RTX 4090 | 82.6 TFLOP/s `[D]` / 165 TFLOP/s `[D]` | FP32 / FP16 tensor (dense) |
| H200 SXM | 67 TFLOP/s `[D]` / 989 TFLOP/s `[D]` / 1,979 TFLOP/s `[D]` | FP32 / BF16 tensor / FP8 tensor |

Peak spread from C3 to H200 at each part's best format: **about 12,000,000x.**

### The same workload on all of them

Case 2 is `B=1, S=128, D=128, H=4, F=128, L=4`, causal. Counting multiply-add
as 2 ops, one complete four-layer forward is **134.2 MFLOP** dense
(117.4 MFLOP if the causal mask's skipped half is deducted). The batch cases
are integer multiples: case 1 (B=64) is 8.59 GFLOP, case 5 (B=128) is
17.18 GFLOP.

| Platform | Case-2 forward | Effective rate | % of peak |
|---|---:|---:|---:|
| ESP32-C3, initial baseline | 42.15 s `[M]` | 3.2 MFLOP/s `[M]` | ~2% |
| ESP32-C3, optimised (opt23) | 1.990 s `[M]` | 67.4 MFLOP/s `[M]` | ~42% |
| 2x ESP32-C3, token-row split | **1.276 s** `[M]` | 105 MFLOP/s `[M]` | ~33%/board |
| Laptop CPU, 1 thread | ~1–3 ms `[E]` | ~0.05–0.1 TFLOP/s `[E]` | ~5% |
| RTX 4090 | ~0.1–0.3 ms `[E]` | — | **<1%** |
| H200 SXM | ~0.1–0.2 ms `[E]` | — | **<1%** |

Two things to read off this table.

**The ESP32 result is a good one for the ESP32.** 42% of the machine's scalar
peak on a Transformer, on a part with no FPU, is close to the ceiling — the
remaining gap is loads, address arithmetic, and quantisation scaling, not
wasted cycles.

**Case 2 is a terrible workload for a GPU.** At `B=1` there is not enough
parallelism to fill 128 SMs, so an H200 running case 2 is bound by kernel
launch overhead and spends well under 1% of its arithmetic capability. A 4090
and an H200 would land within ~2x of each other on this case despite a 6x gap
in peak. That is exactly the "kernel launch overhead" failure mode the rules
list, and it is why the competition ships batch cases 1/3/4/5 alongside it.

Where the GPUs win outright is case 5 and case 14: 17.18 GFLOP of independent
batch work is one large fused kernel on an H200 and is 127 seconds of wall
time on two C3s.

---

## 3. Memory and cache

This is the axis on which the project's design decisions were actually made.

| Platform | Registers / core | L1 | L2 | L3 / LLC | Main memory | Bandwidth |
|---|---|---|---|---|---|---|
| ESP32-C3 | 32 x 32-bit `[D]` | 16 KB cache, **flash XIP only** `[D]` | none | none | **400 KB SRAM** `[D]` | ~1.3 GB/s `[E]` |
| Ryzen AI 9 HX 370 | 512-bit vector file `[D]` | 48 KB D `[M]` | 1 MB/core `[M]` | 16 MB `[M]` | 16 GB LPDDR5x `[M]` | ~136 GB/s `[D]` |
| M4 Pro | NEON | 128 KB D/P-core `[D]` | 16 MB cluster `[D]` | SLC | 24–48 GB unified `[D]` | 273 GB/s `[D]` |
| RTX 4090 | 256 KB/SM `[D]` | 128 KB/SM `[D]` | **72 MB** `[D]` | — | 24 GB GDDR6X `[D]` | 1,008 GB/s `[D]` |
| H200 SXM | 256 KB/SM `[D]` | 256 KB/SM `[D]` | 50 MB `[D]` | — | **141 GB HBM3e** `[D]` | **4,800 GB/s** `[D]` |

The C3 row is the odd one out in kind, not just in degree:

- **SRAM is the whole memory system.** There is no DRAM behind it and no data
  cache in front of it. The 16 KB cache covers execute-in-place reads from the
  4 MB QSPI flash, not SRAM access. Nothing spills; it either fits or the
  program does not link.
- **Of the 400 KB, we get 327,680 B** (`dram0_0_seg`), and the full-forward
  firmware uses 274,564 B of it statically, leaving 53,116 B of heap `[M]`.
- Case 2's attention score matrix alone is `128 x 128 x 4 B = 65,536 B` — larger
  than the free heap. Block-online softmax cuts that working set to **160 B at
  tile 8** `[M]`. On an H200, the same matrix is 64 KB against a 50 MB L2: it
  never leaves cache and no such algorithm is needed.
- The same wall is why WiFi did not fit: the WiFi + lwIP stack costs ~85.5 KB
  static plus ~69 KB heap `[M]`, and the tiled rewrite that made room is
  documented in [`WIFI_ON_A_COMPUTE_NODE.md`](WIFI_ON_A_COMPUTE_NODE.md).

**Ratio: an H200's L2 cache (50 MB) is ~128x the C3's entire SRAM. Its HBM
(141 GB) is ~350,000x.**

---

## 4. Energy

Numbers here are `[E]` on both sides — we have not put a current probe on a
board, and GPU figures are TDP, not measured wall power. Treat this section as
an order-of-magnitude argument, and see §7 for what would make it a result.

| Platform | Active power | Peak efficiency |
|---|---:|---:|
| ESP32-C3 @160 MHz, radio off | ~0.07 W chip, ~0.1 W board `[D]/[E]` | ~1.6 GOP/J `[E]` |
| ESP32-C3, WiFi transmitting | up to ~1.1 W peak `[D]` | — |
| Ryzen AI 9 HX 370 | 28 W nominal, 54 W boost `[D]` | ~68 GFLOP/J `[E]` |
| M4 Pro (package, load) | ~40 W `[E]` | ~230 GFLOP/J `[E]` |
| RTX 4090 | 450 W `[D]` | ~367 GFLOP/J (FP16) `[E]` |
| H200 SXM | 700 W `[D]` | ~1,413 GFLOP/J (BF16) `[E]` |

Peak efficiency is not the interesting comparison — energy for one actual unit
of work is:

**Case 2, one forward (134.2 MFLOP):**

| Platform | Time | Power | Energy/forward |
|---|---:|---:|---:|
| ESP32-C3, optimised | 1.990 s `[M]` | ~0.1 W | **~0.20 J** `[E]` |
| 2x ESP32-C3 | 1.276 s `[M]` | ~0.2 W | ~0.26 J `[E]` |
| Laptop CPU | ~2 ms `[E]` | 28 W | ~0.06 J `[E]` |
| RTX 4090 | ~0.25 ms `[E]` | 450 W | ~0.11 J `[E]` |
| H200 SXM | ~0.2 ms `[E]` | 700 W | ~0.14 J `[E]` |

At batch-1, all five land inside a single order of magnitude — the C3 is
~10,000x slower than the H200 and within ~1.5x on energy, because a big GPU
running a tiny model is paying full power for almost no utilisation.

**Case 5, full batch (B=128, 17.18 GFLOP):**

| Platform | Time | Energy | GFLOP/J |
|---|---:|---:|---:|
| 2x ESP32-C3 | 127.4 s `[M]` | ~25 J `[E]` | ~0.67 `[E]` |
| Laptop CPU | ~34 ms `[E]` | ~1.0 J `[E]` | ~18 `[E]` |
| RTX 4090 | ~1 ms `[E]` | ~0.45 J `[E]` | ~38 `[E]` |
| H200 SXM | ~0.5 ms `[E]` | ~0.35 J `[E]` | ~49 `[E]` |

Once the work is large enough to fill the machine, the efficiency argument
inverts completely: the GPUs are **50–70x** better per joule than the C3
cluster. The microcontroller's energy story holds only at batch 1, where the
GPU cannot use what it is drawing.

---

## 5. Cost

| Platform | Unit cost | What you need around it | Notes |
|---|---:|---|---|
| Seeed XIAO ESP32-C3 | **~$5** | USB cable; host only for flashing | ESP32-C3 die alone ~$1 at volume |
| 2-board cluster (measured config) | **~$10** | 2 USB ports | The 1.276 s case-2 result |
| 8-board cluster (WiFi target) | **~$40** | WiFi AP | Target `ceil(B/8) x 1.99 s` |
| Dev laptop (this host) | ~$1,500 | — | Also the bench host |
| MacBook Pro 14" M4 Pro | ~$2,000–2,500 | — | 24 GB unified |
| RTX 4090 | ~$1,600–2,000 | ~$1,000 host + 850 W PSU | 24 GB caps model size |
| H200 SXM | **~$30,000+** per GPU | HGX baseboard; 8-GPU node ~$300k | Cloud ~$3.50–11/GPU-hr |

**The entire measured two-board cluster costs about 0.03% of one H200.**

Two derived ratios worth quoting in the report, both `[E]`, counting one
`B=1` forward as the unit of work:

- **Throughput per dollar (case 5, B=128):** H200 ≈ 8,500 forward/s per $1k
  `[E]`; 2x C3 ≈ 100 forward/s per $1k `[E]`. **The GPU wins by ~85x.**
- **Latency-bound work per dollar (case 2, B=1):** H200 ≈ 170 forward/s per
  $1k `[E]`; 2x C3 ≈ 78 forward/s per $1k `[E]`. **The GPU's advantage
  collapses to ~2x** — near parity — because it is idle for over 99% of its
  silicon on this shape.

Same hardware, same workload family, and an 85x advantage shrinks to 2x purely
by moving from B=128 to B=1. The cluster never *beats* the H200 per dollar; it
stops losing badly, which is the whole claim.

---

## 6. Summary

| Axis | C3 vs H200 |
|---|---|
| Peak arithmetic | H200 ~12,000,000x `[E]` |
| Case-2 wall time | H200 ~10,000x `[E]` |
| Case-2 energy | H200 ~1.4x better `[E]` |
| Case-5 energy | H200 ~70x better `[E]` |
| On-chip fast memory | H200 L2 is ~128x the C3's whole SRAM `[D]` |
| Main memory | H200 ~350,000x `[D]` |
| Memory bandwidth | H200 ~3,700x `[E]` |
| Unit cost | C3 ~6,000x cheaper `[D]` |
| Floating point | C3 has none; every float is software `[D]` |

What this project demonstrates is not that a microcontroller competes with a
datacenter GPU. It is that the optimisation problem the competition poses —
memory hierarchy, kernel fusion, launch overhead, numeric formats, work
partitioning across devices — is the *same problem* four orders of magnitude
down, and that solving it there is harder to fake: on a 328 KB machine with no
FPU, an algorithm that does not fit does not run at all, so every one of the
23 logged optimisations had to be real.

---

## 7. What would upgrade the `[E]` cells

In rough order of value per effort:

1. **Measure C3 board power** with an inline USB power meter during a case-2
   run — converts the entire §4 ESP32 column from `[E]` to `[M]` and makes the
   batch-1 energy claim defensible.
2. **Run `torch_transformer_benchmark.py` on the dev laptop** (`pip install
   torch`; the module is currently absent on this host) to replace the laptop
   latency estimates with measurements on the reference script itself.
3. **Run the reference script on any CUDA GPU** for a real per-case baseline;
   even a modest card fixes the shape of the GPU column.
4. **Instrument GPU power** via `nvidia-smi --query-gpu=power.draw` during the
   run rather than assuming TDP.

Until (1)–(3) are done, this document is context for the tech report, not
evidence in it.
