# Quantized / Fixed-Point Transformer Inference on MCU-class RISC-V (ESP32-C3)

**Status:** research memo · web + primary-source evidence (arXiv, IEEE, GitHub source code, Espressif docs).
**Date:** researched 2026-08-29. **Audience:** ESP32-C3 (RV32IMC, 160 MHz, no FPU/SIMD, 400 KB SRAM,
~2.4 MB flash weight budget) transformer project — current FAST pipeline ≈ 2.7–4.0 s/forward,
integer attention already in place (see `01_integer_attention.md`, `11_int32_attention_pv.md`,
`int_gemm_kernel.md`). Model: B=1, S=128, D=128, H=4, HD=32, F=128, L=4, ~399K params, Q15×Q12
fixed-point GEMMs, Q15 activations.

This memo answers three questions with citations:

1. **How should QK / softmax / PV be quantized** (8-bit vs 16-bit, per-head scales, integer or LUT softmax) and what accuracy hit is documented?
2. **Which "single-pass / fused" layer-kernel techniques** are published and how do they map to a no-FPU scalar core?
3. **What ESP32 / RISC-V-specific acceleration exists** (ESP-DL, ESP-NN, TFLite-micro RV32) and what is actually usable on the C3?

---

## TL;DR — highest-value actions for the current pipeline

1. **Move to mixed 16x8 (int16 activations + int8 weights)** instead of uniform 16-bit.
   This is TFLM's "16x8" scheme. It halves flash for weights (~2.4 MB → ~1.2 MB) with int16 precision
   on activations where the literature says precision matters most (residual stream, attention logits).
   Evidence: 8-bit *weights* are at/above full-precision for BERT-quality transformers with QAT
   (FullyQT, Q8BERT, I-BERT); 16-bit *activations* protect against the "structured outlier" problem
   (Qualcomm 2021). We already store Q12 weights — going to Q8 weights is a small step, big flash win.
2. **Keep attention logits/softmax path at int16, not int8**, unless you add per-head calibration.
   Evidence: attention is the most quantization-sensitive op (ShadowNPU: it is the op that falls back
   from NPU to CPU in production frameworks); all "universal" int8 pipelines (I-BERT, I-ViT,
   IntAttention) needed extra machinery (polynomial exp, shift-exp, per-head/per-channel scales,
   sparsity clipping) to stay lossless. On our scalar core, int8 saves memory bandwidth but not MAC
   count (1 `mul` either way), so the risk/benefit favors int16 attention.
3. **Replace / fuse the softmax exp path** using one of the verified integer recipes (below). We can
   drop the fp32 exp path entirely. Options, cheapest first: (a) 256-entry LUT(s) exactly as TFLM's
   int16 softmax does per-row exp + reciprocal LUT; (b) I-BERT 2nd-order polynomial exp — no LUT, all
   shifts; (c) IntAttention's 32-entry "IndexSoftmax" + sparsity-aware clipping if we later go int8.
4. **Fuse QKV into one GEMM** (`W_qkv = [W_q|W_k|W_v]`, read activations once, 3 weight pointers,
   one tiled loop, 3 output streams). Published as qkv_fusion (2× vs 3 separate Linears). On the C3
   the win is one fewer pass over x in SRAM and one tile-fill per layer instead of three.
5. **Fuse residual+LN+quantize and the PV requant** into the enclosing GEMM epilogues (already partly
   done — extend the I-BERT integer-LN recipe; see `13_int_layernorm.md` and `int_pipeline.md`).
6. **Ignore ESP-DL / ESP-NN kernels for this chip.** Their op set is CNN-only (no softmax/attention/
   LayerNorm/GELU) and assembly optimizations target ESP32-S3/P4/S31 (AI/PIE/QACC instructions),
   **not** the C3. ESP-PPQ is still the best *offline* calibration/QAT/error-analysis tool to borrow
   for choosing 8-bit vs 16-bit layers (its layerwise-error metric selects mixed-precision layers).

---

## 1. Fixed-point attention quantization (QK / softmax / PV)

### 1.1 Reference schemes (what the canonical papers actually do)

| Scheme | QK logits | Softmax | PV | Precision result | Source |
|---|---|---|---|---|---|
| **Q8BERT** (QAT, fine-tune) | INT8 (per-layer scalar scale, int32 accum) | FP32 softmax is dequant/requant | INT8 | ~4× compression, minimal GLUE loss vs FP32 BERT | [arXiv 1910.06188](https://arxiv.org/abs/1910.06188) |
| **I-BERT** (integer-only) | INT8 MatMul, INT32 accum | **integer-only** polynomial `e^x ≈ 1+x+x²/2` on INT32 logits → requantize | INT8 | RoBERTa GLUE ≈ FP (Base 86.0→86.3, Large 89.0→89.5); all nonlinear ops (GELU/Softmax/LN) in INT32 | [arXiv 2101.01321](https://arxiv.org/abs/2101.01321) |
| **I-ViT** (integer-only) | INT8, per-channel scales, 1/sqrt(d) folded via S^(1/4) decompose | **Shiftmax**: `exp(x)≈2^…` via integer bit-shift | INT8 | ≈ FP (ICCV'23); 3.7–4.1× on GPU int units via TVM | [arXiv 2207.01405](https://arxiv.org/abs/2207.01405) |
| **FullyQT** (8-bit, MT) | INT8 all-inclusive | INT8 softmax | INT8 | **8-bit ≥ FP32 BLEU** on most WMT tasks (first fully-quantized lossless MT) | [arXiv 1910.10485](https://arxiv.org/abs/1910.10485) |
| **IntAttention** (edge, Armv8) | INT8 | **IndexSoftmax**: 32-entry LUT + sparsity-aware clipping + direct integer normalization; **no dequant→softmax→requant** | INT8 | ≈ FP; softmax detour was **up to 65% of attention latency**; 3.7× vs FP16, 2.0× vs int8-lite pipelines | [arXiv 2511.21513](https://arxiv.org/abs/2511.21513) · [code](https://github.com/WanliZhong/IntAttention) |
| **HCCS** (softmax surrogate) | INT8 | **Per-head calibrated clipped-linear** mapped to int8 MAC | INT8 | stable, monotone; needs offline per-head calibration + QAT retrain; target AMD Versal + AI Engine | [arXiv 2604.02292](https://arxiv.org/abs/2604.02292) |

**The consistent headline:** a **fully integer, per-head/per-channel-scaled int8 pipeline is lossless for
BERT-class transformers**, provided (a) QAT or good PTQ with per-tensor/per-embedding-group ranges, and
(b) the non-linearities get a dedicated integer recipe — you cannot just "keep softmax in fp32" inside a
no-FPU core (that is exactly the dequant→softmax→requant detour that costs up to 65% of attention latency
in IntAttention's measurement).

### 1.2 8-bit vs 16-bit — accuracy evidence (incl. perplexity-side)

* **Weights:** 8-bit weights are effectively free with QAT. Q8BERT [1910.06188](https://arxiv.org/abs/1910.06188): fine-tune-time QAT → ~4× compression, minimal loss. FullyQT [1910.10485](https://arxiv.org/abs/1910.10485): fully-quantized 8-bit MT, BLEU ≥ FP32. I-BERT [2101.01321](https://arxiv.org/abs/2101.01321): INT8 weights/activations, GLUE = FP.
* **Activations / attention are the sensitive part:** Qualcomm (arXiv 2109.12948) shows transformers have **high dynamic activation ranges with structured outliers in the residual connections** that drive attention to specific tokens; a low-bit fixed-point format for activations is the hard constraint → their fix is *per-embedding-group* quantization, and they push **weights/embeddings to ultra-low bit-widths** while keeping activations wider. [arXiv 2109.12948](https://arxiv.org/abs/2109.12948) · [code](https://github.com/qualcomm-ai-research/transformer-quantization)
* **Universal 8-bit LLM data (perplexity intuition):** 500k-eval study over Llama-3.1 family: FP8 lossless; **well-tuned INT8 W8A8 ≈ 1–3% degradation**; INT4 weight-only rivals INT8. Format choice matters more than nominal bit-width. [arXiv 2411.02355](https://arxiv.org/abs/2411.02355)
* **Attention-specific sensitivity:** ShadowNPU observes production frameworks **fall back attention from NPU to CPU** because of quantization sensitivity — the community treats attention as the op most likely to break under low precision. [arXiv 2508.16703](https://arxiv.org/abs/2508.16703)
* **Small-model MCU-class evidence:** 4/6-bit QAT transformers for embedded time-series (our workload class) hit 8-bit-comparable loss; a 4-bit model raised test loss by only **0.63%** while running **132× faster / 48× less energy** vs an 8-bit embedded baseline (FPGA). **Counter-intuitive and important:** reducing bit-width did *not* consistently reduce latency/energy — on tiny models the nonlinear ops dominate, so naive bit-width cuts don't win. [arXiv 2407.11041](https://arxiv.org/abs/2407.11041) (best-paper IEEE AIoT'24) · pipeline [TinyTransformer4TS](https://github.com/tianheng-ling/TinyTransformer4TS) · companion [2505.17662](https://arxiv.org/abs/2505.17662).
* **Additional small-model QAT/edge evidence:** QAT position paper for embedded-FPGA time-series transformers [arXiv 2408.16495](https://arxiv.org/abs/2408.16495).

**Bottom line for the C3:** the risk/benefit favors **int16 activations + int8 weights (16x8)**, and int16
attention. Don't chase int8 activations without QAT for the whole graph (and even then, per the FPGA study,
you may not win; on a scalar core int8 mainly saves load bandwidth/flash, not MAC count).

### 1.3 Per-head scales (free precision)

* I-ViT folds `1/sqrt(d_head)` into a **4-step S^(1/4) decomposition of the combined per-channel scales** so the QK dot stays pure integer until the final requant. Per-head precision is the standard granularity there. [2207.01405](https://arxiv.org/abs/2207.01405)
* **HCCS carries per-head calibration parameters** (a scalar scaling per head, tuned offline) to preserve each head's statistics under the clipped-linear surrogate — a cheap way to keep a *different* scale per head without recomputing. [2604.02292](https://arxiv.org/abs/2604.02292)
* For our pipeline (HD=32, H=4): the QK output scale per head is `(s_q · s_k · S²)/sqrt(32)` with the row/col scales folded — a single scalar per head in the requant epilogue. PV output has a per-head scale too. Nearly free, and it removes the need for a global logit scale covering all heads.

### 1.4 Integer / LUT softmax — exact verifiable recipes

**TFLM source is the most actionable MCU-grade reference.** From `tensorflow/lite/kernels/internal/reference/softmax.h` (tensorflow/tflite-micro):

* **INT8 path (int8 in → int8/int16 out), no LUT:** per-row max; `input_diff = q - max`; rescale by `input_beta_multiplier` + `input_left_shift` (β = `1/temperature` folded at prepare); **fixed-point exp in Q5.26 via gemmlowp** (`exp_on_negative_values`, bit-based, no exp); `sum_of_exps` in int32; **integer reciprocal** `GetReciprocal(...)` on the sum; final `exp/Σexp` by integer multiply + rounding divide. (Q5.26 ≈ 26 fractional bits ⇒ effectively *16-bit-class* precision.)
* **INT16 path (int16 in → int16 out):** per-row max on int16; `[-65535,0] → [-10.0,0.0]` by integer multiply-shift; **256-entry exp LUT over [-10,0]**; `sum_of_exps` (Q16.15); **`1/(1+x)` LUT** for the reciprocal after a CLZ/headroom normalize; Q0.15 output. [softmax.h](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/kernels/internal/reference/softmax.h) · [softmax.cc](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/kernels/softmax.cc)
* Both are branch-light and need no FPU — directly portable to RV32IMC. **Our current two-pass exact-max + int PV (`01_integer_attention.md`) can replace its exp with the int16 TFLM LUT recipe or I-BERT's polynomial below.**

**I-BERT-style integer softmax/GELU/LayerNorm** ([2101.01321](https://arxiv.org/abs/2101.01321)):

* GELU ≈ `0.5·x·(1+tanh(k(x+0.044715x³)))`, k=√(2/π); evaluated integer-only via a degree-3 polynomial + tanh recomposition; I-ViT's ShiftGELU uses only bit-shifts.
* Softmax exp on INT32 logits via 2nd-order Taylor `e^x≈1+x+x²/2` in fixed point, then round-to-nearest QUANT-requant into the [0,1] output range; tanh from `tanh(t)=(e^{2t}-1)/(e^{2t}+1)`.
* LayerNorm mean/var computed straight on quantized values; inverse std via **integer square root (Newton)**; normalization = integer division by the reciprocal scale. We already run int32 LN stats (`13_int_layernorm.md`) — matches their recipe.

**Even cheaper surrogates (only after QAT/calibration):** HCCS per-head clipped-linear softmax [2604.02292](https://arxiv.org/abs/2604.02292); IntAttention IndexSoftmax 32-entry LUT + sparsity-aware clipping [2511.21513](https://arxiv.org/abs/2511.21513). Both trade a little accuracy for removing the exp entirely — target if softmax LUT cost is still too high.

### 1.5 Actions — quantization focus

- [ ] Add a **Q8 weight export** (`weights_q8.bin`) and route GEMM weights through int8 (activations stay Q15). Estimate flash and bandwidth; gate stays `|Δ|≤2e-3`.
- [ ] **Per-head QK/PV output scales** as single scalars folded into requant epilogues (I-ViT-style S^(1/4) if full integer dot needed).
- [ ] Replace softmax exp with **TFLM int16 256-entry exp LUT + reciprocal LUT** (int16 path) — keep the current exact-max two-pass structure.
- [ ] Reference-check the current int16 attention against an **int8 logits** variant to quantify sensitivity before considering int8 attention (evidence says don't bother unless bandwidth-starved).
- [ ] Use **ESP-PPQ's layerwise quantization-error metric** offline to pick which GEMMs can drop to 8-bit (its docs recommend exactly this for mixed precision). [esp-dl quant/deploy docs](https://deepwiki.com/espressif/esp-dl/6-model-quantization-and-deployment)

---

## 2. Single-pass / fused transformer layer kernels

### 2.1 Fused QKV projection (published proof + MCU mapping)

* **qkv_fusion** (cuBLAS + FlashAttention-2): concat `W_qkv=[W_q|W_k|W_v]` offline → **1 GEMM** instead of 3 → split+bias+transpose in one pass; **2.08× vs 3 separate `nn.Linear`** (Qwen3-7B, batch=4, seq=512). [github.com/hilaryKChen/qkv_fusion](https://github.com/hilaryKChen/qkv_fusion)
* MCU translation (our 3 Q15×Q12 GEMMs share x in SRAM): one tiled GEMM loop with 3 weight pointers per output tile, x rows read once and reused by Q/K/V weight columns — the pure load-amortization win that matters on this load-bound scalar core (see the cyc/MAC analysis in `int_gemm_kernel.md`). Flash layout: `W_qkv` contiguous so each length-128 weight row is one sequential XIP stream.
* Watch unit: 3 separate GEMMs = 3× x-row re-reads + 3 tile fills; fused = 1× x-read + 1 tile fill per layer; the split/bias/transpose is free (register blocking).

### 2.2 Fused residual + LayerNorm + quantize (no intermediate float buffers)

* Canonical statement (fp32 intermediate buffers are the cost on no-FPU cores): I-BERT's whole point is **no dequant anywhere** — all nonlinear ops run on the INT32 accumulator then one requant to INT8 [2101.01321](https://arxiv.org/abs/2101.01321). IntAttention measures the dequant→softmax→requant detour at **up to 65% of attention latency** [2511.21513](https://arxiv.org/abs/2511.21513).
* Residual scores fold cheaply: `x_res = quant(x + prev_out)` can be done in the GEMM epilogue **before** the value reaches a buffer (our `03_layernorm_fused_quant.md` / `04_oproj_ctx_fusion.md` threads; `int_pipeline.md` maps every remaining fp32 site).
* If LayerNorm itself is the target: integer LN (I-BERT recipe above) is validated lossless; removing LN at inference with retraining is an active line ("Transformers Don't Need LayerNorm at Inference Time", [arXiv 2507.02559](https://arxiv.org/abs/2507.02559), already in our `.firecrawl/dont-need-ln.md`); the SKKU "integer-only transformer with LayerNorm removal + linear approximations" shows the integer-only removal path [IEEE 11137617](https://ieeexplore.ieee.org/document/11137617).
* **Prefetching evidence:** FlashAttention's IO-aware tiling (fit QK/PV tiles in on-chip memory, overlap reads) is provably optimal on SRAM-sized memories [arXiv 2205.14135](https://arxiv.org/abs/2205.14135). On the C3: keep x and the active tile in DRAM, stream W from flash through the I-cache/DMA into SRAM before use; double-buffer the next layer's weights while the current layer runs. PULP-NN's design analysis ("how many MACs you can set up with one load", 2×1/4×2/4×4 tiles) applies verbatim to the scalar C3 even without P-ext [arXiv 1908.11263](https://arxiv.org/abs/1908.11263).

### 2.3 Fused softmax → PV (integer) and single-pass attention

* Keep the **two-pass exact-max softmax** you already have for numerical fidelity; replace exp with the LUT/polynomial recipe and **fold the softmax output scale into the PV GEMM input scale** so PV consumes int16 directly (no separate requant buffer). Reference: IntAttention's direct-integer-normalization does the same end-to-end, avoiding the float detour [2511.21513](https://arxiv.org/abs/2511.21513). For B=1, S=128 with a full attention row in SRAM, FlashAttention-style tiling is unnecessary; the win is in the fused requant + skipped buffers.
* "Single-pass" also applies to **oproj + context write + residual**: fuse the PV output requant and the attention-context store into one epilogue (see `04_oproj_ctx_fusion.md`; `int_pipeline.md` identifies the fp32 attention-ctx store as ~0.1 s to kill).

### 2.4 Actions — fusion focus

- [ ] Implement **fused QKV**: one tiled GEMM, `W_qkv` concat (weights pre-concatenated offline), 3 output streams, per-head QK scale in epilogue.
- [ ] Kill remaining fp32 residual/LN/ctx sites with integer LN + fold requants into GEMM epilogues (map `int_pipeline.md` §1 sites to the I-BERT integer recipes).
- [ ] **Double-buffer next-layer weights** from flash into SRAM during current-layer compute (FlashAttention IO principle); measure whether XIP misses dominate.
- [ ] Consider **Q8 weights with 16-bit activations** — halves W traffic on the load-bound core.

---

## 3. ESP32 / RISC-V-specific transformer inference acceleration

### 3.1 ESP-DL (including ESP-PPQ quantization) — **status: not transformer-capable, not C3-optimized**

* ESP-DL op set is **Conv2d / Pool2D / Gemm / Add / Mul / FFT etc.** — **no softmax, attention, LayerNorm, GELU** operators. Assembly-optimized kernels target ESP32-S3 (vector instructions), ESP32-P4/S31 (PIE/QACC SIMD) — the C3 gets generic C. The ESP-DL runtime does not cover a transformer. [espressif/esp-dl](https://github.com/espressif/esp-dl) · [README](https://github.com/espressif/esp-dl/blob/master/esp-dl/README.md)
* **ESP-PPQ (inside ESP-DL) is still valuable offline:** PTQ + QAT, **layerwise + graphwise quantization-error analysis** explicitly to find layers that should stay 16-bit in a mixed-precision model, calibration-dataset guidance, and `.espdl`/`.info`/`.json` (scales, zero-points) export. [quantization & deployment docs](https://deepwiki.com/espressif/esp-dl/6-model-quantization-and-deployment)
* Static IRAM/PSRAM memory planner and dual-core scheduling exist, but are S3-centric and don't apply to single-core C3. (Don't burn effort here.)

### 3.2 ESP-NN (Espressif NN kernel library for TFLM) — **C3 = generic kernels only**

* ESP-NN provides optimized kernels (elementwise add/mul, conv, depthwise, fc, pooling, prelu/relu6, **softmax**, hard_swish, mean) for TFLite Micro; assembly kernels exist for ESP32-S3 (vector) and ESP32-P4/S31 (PIE/QACC). **ESP32 and ESP32-C3 use "generic optimisations" (ANSI-C-level)** — no SIMD win available. No attention or LayerNorm kernels at all. Int8-centric. [espressif/esp-nn](https://github.com/espressif/esp-nn)
* Value for us: the C-level kernels (esp-nn + upstream TFLM) are reference implementations we can adopt/port for **fc / softmax**; e.g. TFLM int16 softmax recipe above. Espressif's TFLM fork (esp-tflite-micro) is the integration point. [espressif/esp-tflite-micro](https://github.com/espressif/esp-tflite-micro)

### 3.3 TFLite Micro on RV32 / RISC-V evidence

* **TFLM has no RISC-V-specific optimized dispatch in mainline** — RISC-V targets run the portable reference kernels (which is what we'd port anyway). Confirmed by the Polimi study "RISC-V Meets TFLite Micro: Insights from MLCommons Tiny Benchmark": they run TFLM on RV32 via Spike/Gem5 and analyze **CPI and branch mispredictions comparing in-order vs OoO cores** — directly relevant: our 4-stage in-order core favors branchless inner loops and small repeated kernels (softmax/exp LUT, unrolled GEMM) because a mispredict/extra branch is paid every iteration. [github.com/fabrizioaymone/riscv-tflite](https://github.com/fabrizioaymone/riscv-tflite)
* TFLM acceleration on embedded RISC-V is an active topic (DSP/B-extension-based GEMM kernels); background: [IEEE 11304685 "Accelerating TensorFlow Lite Micro on Embedded RISC-V Platforms via …"](https://ieeexplore.ieee.org/document/11304685). The C3 has **no B/P/V/A extensions**, so those speedups don't apply — see the ISA audit in `rv32imc_mcu_transformers.md` (verified `misa` B=0, P=0, V=0).
* TFLM softmax integer recipes (used above) are mainline source: [softmax.cc](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/kernels/softmax.cc) · [reference/softmax.h](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/kernels/internal/reference/softmax.h).

### 3.4 Closest published systems (steal designs, don't reinvent)

* **TinyTransformer4TS** (Ling et al.) — *integer-only quantized tiny transformers for embedded time-series* (forecast/classification/anomaly), PTQ+QAT, hardware-aware HPO (Optuna), RTL for ultra-small FPGAs (Spartan-7 XC7S15 8k LUTs, Lattice iCE40 UP5K). Closest published workload to ours. [GitHub](https://github.com/tianheng-ling/TinyTransformer4TS) · [paper 2407.11041](https://arxiv.org/abs/2407.11041) · [automation paper 2505.17662](https://arxiv.org/abs/2505.17662)
* **Deeploy** (PULP, ETH Zürich) — end-to-end **small language model on RV32 MCU** (Siracusa: RV32 with ML instruction extensions + NPU), 8/4-bit integer codegen, 340 tok/s, 490 µJ/token for a TinyStories SLM **without external memory**. The 0.01-level guidance: on MCU-class, sub-16-bit SLM attention is feasible when the model is trained/quantized for it — but their cores have ML extensions (SIMD) we lack. [arXiv 2408.04413](https://arxiv.org/abs/2408.04413)
* **tiny-transformer-esp32** — tiny (~134k params) transformer targeting ESP32-C3, FP32 ~0.51 MB → INT8 ~0.128 MB (post-quant; no timing published). Confirms C3 transformer feasibility, but it is a training/quant demo, not a tuned runtime. [GitHub](https://github.com/Rithvik-007/tiny-transformer-esp32)
* **riscv-gpt2-inference** — GPT-2 inference in C on RISC-V; uses **RVV intrinsics** (`rv64gcv`) for GEMM/LN — *for comparison only*: the C3 has no V, so this is the scalar baseline they compare against. [GitHub](https://github.com/IvanEnclonar/riscv-gpt2-inference)
* **T-Fixup** — ICML'20 init scaling that trains transformers **without LR warmup**; practically useful on-device: fine-tune-time QAT of our tiny model with short schedules converges reliably. [PMLR](https://proceedings.mlr.press/v119/huang20f.html) · [code](https://github.com/layer6ai-labs/T-Fixup)
* **Gradient-based QAT references** (learned scales): LSQ (learned step sizes, STE) [arXiv 1902.08153](https://arxiv.org/abs/1902.08153) · NVIDIA QAT explainer for low-precision accuracy recovery [blog](https://developer.nvidia.com/blog/how-quantization-aware-training-enables-low-precision-accuracy-recovery/) · continuous-relaxation QAT for LLMs [arXiv 2410.10849](https://arxiv.org/abs/2410.10849).

### 3.5 What the C3 *cannot* use (from the earlier ISA audit)

RV32IMC only: no `B` (bitmanip), no `P` (packed SIMD), no `V` (vector), no AI/PIE/QACC. So PULP-NN P-ext kernels, RVV kernels, ESP-DL/ESP-NN S3-P4 assembly, and DSP-based TFLM-RV acceleration are all unavailable. Only instruction scheduling, register blocking, memory placement (IRAM/DRAM/flash-XIP), and fixed-point restructuring can move cycles — as documented in `rv32imc_mcu_transformers.md` and `int_gemm_kernel.md`.

### 3.6 Actions — platform focus

- [ ] Adopt **TFLM int16 softmax** (exp LUT + reciprocal LUT) as the new softmax kernel; measure against the current fp32-exp path; keep exact-max two-pass.
- [ ] Pull **TFLM int8/int16 fc + LayerNorm reference implementations** for any op not already hand-written (esp-nn's C-level fc is a good speed baseline to beat).
- [ ] Borrow **ESP-PPQ layerwise error analysis** (offline) to choose Q8 vs Q12 weights per layer → target `weights_q8.bin` for the non-sensitive projections.
- [ ] If int8 activations are ever pursued, validate per-head calibration thresholds with a short **QAT fine-tune using T-Fixup init** (no warmup needed).

---

## 4. Sources (URLs)

**Papers & code**

- I-BERT: integer-only BERT (GELU/Softmax/LN polynomials + integer sqrt) — https://arxiv.org/abs/2101.01321
- Q8BERT — https://arxiv.org/abs/1910.06188
- I-ViT: Shiftmax / ShiftGELU, S^(1/4) decomposed scales — https://arxiv.org/abs/2207.01405 · https://github.com/zkkli/I-ViT
- FullyQT: fully quantized 8-bit MT, lossless — https://arxiv.org/abs/1910.10485
- IntAttention / IndexSoftmax: 32-entry LUT, sparsity clipping, integer norm, training-free — https://arxiv.org/abs/2511.21513 · https://github.com/WanliZhong/IntAttention
- HCCS: per-head clipped-linear softmax surrogate — https://arxiv.org/abs/2604.02292
- I-LW-DETR: fully integer-only lightweight detector (SD-ShiftGELU, constrained Shiftmax) — https://arxiv.org/abs/2607.24981
- Efficient Transformer Quantization challenges / per-embedding-group PTQ (Qualcomm) — https://arxiv.org/abs/2109.12948 · https://github.com/qualcomm-ai-research/transformer-quantization
- "Give Me BF16 or Give Me Death": 500k-eval precision trade-offs — https://arxiv.org/abs/2411.02355
- ShadowNPU: attention falls back from NPU due to quantization sensitivity — https://arxiv.org/abs/2508.16703
- Integer-only 4/6-bit QAT transformers for embedded time-series (best paper IEEE AIoT'24) — https://arxiv.org/abs/2407.11041
- Tiny transformers for embedded time-series on FPGAs (ISVLSI'25 automation) — https://arxiv.org/abs/2505.17662 · https://github.com/tianheng-ling/TinyTransformer4TS
- QAT position for embedded time-series transformers — https://arxiv.org/abs/2408.16495
- Deeploy: SLM on RV32 MCU without external memory — https://arxiv.org/abs/2408.04413
- PULP-NN design analysis — https://arxiv.org/abs/1908.11263
- FlashAttention IO-aware tiling — https://arxiv.org/abs/2205.14135
- LSQ (gradient-based QAT scales) — https://arxiv.org/abs/1902.08153 · Continuous-relaxation QAT — https://arxiv.org/abs/2410.10849
- T-Fixup (init without warmup) — https://proceedings.mlr.press/v119/huang20f.html · https://github.com/layer6ai-labs/T-Fixup
- Transformers Don't Need LayerNorm at Inference Time — https://arxiv.org/abs/2507.02559 (see `.firecrawl/dont-need-ln.md`)
- Integer-only transformer with LayerNorm removal + linear approximations (SKKU/IEEE) — https://ieeexplore.ieee.org/document/11137617
- NVIDIA QAT / low-precision accuracy recovery explainer — https://developer.nvidia.com/blog/how-quantization-aware-training-enables-low-precision-accuracy-recovery/

**Kernels / runtimes / ESP32**

- TFLM softmax reference (int8 gemmlowp path; int16 exp-LUT + reciprocal-LUT path) — https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/kernels/internal/reference/softmax.h · https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/kernels/softmax.cc
- qkv_fusion (fused QKV, 2×) — https://github.com/hilaryKChen/qkv_fusion
- ESP-DL / ESP-PPQ — https://github.com/espressif/esp-dl · quantization + deployment docs — https://deepwiki.com/espressif/esp-dl/6-model-quantization-and-deployment
- ESP-NN — https://github.com/espressif/esp-nn · esp-tflite-micro (TFLM fork) — https://github.com/espressif/esp-tflite-micro
- TFLite Micro on RV32 (MLCommons Tiny, in-order vs OoO, CPI/branch) — https://github.com/fabrizioaymone/riscv-tflite
- TFLM acceleration on embedded RISC-V (IEEE) — https://ieeexplore.ieee.org/document/11304685
- tiny-transformer-esp32 — https://github.com/Rithvik-007/tiny-transformer-esp32
- riscv-gpt2-inference (RVV reference implementation) — https://github.com/IvanEnclonar/riscv-gpt2-inference
- ESP32-C3 ISA facts / TRM (our `rv32imc_mcu_transformers.md` audit) — https://documentation.espressif.com/esp32-c3_technical_reference_manual_en.pdf

---

*Research notes: firecrawl `search` endpoint was intermittently rate-limited during this study; arXiv export
API and direct scrapes were used as fallbacks. The firecrawl research-index endpoint returned 404 and was
not used. Raw scraped pages live under `.firecrawl/mcq/pages/` and `.firecrawl/` in the repo root.*
