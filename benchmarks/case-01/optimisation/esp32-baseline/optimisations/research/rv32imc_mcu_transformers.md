# RV32IMC / ESP32-C3 Transformer Inference: External Best Practices, Scheduling, and Quantization Evidence

**Status:** research memo (evidence-backed; citations = URLs). Author: research agent for the ESP32-C3
(160 MHz) 4-layer transformer (B=1, S=128, D=128, H=4, HD=32, L=4) optimization, current 2.982 s/forward.
**Date:** researched from live web + primary sources (Espressif TRM/CSR, arXiv, RISC-V specs, GitHub).

---

## 0. Verified platform facts (start here — they constrain everything)

| Fact | Evidence |
|---|---|
| Core is Espressif's own **"ESP-RISC-V CPU"**, **not** a SiFive E31/E24 core | ESP32-C3 TRM ch.1: "ESP-RISC-V CPU is a 32-bit core based upon RISC-V ISA comprising base integer (I), multiplication/division (M) and compressed (C) standard extensions. The core has 4-stage, in-order, scalar pipeline optimized for area, power and performance." ([TRM PDF](https://documentation.espressif.com/esp32-c3_technical_reference_manual_en.pdf); mirrored: [SparkFun copy](https://cdn.sparkfun.com/assets/9/1/2/7/8/esp32-c3_technical_reference_manual_en.pdf)). Also [ESP32-C3 Wireless Adventure ch.5.2](https://espressif.github.io/esp32-c3-book-en/chapter_5/5.2/index.html) (4-stage pipeline, 160 MHz, 400 KB SRAM + 384 KB ROM). |
| Silicon implements **RV32IMC only** — no A/F/D/V, **no B (bitmanip), no P (packed SIMD)** | TRM ch.1 `misa` (0x301) register, read-only hardware value: M=1, I=1, C=1; A=0, F=0, D=0, V=0; **B="Reserved = 0"**, P bit field "Reserved = 0"; U-mode = 1, S-mode = 0. |
| Single-core, up to 160 MHz, 4-stage in-order scalar, 1 insn/cycle peak issue | TRM ch.1 + ch.3 ("single-core processor with a four-stage pipeline"); no dual-issue, no OoO, no vector units. |
| Cycle counter available in U-mode | `mcycle`/`mcycleh` read via `csrr`; used for cheap profiling. See [ctrlsrc.io "Counting CPU cycles on ESP32-C3 and ESP32-C6"](https://ctrlsrc.io/posts/2023/counting-cpu-cycles-on-esp32c3-esp32c6/) (documents that e.g. `j` costs 2 cycles on C3, 1-2 on C6 — i.e., per-instruction latencies are *not* all 1). |
| Flash execution penalty is real | Executing hot inner loops from SPI flash (I-cache) costs fetch bubbles; TRM advertises zero-wait SRAM/cache access only over IRAM/DRAM interfaces. See [ESP32 forum: "ESP32-C3 Instruction load latency from flash"](https://esp32.com/viewtopic.php?t=46188). **Implication: hot GEMM loops must run from IRAM; data from DRAM.** |
| ESP-IDF GCC target string | RV32IMC: `-march=rv32imc` (older GCC) / `-march=rv32imc_zicsr_zifencei -mabi=ilp32` (GCC >= 12, since `zicsr`/`zifencei` were split from I). See [ESP-IDF GCC migration guide](https://docs.espressif.com/projects/esp-idf/en/stable/esp32c3/migration-guides/release-5.x/5.1/gcc.html). |

**Why this matters:** every "killer trick" that depends on vector (RVV), packed-SIMD (P-ext), bitmanip (B), or
atomics (A) is *unavailable on the C3 in silicon*, not just in software. Only instruction scheduling, register
blocking, memory placement, and fixed-point arithmetic restructuring can move your GEMM cost.

---

## 1. Q1 — GEMM inner-loop scheduling to get below ~5 cyc/MAC (RV32IMC, in-order, single-issue)

### 1.1 Instruction-count lower bound (what the hardware can physically do)

Per `int16 x int16 -> int32` MAC the ISA forces at least:
- load activation (1x `lh`), load weight (1x `lh`), `mul` (1), `add` (1) = **4 instructions minimum/MAC**,
  if you cannot reuse a loaded operand across outputs.

With a **tile** you amortize loads:
- **8x1 tile (your head-GEMM shape), weight-stationary:** per K-step = 1 weight lh (reused by 8 rows) + 8 act lh + 8 mul + 8 add = **25 insn / 8 MAC ~= 3.1 cyc/MAC floor** (loop + pointer overhead excluded).
- **4x2 tile (your oproj/f1/f2 `j-tile-2 x IBLK=4` shape), both operands reused:** per K-step = 4 act lh (each reused by 2 cols) + 2 weight lh (each reused by 4 rows) + 8 mul + 8 add = **22 insn / 8 MAC ~= 2.75 cyc/MAC floor**.

Since the core issues at most 1 insn/cycle, these are hard floors on this silicon. **Two conclusions:**
1. Your 5.2 cyc/MAC (qkv) and ~5.2 cyc/MAC core (oproj/f1/f2) are ~1.6-1.9x away from the floor: real headroom
   exists from *scheduling and loop hygiene*, not from a fundamentally better algorithm.
2. Sub-3 cyc/MAC on this core is impossible for 16-bit operands without SIMD — anyone claiming otherwise for
   plain RV32IMC is selling something.

Supporting citation for the load-vs-compute trade-off: PULP-NN's matrix-multiplication design analysis is
explicitly built around "how many MAC instructions you can set with one load" with kernel tiles 2x1, 4x2, 4x4 —
wider tiles win on scalar cores too, even without their `pv.sdotsp.b` (P-ext) SIMD. ([PULP-NN, arXiv:1908.11263](https://arxiv.org/abs/1908.11263); Fig.9 region).

### 1.2 The dominant stall: dependent MUL->ADD chains (why you need >= 4 independent accumulators)

In-order scalar cores stall the pipeline on a read-after-write (RAW) dependency to a slow op. A single
accumulator loop `acc += a[k]*w[k]` costs `mul_latency + add` per MAC; if MUL latency is 2-3 cycles you
automatically land at 5-6 cyc/MAC — exactly the band you are in. The fix (already partly in place: you use
"8 int32 accs") is to keep **as many independent accumulator chains as fit the register file (~8 is easy in
rv32imc; 16 is fine for a pure inner loop)** and interleave so no MUL's result is consumed by the immediately
next instruction. Sketch of the 4x2 inner loop body (8 accs, 2 cols):

```
    # regs: a0..a3 = 4 activations for the 4 rows, w0,w1 = 2 weights (this K step)
    lh  t0, 0(a_act)      # act row0 (reused by both cols)
    lh  t1, 2(a_act)
    lh  t2, 4(a_act)
    lh  t3, 6(a_act)
    lh  w0, 0(w_ptr)      # weight row k, col0 (reused by 4 rows)
    lh  w1, 2(w_ptr)      # weight row k, col1
    # 8 independent MACs; interleave columns so RAW distance >= 2
    mul  aA0, t0, w0 ; add  acc00, acc00, aA0
    mul  aB0, t0, w1 ; add  acc01, acc01, aB0
    mul  aA1, t1, w0 ; add  acc10, acc10, aA1
    mul  aB1, t1, w1 ; add  acc11, acc11, aB1
    mul  aA2, t2, w0 ; add  acc20, acc20, aA2
    mul  aB2, t2, w1 ; add  acc21, acc21, aB2
    mul  aA3, t3, w0 ; add  acc30, acc30, aA3
    mul  aB3, t3, w1 ; add  acc31, acc31, aB3
    addi aa, aa, 8    ; addi ww, ww, 4 ; addi kk, kk, 1 ; bne kk, K, .loop
```
Per 8 MACs: 6 loads + 8 mul + 8 add + ~4 loop/addressing = ~26 insn -> **~3.3-3.6 cyc/MAC realistic, ~2.75
ideal**. Squeeze recommendations:
- **Pre-increment addressing** (`addi` on the pointer instead of load-with-offset); use compressed `c.addi`/`c.add` where possible.
- **Unroll K by 2-4** (amortize `addi`/`bne`, expose more ILP); keep RAW dependencies >= 2 instructions apart.
- **Run the loop from IRAM** — flash I-cache misses add a fetch bubble per miss and wreck in-order issue.
- **Keep accumulation int32; quantize once per output, not per MAC** (`sra` inside the loop adds 1+ insn/MAC).

### 1.3 Measured/reference baselines (RV32IMC-class, scalar and SIMD)

- **PULP-NN** (GAP-8, 8x RV32IMCXpulpV2 cores + P-ext SIMD): up to **15.5 MACs/cycle INT-8** across 8 cores,
  "63x with respect to a sequential implementation on a single RISC-V core implementing the baseline RV32IMC
  ISA"; INT-4/INT-2 at 0.186/0.181 cyc/MAC ~= 2.4x over INT-8. On a **single scalar RV32IMC core** their
  load/MAC cycle analysis is the guidance in Sec 1.2. ([arXiv:1908.11263](https://arxiv.org/abs/1908.11263); [RSTA version](https://royalsocietypublishing.org/rsta/article/378/2164/20190155/111584/PULP-NN-accelerating-quantized-neural-networks-on)).
- **muRISCV-NN** — a RISC-V port of CMSIS-NN (bit-exact w/ CMSIS-NN, integrates with TFLM + microTVM, runs
  MLPerf-Tiny MobileNet/ResNet/AutoEncoder), with **RV32IMC scalar kernels as well as RVV (Zve*) and P-ext
  (v0.9.6) kernels**. This is the closest thing to "CMSIS-NN-style weight-stationary kernels you should copy"
  for RISC-V. ([GitHub](https://github.com/tum-ei-eda/muriscv-nn); paper: ["muRISCV-NN: Challenging Zve32x Autovectorization...", 10.1145/3637543.3652878](https://dl.acm.org/doi/10.1145/3637543.3652878))
- **CMSIS-NN** (ARM's reference for exactly this style of scalar kernel engineering) — weight-stationary 2x2
  output tiles, 8+ accumulators, `arm_convolve_1x1_s8_fast` (1x1 conv -> GEMM), single requant at output with
  precomputed offsets; shows how much single-issue scalar kernels improve when per-element requant is stripped.
  ([CMSIS-NN docs](https://arm-software.github.io/CMSIS-NN/v7.0.0/index.html), [GitHub](https://github.com/ARM-software/CMSIS-NN))
- **InstMeter** measures *per-instruction* cycles on ESP32-C3 to predict DL energy/latency — evidence that
  instruction-level (per-op) cost modeling on the C3 is both measured and practical for a DL stack.
  ([arXiv:2603.04134](https://arxiv.org/abs/2603.04134))

**Bottom line for Q1:** 4x2 tiles + 8 independent accs + IRAM execution should take the qkv (5.2) and
oproj/f1/f2 (5.2 core) GEMMs toward ~3.3-4.0 cyc/MAC. Don't expect below ~3 without SIMD you don't have.

---

## 2. Q2 — Cheaper fixed-point softmax (exp2 LUT is already near-optimal)

Your softmax = exp2 LUT (~20 us/row) inside a ~0.6 s QK+PV path. **Softmax is not your bottleneck**; the QK/PV
GEMMs are (~16.8 M MACs/forward @ ~5 cyc/MAC). Chasing cheaper softmax has near-zero payoff unless it also
removes GEMM work. Evidence that the LUT approach is already state of practice:

- Standard int8/int16 softmax in quantized inference is exactly a **256/512-entry exp LUT + one per-row
  1/sum (integer division)**; "hardware" does the same single-cycle memory-read trick. See the integer-only
  numeric treatment at [cahuja1992.github.io "The Integer Trick - How Hardware Really Calculates Softmax"](https://cahuja1992.github.io/19INT8Softmax).
- **I-BERT** (integer-only BERT) replaces exp with a **second-order polynomial** via the base-2 identity
  `2^y = 2^floor(y) * 2^frac(y)`, keeping max abs error 1.9e-3, and computes GELU/Softmax/LayerNorm fully in
  integers — the canonical "no exp function at all" design. Relative to a LUT the gain is tiny; both are
  O(rows x 128) lookups + a division. ([arXiv:2101.01321](https://arxiv.org/abs/2101.01321), Sec 3.4-3.6)
- **HCCS** (2026): "clipped-linear softmax surrogate" (head-calibrated, int8-native) removes exp *and* division;
  it is motivated by exp/LUT being a bottleneck on *int-vector DSP machinery* (AMD AI Engine), **not** on
  scalar MCUs — it underlines that on cores without exp hardware the LUT is the low-cost baseline.
  ([arXiv:2604.02292](https://arxiv.org/abs/2604.02292))

Suggestions that *do* matter for the QK/PV path instead of swapping the LUT:
1. **One integer division per row** for `1/sum` — compute the reciprocal scale once, multiply each LUT entry by
   it, instead of per-element divides.
2. Fold **sqrt(d_k) scaling (÷sqrt(32))** into the QK epilogue/LUT scale so no per-element scale.
3. Because attention rows are independent, unroll the row loop so softmax(row i) overlaps the PV GEMM of row
   i-1's output — hide the ~20 us/row completely.
4. Keep everything int32 through P*V; quantize P only after the row divide (I-BERT/TFLM flow, Sec 5).

"Anything beat our ~5.3 cyc/MAC QK+PV meaningfully?" — Only floating the epilogues out and the Sec 1 scheduling
(4x2 tiles already apply to QK/PV since HD=32, S=128). A scalar-bound ~3.5-4 cyc/MAC is the realistic ceiling
gain (~1.3-1.5x). You are not leaving huge softmax money on the table.

---

## 3. Q3 — MUL latency on ESP32-C3: what it is and whether it explains 5 cyc/MAC

- The C3 core is **not SiFive E31** (see Sec 0). It is Espressif's ESP-RISC-V, 4-stage in-order scalar. For a
  *reference on the class*: SiFive E-series docs state the E31 has "a 32-bit per cycle hardware multiply and a
  1-bit per cycle hardware divide" (1-cycle multiply *throughput* on that SiFive core) — but **no
  Espressif-published MUL latency exists for the C3**. ([E31 manual](https://manuals.plus/m/65ab5f3bd94bdcca68a3ee11956d7858d237229d58ed6b060be874d418c59aa6.pdf); [SiFive E24 manual](https://sifive-china.oss-cn-zhangjiakou.aliyuncs.com/Standard%20Core%20IP/e24_core_complex_manual_21G2.pdf))
- **Measurable in 30 min on your board** (you have hardware + toolchain — you measured 2.9 s/forward). Use the
  U-mode `mcycle` CSR (methodology: [ctrlsrc.io](https://ctrlsrc.io/posts/2023/counting-cpu-cycles-on-esp32c3-esp32c6/)):

```c
// dependent chain  -> MUL latency + ADD (cycles per MAC in a single-accumulator loop)
for (int i=0;i<1000;i++) x = x*w;
// independent chains -> issue-limited throughput
for (int i=0;i<1000;i++){ x0=x0*w0; x1=x1*w1; x2=x2*w2; x3=x3*w3; }
```
Interpretation:
  - dependent ~ 4-5 cyc/MAC and independent ~ 1.5-2  -> you are RAW-stall/accumulator-starved: add ILP (Sec 1.2).
  - independent also ~ 3+ -> you are load/issue-bound (flash fetch or load-port): move to IRAM/DRAM.
- **Does MUL latency influence 5 cyc/MAC?** Structurally yes: with a 1-2 cycle MUL latency a single-accumulator
  inner loop lands exactly at 5-6 cyc/MAC. Your 8-row head GEMM already has 8 independent accumulators, so its
  5.2 cyc/MAC is *not* mul-latency-limited — it is load/pointer/loop-overhead-limited -> Sec 1.2 changes apply.
- **Outer- vs inner-product layout:** on a scalar in-order core the distinction matters less than on vector
  cores (where outer product broadcasts operands). What matters is *which operand is reused across which
  output*, captured by the tile shape: 4x2 reuses each activation across 2 output columns and each weight
  across 4 rows -> lowest load count (Sec 1.1). Keep weights stationary (same weight row serves every query
  row), stream activations.

---

## 4. Q4 — Published ESP32-C3 / RV32IMC transformer & quantized-inference numbers; tricks worth stealing

No published ESP32-C3 transformer benchmark shows sub-100 ns/MAC for scalar int16 GEMM — that needs SIMD the
part lacks. The useful published anchors:

| Source | What it gives | Steal this |
|---|---|---|
| **PULP-NN** ([arXiv:1908.11263](https://arxiv.org/abs/1908.11263)) | 8x RV32IMC+P-ext cluster: 15.5 MACs/cyc INT-8; 63x vs single-core RV32IMC baseline | im2col-free inner GEMM with 4 accumulator registers; load/MAC ratio analysis; wide-acc then single requant |
| **muRISCV-NN** ([GitHub](https://github.com/tum-ei-eda/muriscv-nn)) | RISC-V (incl. RV32IMC scalar) reimplementation of CMSIS-NN; TFLM/microTVM integration; bit-exact with CMSIS-NN | **Drop-in scalar kernels** — the exact "CMSIS-NN-style weight-stationary double-buffered" source you asked about, on RISC-V |
| **Tiny transformers on low-power MCUs** ([arXiv:2404.02945](https://arxiv.org/abs/2404.02945)) | Tiny transformers on STM32H7 (M7), STM32L4 (M4), GAP9 (RV32IMCXpulpV2). **Fused-Weight Self-Attention (FWSA)** merges QK and PV into weight-fused GEMMs, eliminating Q/K/P intermediate stores and cheap softmax overhead; ~2.9x avg speedup over stock libs on RISC-V+ARM; end-to-end ~2.8 ms (M7) / ~16 ms (M4-class) for small attention models; notes general NN libs lack efficient softmax -> first attention-tailored MCU kernel set | **FWSA (fuse QK and PV as one GEMM family, drop Q/K/P intermediate stores)** — your single biggest structural win; **don't rely on TFLM for attention** |
| **InstMeter** ([arXiv:2603.04134](https://arxiv.org/abs/2603.04134)) | Per-instruction instruction dictionary for ESP32-C3 (and M4/M7/M33), TFLM energy/latency modeling | Per-op latency/energy modeling to guide micro-optimization |
| **ESP-DL / ESP AI roundup** ([badiot](https://badiot.com/2026/02/26/esp32-ai-tinyml-espressif/)) | Espressif's quantized inference (esp-dl) targets **ESP32-S3 and ESP32-P4 only** (S3 = Xtensa w/ vector instr; P4 = RISC-V). **No C3 support.** | Confirms no vendor fast path for C3 — your hand kernels are the right call |
| **espllm** ([GitHub](https://github.com/ahmedbarakat207/espllm)) | Bare-metal int8/BitNet-b1.58 MoE transformer + MQA + RoPE engine for ESP32 (Xtensa) / ESP8266; ~81 KB active SRAM, ~3 MB flash | Architecture reference for quantized attention/MoE in 100-KB-class SRAM; not C3 numbers |
| **tiny-transformer-esp32** ([README, ESP32-C3-compatible](https://github.com/RajBalan2002/tiny-transformer-esp32)) | 134k-param transformer: FP32 ~0.51 MB, INT8 ~0.128 MB (fits your 320 KB class) | Confirms INT8/Q15-class C3 footprints are realistic; no timing numbers published |

**"TFLM RVV kernels":** TFLM ships no official RISC-V-vector kernels; the RISC-V vector kernels you have heard
about are muRISCV-NN's (Zve32x) integrated into TFLM — useless for the C3 (no V-extension). The scalar RV32IMC
muRISCV-NN/CMSIS-NN kernels are the ones portable to you.

---

## 5. Q5 — Deferring fp32 epilogues via wider Q15 residual streaming: is the plan sound?

**Short answer: sound — it is the mainstream design.** The canonical published pattern is I-BERT's: INT8
matmuls with **INT32 accumulation, then Softmax/GELU/LayerNorm computed directly on the accumulated integer
result (integer-only implementations), then requantized once** — integer GELU (2nd-order poly), integer
Softmax (exp2 + LUT/poly), integer LayerNorm (integer sqrt); max errors 1.8e-2 / 1.9e-3.
([arXiv:2101.01321](https://arxiv.org/abs/2101.01321)). TFLM/CMSIS-NN int8 does the same: int32 accumulation
through a layer, single requant at the layer boundary. Your "keep attention/FFN outputs Q15 until LayerNorm"
is this pattern generalized to 16-bit mids with LayerNorm as the single requantization point — good.

**Three concrete risks to close:**

1. **int32 overflow (the main one).** Q15xQ15 (32767^2 ~= 2^30) over a K=128 dot product ~= 2^37 — ~64x
   over the int32 limit (2^31). So "Q15 until LayerNorm" is safe only if per-layer input scaling drops the
   activation operands below Q15 (as your QKV path already does with Q12-style activations): Q15 weights x
   Q12-class activations -> ~2^26-2^27 per product, x128 ~= 2^33-2^34, *still 4-8x over int32* — so even that
   path needs a mid-shift (arithmetic right-shift inside the accumulation or per-tile scale) or fewer
   fractional bits. **Recompute per layer**: pick qa so that `K * 2^(qa+qw) <= 2^31` (e.g. qa=12 with
   per-tile >>4, or qa+qw<=24 for K=128). Bank the extra factor at LayerNorm, not per MAC.
2. **Exactly-once rounding.** One requant at LayerNorm (round-to-nearest `(x + (1<<(f-1))) >> f`) is strictly
   lower-error than requant at every GEMM boundary (double rounding). Do residual adds in int32 pre-LN.
3. **Epilogue fusion (bigger, surer win than softmax).** oproj/f1/f2 currently pay ~1.0 cyc/MAC
   (`floatsisf + mulsf3 + addsf3` x3) on top of a 5.2 cyc/MAC core = 6.2 total. Replacing that with the
   **fixed-point min/max + single-shift Q15-quant pass you already use on the head GEMM removes
   ~0.16 s/forward** (25.2 M MACs x 1 cyc/MAC / 160 MHz ~= 0.16 s).

**Budget check:** 320 KB SRAM fits this: 4 layers of D=128 activation streams in int16 + Q/K/V/P working
buffers (S x D x 2 B ~= 32 KB) + weight tiles in flash; keep only the current layer's activations in SRAM
(layer-stationary schedule, which you already have). Q15 residuals instead of fp32 halve those buffers.

---

## 6. Sidebar — the "bits extension (BMAC/BPMSET)" question on ESP32-C3

**The premise is wrong on two independent counts; do not build on it.**

1. **No B or P hardware on the C3.** `misa` (read-only, in silicon) reports only **I, M, C**
   (A=F=D=V=B=0; P field reserved=0) — TRM ch.1 (Sec 0). Any `-march=rv32imc*_b*` / `*_p` compilation emits
   instructions the C3 cannot execute: they raise illegal-instruction exceptions (or worse, trap-loop). ESP-IDF
   ships plain `rv32imc[_zicsr_zifencei]` GCC; there is nothing extra to enable — there is no hardware behind
   it. Also note ESP32-C3 never got an Espressif NN library (esp-dl covers S3/P4; Sec 4).
2. **`BMAC` and `BPMSET` are not RISC-V instruction names.** The bitmanip single-bit group (Zbs) is
   `bset/bclr/binv/bext`; Zbb = clz/ctz/cpop/zext.h/rev8/... ; Zba = sh1add/sh2add/sh3add; Zbc/Zbm etc. —
   see [RISC-V bitmanip examples](https://docs.riscv.org/reference/isa/v20260120/unpriv/bitmanip-examples.html)
   and [B-extension spec](https://docs.riscv.org/reference/isa/v20260120/unpriv/b-st-ext.html). The Packed-SIMD
   (P) extension has names like `smul8/umul8`, `kadd16/ksub16`, `smaqa`, `smal`, `pbsad`, and PULP-NN's
   `pv.sdotsp.b/.h` (GAP-8). "BMAC"/"BPMSET" match neither spec — likely a conflation of "bitmanip + MAC" from
   marketing or a different ISA.
3. **What inline asm *can* do on the C3 today (RV32IMC only):** dense straight-line inner loops with
   `lh/lw/mul/add/sra`, pre-increment addressing, compressed 2-byte encodings (`c.addi`, `c.add`, `c.j`), and
   `csrr mcycle` timing. No SIMD, no single-cycle popcount/ctz/clz, no atomics; shifts are barrel (1 cyc).
   For B/P-class speed you need ESP32-P4 or a chip with a real P/RVV unit — not the C3.

---

## 7. Concrete recommendation list (ranked by expected payoff)

1. **[1.3-1.7x on GEMM stages]** Re-shape inner loops to 4x2 tiles, 8 independent int32 accumulators,
   weight-stationary, pre-increment addressing, K-unroll 2-4, **execute from IRAM** — target ~3.3-4.0 cyc/MAC
   on qkv/oproj/f1/f2 (from 5.2). First run the `mcycle` microbenchmark (Sec 3) to confirm you are
   issue-bound, not mul-stall-bound.
2. **[~0.16 s/forward]** Kill oproj/f1/f2's fp32 soft-float epilogue (floatsisf+mulsf3+addsf3, ~1.0 cyc/MAC)
   with the same amax/minmax + single-shift Q15-quant pass you proved on the head GEMM. Cheapest sure win.
3. **[structural]** Evaluate FWSA-style fused QK+PV GEMMs (arXiv:2404.02945) to drop Q/K/P intermediate
   stores; keep softmax as-is (exp2 LUT ~20 us/row is not your bottleneck).
4. **[Q15 streaming]** Keep Q15 residuals + integer LayerNorm as single requant (I-BERT pattern); re-derive
   per-layer scale so `K * 2^(qa+qw) <= 2^31`; residual adds pre-LN in int32. Reclaims ~half the activation
   buffer vs fp32 streaming.
5. **Don't:** chase P-ext/B-ext (not in silicon), chase exp-less softmax (already optional), or expect <3
   cyc/MAC scalar GEMM.

---

## References (URLs)

- ESP32-C3 TRM (official): https://documentation.espressif.com/esp32-c3_technical_reference_manual_en.pdf
- ESP32-C3 TRM (SparkFun mirror, used for misa/CPU text): https://cdn.sparkfun.com/assets/9/1/2/7/8/esp32-c3_technical_reference_manual_en.pdf
- ESP32-C3 Wireless Adventure ch.5.2 (4-stage, 400 KB SRAM): https://espressif.github.io/esp32-c3-book-en/chapter_5/5.2/index.html
- ESP-IDF GCC migration (rv32imc_zicsr_zifencei): https://docs.espressif.com/projects/esp-idf/en/stable/esp32c3/migration-guides/release-5.x/5.1/gcc.html
- ctrlsrc.io - count cycles on ESP32-C3/C6 (mcycle): https://ctrlsrc.io/posts/2023/counting-cpu-cycles-on-esp32c3-esp32c6/
- PULP-NN: https://arxiv.org/abs/1908.11263
- muRISCV-NN: https://github.com/tum-ei-eda/muriscv-nn ; paper: https://dl.acm.org/doi/10.1145/3637543.3652878
- CMSIS-NN: https://arm-software.github.io/CMSIS-NN/v7.0.0/index.html ; https://github.com/ARM-software/CMSIS-NN
- I-BERT (integer-only softmax/GELU/LN): https://arxiv.org/abs/2101.01321
- HCCS (softmax surrogate): https://arxiv.org/abs/2604.02292
- Integer softmax LUT deep-dive: https://cahuja1992.github.io/19INT8Softmax
- Tiny transformers on low-power MCUs (FWSA): https://arxiv.org/abs/2404.02945
- InstMeter (ESP32-C3 instruction dictionary): https://arxiv.org/abs/2603.04134
- ESP-DL / ESP32 AI roundup (S3/P4 only): https://badiot.com/2026/02/26/esp32-ai-tinyml-espressif/
- espllm (MoE transformer engine on ESP32): https://github.com/ahmedbarakat207/espllm
- tiny-transformer-esp32 (C3-compatible, 134k params, INT8 0.128 MB): https://github.com/RajBalan2002/tiny-transformer-esp32
- ESP32 forum: instruction load latency from flash: https://esp32.com/viewtopic.php?t=46188
- RISC-V bitmanip spec/examples: https://docs.riscv.org/reference/isa/v20260120/unpriv/b-st-ext.html ; https://docs.riscv.org/reference/isa/v20260120/unpriv/bitmanip-examples.html
- SiFive E31 manual (32-bit/cycle multiply note): https://manuals.plus/m/65ab5f3bd94bdcca68a3ee11956d7858d237229d58ed6b060be874d418c59aa6.pdf
- SiFive E24 manual: https://sifive-china.oss-cn-zhangjiakou.aliyuncs.com/Standard%20Core%20IP/e24_core_complex_manual_21G2.pdf
