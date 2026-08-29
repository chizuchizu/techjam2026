# RV32IMC fixed-point GEMM + int32→Q15 requant epilogue techniques for MCU inference

**Date:** 2026-08-30 · **Research memo** (web + primary sources via firecrawl)
**Target:** ESP32-C3 (RV32IMC, 160 MHz, 4-stage in-order, no FPU / no bitmanip / no P-ext), the
6-per-layer Q15×Q12 → int32 GEMMs, and the attention QK/PV path of the esp32-baseline 4-layer
transformer (B=1, S=128, D=128, H=4).
**Scope vs. existing notes:** `int_gemm_kernel.md` already covers register blocking, `lw`-pair
"SIMD-in-register" loads, and dropping int64 accumulation; `int_pipeline.md` covers integer LayerNorm
and log/scale handling. **This note adds:** (1) published RV32/RISC-V GEMM kernels that actually exist
(none for pure RV32IMC — why, and what to copy from them), an instruction/register budget for a
hand-written 8 vs 16-lane MAC asm inner loop, and soft-float avoidance; (2) exact, citable
requantization epilogue math (gemmlowp/TFLM/CMSIS-NN) translated into `mul`/`mulh` instruction
sequences for RV32IMC, incl. bias folding and rounding-right-shift; (3) fully-integer softmax/attention
recipes (LUT exp, headroom-based 1/sum, reciprocal LUT) and their published accuracy at 8/16-bit.

Project state at write time: int32 accumulator persists; j-outer tiling with 8-row i-tiles + a KB0
8-MAC asm kernel already deployed. Everything below is additive.

---

## 1. RV32IMC fixed-point GEMM inner loops — assembly 8–16 lane MAC, j-tiling, soft-float avoidance

### 1.1 Reality check: what is *published* for RV32/RISC-V MCUs (and what is not)

There is **no published, production 8–16 lane MAC GEMM written against plain RV32IMC** (scalar, no V/P/B).
Every open-source RISC-V ML kernel assumes an instruction extension:

| Source | ISA the kernels target | What it looks like | Relevance to C3 |
|---|---|---|---|
| **PULP-NN** (Garofalo et al., arXiv:1908.11263) | RISC-V cores + DSP (P-ext) cluster | `dotp`, `pslev` packed loads, up to 15.5 INT-8 MAC/cycle on an octa-core cluster; **63× vs. a sequential single-core baseline implementing baseline RV32IMC** | Shows the *ceiling* of the RISC-V family, not what IMC can do; the RV32IMC baseline figure is exactly our constraint: scalar 1 MAC/cycle issue bounded |
| **muRISCV-NN** (tum-ei-eda) | RVV 1.0 + P 0.9.6 | Drop-in CMSIS-NN fork; RVV/P vector kernels, e.g. `muriscv_nn_softmax_s16.c` (identical algorithm to CMSIS), packed q15x2/q7x4 helpers | Structure is portable, instructions are not. The softmax s16 source copy is *pure C* (see §3) and is directly reusable on IMC |
| **ESP-DL** (espressif) | ESP32-S3 (Xtensa PIE) and **ESP32-P4 (RISC-V, custom `esp.*` packed/vector ext)** | Hand-written `.S` kernels: `esp-dl/dl/base/isa/esp32p4/dl_esp32p4_s*.S`, incl. `..._s16_requantize_linear.S`, `..._conv2d.S` | Notably: Espressif writes NN kernels for RISC-V targets in **assembly**, but P4's `esp.ldqa.s16.128`/`esp.vmulas.s16.qacc` ISA is absent on the C3. ESP-DL supports S3 and P4 only (no C3) |
| **CMSIS-DSP / CMSIS-NN** (ARM) | Cortex-M (NEON/Helium + DSP SIMD) | Scalar-C `arm_mat_mult_fast_q15.c` (transpose-B, 2 A-rows × 2 B-cols, `read_q15x2_ia` dual loads, 4 int32 accs) and `arm_nn_vec_mat_mult_t_s16.c` (4-parallel s16×s8 with int64 epilogue) | The scalar-C structure transfers 1:1; the ARM-specific `__SMLAD`/`SMMUL` don't. The q15 and s16 *C* algorithms are the best published scalar templates |
| **TFLite-Micro** (tensorflow) | any; **RISC-V uses generic C++ only** | `tensorflow/lite/micro/kernels/` contains optimized trees only for `xtensa`, `ceva`, `arc_mli`, `cmsis_nn`, `ethos_u` — **there is no riscv tree** | On RV32 TFLM executes the generic C++ integer kernels (verified by repo tree listing 2026-08-30). Any RV32IMC asm kernel is a differentiator vs. the whole TFLM-on-RV32 ecosystem |
| **Deeploy** (Scherer et al., arXiv:2408.04413) | Multicore RV32 cluster **with** ML instruction extensions + NPU | Auto-generated C kernels for Siracusa; 490 µJ/token @ 340 tok/s SLM on MCU | Again extension/NPU-dependent; demonstrates that plain-RV32IMC attention/GQ doesn't exist in their stack either |

**Takeaway:** the *only* realistic peer implementations we can cite for "Q7/Q15 × Q8/Q12 GEMM on a
no-FPU 32-bit MCU" are (a) CMSIS-NN's scalar **s16 = int16×int8 → int32 with int64 epilogue**
(`arm_nn_vec_mat_mult_t_s16.c`), (b) CMSIS-DSP's scalar q15 (int16×int16→int32) — both are what the
C3 port already mirrors at C level. The asm layer on top of the existing 8-MAC kernel is ours to write.

### 1.2 Register budget for 8 vs 16-lane MAC asm on RV32IMC (why 8 is the natural C-ABI size)

RV32 has 32 × 32-bit GPRs; x0 (zero), x1 (ra), x2 (sp), x3 (gp), x4 (tp) are effectively unavailable.
Practical budgets:

- **C-ABI asm kernel (the current 8-MAC one):** a0–a7 usable as scratch after arg setup, plus t0–t6,
  s0–s11 with save/restore → ~16–18 registers for a leaf.
- **8-lane microkernel (8-row i-tile × 1 j-col, K-inner):** 8 int32 accumulators + 1 resident B value
  (reused across 8 rows) + the A pointer(s) + loop counter + address = ~14–16 regs → **fits without
  spills**. This is why the deployed 8-MAC kernel is the right C-callable size.
- **16-lane (8 rows × 2 cols):** 16 accs + 2 B values + A operands/pointers + counters ≈ 21–24 regs.
  Feasible **only as a leaf asm kernel that borrows the full non-reserved register file** (custom ABI,
  wrapper saves/restores s-regs). The B-second-column load then doubles weight traffic unless you use
  `lw`-pairing (§1.3) so one 32-bit load delivers 2 weights. Alternative that keeps 16 accs *and* fits
  the C ABI: stream the A operand from SRAM every k-step (no A caching) at the cost of 8 extra `lhu`/iter
  → usually worse than 8-lane + L1-resident A.
- **Instruction floor (verified in `int_gemm_kernel.md`):** ≥1 insn/MAC issue for MUL+ADD alone,
  ~2.75 cyc/MAC with a 4×2 tile and no loads, ~0.625 loads/MAC at 8×2 before pairing. On a single-issue
  4-stage in-order core this is also roughly the stall floor when load→MUL latency is hidden by 8
  independent accumulators.

**Recommendation:** keep the 8-row i-tile; add *j-tile-2* only inside a leaf asm kernel that uses the
full register file, and feed both B columns with two `lw` loads (see §1.3). Don't push 16 accs in C.

### 1.3 RV32IMC-specific instruction tricks to bake into the asm inner loop

1. **32-bit "dual q15" loads.** One `lw` fetches two adjacent q15 operands; the **high lane** is
   `srai rd, t, 16` and the **low lane** is `slli rd, t, 16; srai rd, rd, 16` (both sign-extended). Per 2 MACs
   this is 1 load + 2 extracts + 2 `mul` + 2 `add` (vs 2 `lhu` + 2 `mul` + 2 `add`). CMSIS-DSP's
   `read_q15x2_ia` + `__SMLAD` is the ARM counterpart — the load-pairing idea transfers, the single
   dual-MAC instruction does not (RV32 needs 2 MUL + 1 ADD).
2. **`mulh` for free high halves.** RV32M's `mulh/mulhsu/mulhu` each issue in the same pipeline slot as
   `mul`. Any `(int64)x*y >> s` in the epilogue or in fp32-free scaling compiles to `mul`+`mulh`
   (2 instructions) — this is NOT soft-float. Write the 64-bit ops explicitly as `(int64_t)a*b` and let
   GCC emit `mul`/`mulh`; never call libgcc's 64-bit multiply as a function.
3. **Branchless rounding arithmetic right shift.** `RoundingDivideByPOT(x, e)` (gemmlowp) =
   `x>>e + ((x & ((1<<e)-1)) > (((1<<e)-1)>>1) + (x<0))`. RV32 version: `srai t, x,31` (sign mask),
   combine with the remainder-threshold compare; ~5 instructions, no branch. See §2.
4. **Avoid any `float`/`__floatunsisf`/libm** in kernels: ESP-IDF defaults to soft-float
   (`-msoft-float` for RV32 no-FPU) and a stray float cast drags `__addsf3`+`__muldf3` into the loop
   (~30–80 cycles each). Compile hot translation units with `-fno-single-precision-constant
   -ffreestanding` and grep the `.lst` for `__softfloat`/`__muldf3` symbols as a regression gate.
5. **`clz` is not free.** No Zbb on C3 ⇒ `__builtin_clz` lowers to a slow bit test (~10–20 cycles) or a
   256-entry table. The headroom computations in integer softmax and in Pow2 scaling should use a
   branchless clz helper (table + `or`) — see §3.

---

## 2. Fast int32→int16 (Q15) requantization epilogues, no soft-float

The per-GEMM epilogue `(int64)acc*Gx + Bx >> 30` is already integer, but the *canonical* published
forms below are shorter, add proper rounding, and give the exact RV32 instruction sequences. All three
ecosystems (gemmlowp/TFLite, CMSIS-NN, ESP-DL) converge on: **int32 accumulate → `(acc+bias) *
multiplier` in the high half → rounding right shift → saturate to Q15.**

### 2.1 The canonical: `MultiplyByQuantizedMultiplier` (TF/TFLite)

- Scalar form (TFLM `kernels/internal/common.h`, non-NEON branch):
  `MultiplyByQuantizedMultiplierSmallerThanOneExp(x, mult, shift<0)` =
  `RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(x, mult), -shift)`.
- **SaturatingRoundingDoublingHighMul(a,b)** = `(int32)((int64)a*b + nudge) >> 31`, with
  `nudge = 1<<30` for a·b≥0 else `1-(1<<30)` (rounds ties away from zero; saturates only on
  a==b==INT32_MIN). This equals ARM `VQRDMULH`. **RV32IMC sequence (no 64-bit lib call):**
  ```
  mul   lo, a, b      # low 32
  mulh  hi, a, b      # high 32   (same latency class as mul)
  srli  r, lo, 30     # rounding bit   (l>>30)&1
  andi  r, r, 1
  srli  c, lo, 31     # carry bit      (l>>31)
  slli  t, hi, 1
  add   t, t, c
  add   t, t, r       # + sign-dependent nudge; ~7 insns, branch-free
  ```
  This is the exact decomposition CMSIS-NN comments for its `arm_nn_doubling_high_mult_no_sat`
  (always-positive `+0x40000000` nudge): `(m1*m2) >> 31 + rounding == (u << 1) + (l >> 31) +
  ((l >> 30) & 1)` (u = mulh, l = mul), where "rounding" is the 30th bit of m1·m2. Careful: gemmlowp's
  *saturating* `SaturatingRoundingDoublingHighMul` biases the nudge by sign
  (`nudge = 1<<30` if a·b≥0 else `1-(1<<30)`) to round ties away from zero; on RV32 that costs one extra
  `srai`+`and`+`sub` to build the signed nudge before the add. `arm_nn_requantize`'s default branch uses
  the +2^30 form (no sat), while `MUL_SAT` (=`arm_nn_doubling_high_mult`) uses the sign-biased form.
- **RoundingDivideByPOT** double-rounds (each of high-mul and the shift rounds). Newer
  `TFLITE_SINGLE_ROUNDING` collapses to one rounding step and is what TFLM uses for int16 outputs:
  single round → smaller bias, fewer instructions.
- Note on the current project's `(int64)acc*Gx+Bx >> 30`: `multiplier = Gx << 1` and
  `shift = 31`-style reparameterization turns it into the exact `SaturatingRoundingDoublingHighMul`
  shape (`>>31` not `>>30`), which is what makes the 7-instruction sequence apply; keep the rounding
  offset positive for non-negative products and clamp the rare −2^31·−2^31 case.

### 2.2 CMSIS-NN: `arm_nn_requantize` and `arm_nn_requantize_s64` (direct Q15-output path)

From `Include/arm_nnsupportfunctions.h` (raw source captured):

- **`arm_nn_requantize(val, mult, shift)`** = `(val*mult) / 2^(31-shift)` with rounding. Single-round
  branch: `result = ((val*(int64)mult) >> (30-shift) + 1) >> 1`. Inline-assembly branch uses exactly
  the `mulh`-style doubling-high-mult + `divide_by_power_of_two` built from the bit identity above —
  i.e. **the RISC-V port is a 1:1 of this decomposition** (it never needs the ARM `smull`).
- **`arm_nn_requantize_s64(val64, reduced_multiplier, shift)`** (used for int16 outputs from int64
  accumulate, e.g. the s16 vec-mat kernel):
  `result = (val64 * reduced_multiplier) >> (14 - shift); result = (result+1) >> 1;`
  → this is the **int64→int16 Q15 epilogue**. The `val64*reduced_multiplier` is 2 RV32 instructions
  (`mul`+`mulh`), then the `>>(14-shift)` + round is ~4 more → **~7–9 instructions per output
  element, no soft-float.** This is the exact shape to adopt if you ever fold attention into an int64
  accumulator before requant; with the current int32 accumulators use §2.1 instead (2–3 insns less).
- **`arm_nn_divide_by_power_of_two(dividend, exp)`** (rounding divide by 2^e, midpoints away from
  zero): threshold = `((1<<e)-1)>>1 (+1 if negative)`; RV32 = the §1.3/2.1 branchless ~5-insn form.

### 2.3 Bias folding and per-channel multipliers

- **Bias folding** = add the int32 bias to the accumulator before the single multiply
  (`acc += bias -> (acc)*mult >> shift`); TFLM integer FC and CMSIS-NN both do exactly this
  (`arm_nn_vec_mat_mult_t_s16`: `result_64 += *bias++` then one `arm_nn_requantize_s64`). When bias is
  folded in, it must be re-quantized at the *multiplier* scale, not at the activation scale, or you get
  systematic offset error.
- The project already folds BX into the 64-bit `(acc*Gx + Bx) >> 30`; the published equivalent is to
  keep `Bx` as a *single additional int64 constant added after the multiply-doubling step*, which keeps
  products in int32 and only widens the sum (1 `mulh`-folded add). Either way: do the rounding add
  *once*, after bias, never per-MAC.
- Per-channel: load `mult[col]`/`shift[col]` once per output column in the j-outer epilogue (not per
  element). On RV32, keep mult+shift packed as (shift<<24|mult) in 32-bit words and unpack once per
  column — CMSIS-NN does exactly this with `output_mult/output_shift` arrays.

### 2.4 Scale-free (dyadic / POW) requant and "≈ no multiply" special cases

- If the required scale is an exact power of two (e.g. moving Q12 weights ↔ Q15 acts or repeating same
  Q15→Q15 layers), the epilogue collapses to **right-shift + rounding only** — `arm_nn_requantize`
  with `multiplier` normalized to 2^k handles it in `~4` instructions. Full dyadic pipelines (I-ViT's
  `fixedpoint_mul`, FQ-ViT PTF power-of-two factors) exist precisely to maximize these shift-only
  epilogues (see `int_pipeline.md`).
- Avoid float-generating casts entirely; use the Q15 macros from CMSIS-NN (`NN_Q15_MAX/MIN`, `CLAMP`)
  for the final saturation to int16.

---

## 3. Softmax / attention fully integer on 32-bit MCU without FPU — published schemes + accuracy

### 3.1 Two published, directly copyable integer softmax engines (both are pure C, RV32IMC-ready)

**(a) LUT-based int16 softmax — TFLM `SoftmaxInt16` / CMSIS-NN `arm_softmax_s16` (identical math):**
full integer pipeline, no divide, no exp:
1. row max → `diff = input - max`;
2. `scaled = MultiplyByQuantizedMultiplier(diff, input_multiplier, input_left_shift)` (int64 multiply +
   rounding shift — the §2 sequence);
3. recenter to symmetric int16, then **exp LUT lookup with linear interpolation (513-entry, 1 KB)**:
   `index = 256 + (sat >> 7); offset = sat & 0x7f; exp = lut[index] + ((lut[index+1]-lut[index])*offset + 64) >> 7`;
4. accumulate `sum_of_exps` (Q16.15);
5. `headroom = clz(sum)`; `shifted_sum = ((sum << (headroom-1)) + (1<<13)) >> 14`;
6. **reciprocal via 513-entry 1/(1+x) LUT** (`sym = shifted_sum - 98304`, same interpolation);
7. per element: `result = (exp_c * one_over_one_plus_x + (1 << (right_shift-1))) >> right_shift` with
   `right_shift = 31 - headroom`, then clamp [0, 32767].
   *Cost on RV32IMC:* ~10–18 instructions/element plus 2 LUT loads; the division by the row sum never
   happens (replaced by clz + reciprocal LUT). `headroom` needs the §1.3 `clz` helper (no Zbb).
   This is the closest published match to the project's Q15 attention and its ~O(0.02 s) exp block.

**(b) LUT-free int8 softmax — CMSIS-NN `arm_nn_softmax_common_s8` / gemmlowp `exp_on_negative_values`:**
- scaled fixed-point 5-integer-bits diff, filtered by `diff_min` (prunes exp≈0 terms, keeps accuracy);
- exp via **`arm_nn_exp_on_negative_values`**: a Q15 fractional polynomial
  (`result = 1895147668 + MUL_SAT(...)`) + a 4-stage "EXP_BARREL_SHIFTER" series of signed
  doubling-high-mults by {e^(−2^−k)} constants selected by bits of the remainder — branch-free but
  **~15 × MUL_SAT (~90 instructions) per exp on scalar RV32IMC** → only viable where log2/exp granularity
  is coarse enough to amortize, or where a LUT is unacceptable (e.g. per-request dynamic scales).
- 1/sum for int8 out: `clz(sum)` normalization + polynomial `one_over_one_plus_x` Newton refine,
  again MUL_SAT-based (~40 insns).
- **Accuracy reference:** this is the exact engine behind MLPerf-Tiny int8 softmax on Cortex-M; at 8-bit
  with `diff_min` it keeps 12-bit accumulation (`ACCUM_BITS=12`) and is the standard int8 attention
  recipe in CMSIS-NN.

**(c) Pragmatic ranking for the C3 Q15 attention** (matches current code): use (a) LUT exp + reciprocal
LUT for Q15 QK; (b) is for int8-only pipelines or when a 2 KB pair of LUTs is unacceptable. A pure
shift-based exp (`0x7F000000 >> scaled`) is the cheapest but only power-of-2 accurate — fine as a
fallback only where QK is clamped aggressively.

### 3.2 Published quantized attention schemes with accuracy (GPU/CPU, but the numeric recipes transfer)

| Scheme | Integer part | Reported accuracy vs FP32 |
|---|---|---|
| **I-BERT** (arXiv:2101.01321) | INT32 MatMul/LN/GELU/Softmax end-to-end; integer sqrt + `1/σ`; full integer | RoBERTa-Base 86.0→86.3, RoBERTa-Large 89.0→89.5 (INT8 ≥ FP32) — already in `int_pipeline.md` |
| **I-ViT** (arXiv:2207.01405) | dyadic pipeline; I-LayerNorm Newton rsqrt; fixedpoint exp/softmax | ViT-S 81.39→81.27, ViT-B 84.53→84.76, DeiT-T 72.21→72.24 — in `int_pipeline.md` |
| **FQ-ViT / LIS** (arXiv:2111.13824) | log2 softmax: exp→BITSHIFT (4-bit attention budgets) | ~1% lossless first-time full-quant ViT — in `int_pipeline.md` |
| **QFlash** (arXiv:2604.25306, IJCAI-ECAI '26, github EfficientCompLab/qflash) | **integer-only FlashAttention**: tile-accumulated scale explosion solved by integer rescale; **shift-based exponential**; uniform-scale for integer comparisons; single Triton kernel | up to 6.73× vs I-ViT speedup; SQNR ≈32.5 dB A2 / 31.0 dB A7; ViT-S 81.38→82.24, ViT-B 85.10→86.84, DeiT-T 72.21→71.70, Swin-T 81.35→80.06 (per-tensor) |
| **IGQ-ViT** (arXiv:2404.00928, CVPR '24) | per-input-instance channel-group quant; **softmax attention quantized across tokens** | closes the ViT PTQ degradation gap (activation channel variance across instances is the main ViT quant killer — same lesson as the project's per-tensor Q15 work) |
| **Deeploy** (arXiv:2408.04413) | compiler-generated integer kernels for RV32 cluster + NPU (SLM, TinyStories) | 490 µJ/token @ 340 tok/s on MCU — proves fully-integer transformer inference *architecture* on RISC-V MCUs, but needs P-ext/NPU |

For the project's block-attention (row ≤ 128, full row softmax — not FlashAttention-tiled), the
**QK^T integer dot + row softmax + PV int32 dot** already present in the code is exactly the
non-tiled instance of what QFlash proves viable; the two things QFlash adds that apply here are
(1) keep one **uniform per-row scale** for QK before integer comparison/softmax (avoids scale-insensitive
rounding per token), and (2) the shift-based exp is safe on a scalar core where it is cheap (unlike
QFlash's GPU complaint) — but LUT exp is still ~5× cheaper per element on RV32.

---

## 4. Sources (all fetched 2026-08-30)

- PULP-NN: Garofalo, Rusci, Conti, Rossi, Benini, *PULP-NN: Accelerating Quantized Neural Networks on
  Parallel Ultra-Low-Power RISC-V Processors*, arXiv:1908.11263 — https://arxiv.org/abs/1908.11263
- muRISCV-NN (TUM e-DA) — https://github.com/tum-ei-eda/muriscv-nn
  (`Source/SoftmaxFunctions/muriscv_nn_softmax_s16.c`, `muriscv_nn_softmax_common_s8.c`, README)
- ESP-DL (Espressif) — https://github.com/espressif/esp-dl
  (`esp-dl/dl/base/isa/esp32p4/dl_esp32p4_s16_requantize_linear.S`; README: S3/P4 only)
- CMSIS-NN (ARM) — https://github.com/ARM-software/CMSIS-NN
  `Include/arm_nnsupportfunctions.h` (arm_nn_requantize, arm_nn_requantize_s64,
  arm_nn_doubling_high_mult_no_sat bit-decomposition, arm_nn_divide_by_power_of_two,
  arm_nn_exp_on_negative_values, arm_nn_one_over_one_plus_x_for_x_in_0_1),
  `Source/SoftmaxFunctions/arm_softmax_s16.c`, `arm_nn_softmax_common_s8.c`,
  `Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s16.c`, `arm_nn_mat_mul_core_1x_s8.c`
- gemmlowp — https://github.com/google/gemmlowp/blob/master/fixedpoint/fixedpoint.h
  (RoundingDivideByPOT, SaturatingRoundingDoublingHighMul, EXP_BARREL_SHIFTER)
- TFLite Micro — https://github.com/tensorflow/tflite-micro
  `kernels/internal/reference/softmax.h` (SoftmaxInt16 full algorithm),
  `kernels/softmax_common.cc` (513-entry LUT gen, input_multiplier/left_shift),
  `kernels/internal/common.h` (LUTSize=513, MultiplyByQuantizedMultiplier*),
  kernels tree (no `riscv` optimized dir — verified)
- TF Lite quantization_util.h — https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/internal/quantization_util.h
- QFlash — https://arxiv.org/abs/2604.25306 ; code https://github.com/EfficientCompLab/qflash
- IGQ-ViT — https://arxiv.org/abs/2404.00928
- Deeploy — https://arxiv.org/abs/2408.04413
- libfixmath (canonical no-FPU Q16.16 math library) — https://github.com/PetteriAimonen/libfixmath
- RISC-V *unprivileged ISA* (RV32M mul/mulh) — https://github.com/riscv/riscv-isa-manual
- RISC-V Packed-SIMD (P) spec (the dual-MAC/16-bit ops RV32IMC emulates) — https://github.com/riscv/riscv-p-spec
