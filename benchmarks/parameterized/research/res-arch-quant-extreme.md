# Integer/Fixed-Point Transformer Math for ESP32-C3 and other No-FPU MCUs

**Primary-source survey.** Focus: Q15 activations x Q12 weights, int32/int64 accumulator
overflow, fixed-point softmax/exp LUT, causal masking, fixed-point GELU, and existing
open-source integer transformer kernels in C. Code names/URLs are exact; originals were
downloaded to `/tmp/research/sources/` and scraped pages to `/tmp/research/.firecrawl/`.



## 0. One-paragraph answer

A no-FPU MCU cannot run float MACs at usable throughput, so the whole transformer must run on
8/16-bit integer ops. Use int8 activations x int8 weights (CMSIS-NN
`arm_nn_vec_mat_mult_t_s8` / PULP `pulp_nn_linear_i8_i8_i8`) with int32 accumulators and
`(acc*mul + round) >> shift` requantization. Keep softmax and GELU integer (max-subtract
base-2 exp with a small LUT, and GELU via a q15 tanh/sigmoid LUT). If you truly need Q15
activations x Q12 weights, the product is Q27 with worst-case magnitude 2^27, so an int32
accumulator overflows after only ~16 worst-case terms: use int64 accumulators, split-K with
intermediate requantization, or drop to the CMSIS Q15xQ7 design (int32 safe to 512 columns).

For the 0.002 abs / 0.02 rel gate: a correctly requantized Q15 output passes the absolute gate
easily; the relative gate is binding and needs ~7+ fractional bits on the logits and >=15
fractional bits (or a good exp approximation + Q15) on softmax, because 8-bit probabilities
(step 1/255) fail a strict 0.002 absolute gate.



## 1. Fixed-point representation and product scale

Standard Q notation: **Qm.n** = m integer bits + n fractional bits (sign bit included in m).
A value x is stored as integer `I = round(x * 2^n)`; recovery gives `x ~= I / 2^n`.

The product of Qm1.n1 x Qm2.n2 has scale **2^-(n1+n2)**:

    (I_a * I_w) is the integer form of (a * w) with n1+n2 fractional bits.

A Q15 activation (|I_a| <= 2^15) times a Q12 weight (|I_w| <= 2^12) is a **Q27** product; the
integer accumulator `acc = sum(I_a * I_w)` represents `sum(a*w)` up to scale 2^-27. Output
requantization then applies a multiplier and right shift back to the output format.


## 2. Integer GEMM: Q15 x Q12, and the int32 overflow question

### 2.1 Worst-case per-term magnitude (signed)

Two weight conventions matter. Activations are full-range int16 Q15 (|I_a| <= 32768), i.e.
activations in [-1.0, 1.0).

| Operand coding | max |I_a*I_w| | log2 | terms in int32 | terms in int64 |
|---|---|---|---|---|---|
| acts Q15 (32768) x wgts Q12 (4096, range +-1.0) | 134,217,728 (2^27) | 27.0 | 16 | 6.87e10 |
| acts Q15 (32768) x wgts full-int16 Q12 (32768, range +-8) | 1,073,741,824 (2^30) | 30.0 | 2 | 8.59e9 |
| acts Q15 (32768) x wgts Q7 (128, CMSIS classic) | 4,194,304 (2^22) | 22.0 | 512 | 2.20e12 |
| acts Q7 (128) x wgts Q7 (128) | 16,384 (2^14) | 14.0 | 131,072 | 5.63e14 |

Term count = floor(2^31 / max_term) for int32 and floor(2^63 / max_term) for int64. Worst case
assumes every term at full positive range - a bound, not a typical value.

### 2.2 Practical (RMS) bound, uniform +-1.0 operands

If activations and weights are i.i.d. uniform on [-1,1], E[(a*w)^2] = 1/9 and a sum of N
products has standard deviation sqrt(N)/3. Corrected for the integer product scale:

- Q15xQ12 (scale 2^-27): int32 3-sigma limit ~= 256 terms.
- Q15xQ15 (scale 2^-30): int32 3-sigma limit ~= 4 terms -> always int64 for real GEMM.
- Q7xQ7   (scale 2^-14): effectively unlimited (~1.7e10 terms).
- Q15xQ7  (scale 2^-22): ~= 262,144 terms.

### 2.3 Conclusion / failure mode

**Q15 activations x Q12 weights do NOT fit in an int32 accumulator** for transformer hidden
sizes (d >= 64) in the worst case, and are statistically marginal at N=256 (3-sigma). Three
workarounds, all used in shipping kernels:

1. int64 accumulator, then 64-bit->8/16-bit requant. This is what CMSIS-NN does for
   int16xint16 GEMM (`arm_nn_vec_mat_mult_t_s16_s16`: `int64_t result` then
   `arm_nn_requantize_s64`). q15xq12 gives a 32-bit product; accumulate in int64.
2. Downshift weights to fewer fractional bits (Q12 -> Q7): halves dynamic-range fidelity but
   makes int32 safe to 512 terms (the CMSIS-NN precedent).
3. Split-K with intermediate requantization: accumulate K in chunks of <=16 columns, requant
   each chunk, sum the few partials in int32 (block accumulation).

The int8xint8 path is why production MCU transformers are int8: per term <= 2^14, int32 is
safe to 131,072 terms, and SIMD dot-products (SMLAD, PULP sdotsp4, MVE vmlava) pack 4-8 terms
per instruction.


### 2.4 Requantization (second half of GEMM)

PULP / pulp-transformer (`pulp_nn_utils.h`):

    static int8_t pulp_nn_quant_i8(int32_t phi, int16_t m, int8_t d) {
      int32_t x = (m * phi) >> d;
      return clips8(x);              // __builtin_pulp_clip_r(x, 127), saturate to s8
    }
    static int8_t pulp_nn_bn_quant_i8(int32_t phi, int32_t k, int32_t lambda, int8_t d) {
      return clips8(((k * phi) + lambda) >> d);   // fused per-channel scale/offset
    }
    static int8_t pulp_nn_requantshift_i8_i8(int32_t Im_in, int32_t mul, int32_t add, int32_t div) {
      int32_t intermediate = Im_in * mul + add;
      return (int8_t)clips8(intermediate >> div);
    }

Note `mul` is int32 in `requantshift` (attention/fused kernels) but int16 in
`pulp_nn_quant_i8` (FC kernel) - the Q8 multiplier is 16-bit to avoid a 48-bit product.

CMSIS-NN generalized form: `(acc*mul + (1 << (shift-1))) >> shift`, saturating to the output
type, with `mul` derived from the three real scales `M = round(2^shift * s_in*s_w / s_out)`
(legacy `arm_nn_requantize`), or the gemmlowp two-shift form `s = m0 * 2^(-n)` with
`m0 in [0.5, 1)` so the int32 multiplier never overflows (`arm_nn_requantize_s64` for int64).


## 3. Accumulator precedent table (CMSIS-NN, verified from source)

| Kernel | data | product max | accumulator | safeguard |
|---|---|---|---|---|
| `arm_nn_mat_mult_kernel_q7_q15` / `arm_fully_connected_mat_q7_vec_q15` | act Q15 (carries Q7 content), weight Q7 | 2^22 | int32 (`__SMLAD`) | reordered input; `MAX_COL_COUNT (512)` bounds the unrolled SMLAD loop; out `__SSAT(sum >> out_shift, 8)` |
| `arm_nn_vec_mat_mult_t_s8` / `arm_nn_mat_mult_nt_t_s8` | act s8, weight s8 | 2^14 | int32 | none needed up to 131k terms |
| `arm_nn_vec_mat_mult_t_s16_s16` | act s16, weight s16 | 2^30 | **int64** + `arm_nn_requantize_s64` | DSP fast path int32 for first <=512 cols then int64; `MAX_COL_COUNT (512)` in `arm_nnsupportfunctions.h`, comment "int64 accumulation is needed to not lose precision" |

The classic Q7xQ15 MAC body (`arm_nn_mat_mult_kernel_q7_q15.c`, CMSIS 5.8 DSP path):

    q31_t sum = ((q31_t)(bias[i]) << bias_shift) + NN_ROUND(out_shift);
    ...
    sum = __SMLAD(inA11, inB1, sum);   // two q15xq15 products + int32 sum, saturating
    ...
    *px++ = (q7_t)__SSAT((sum >> out_shift), 8);

`__SMLAD` adds two Q15xQ15 products per instruction; with Q7-content activations each product
is <= ~2^22, so hundreds of dual-MACs fit in the int32 sum.


## 4. Integer / fixed-point softmax and exp LUTs

All integer softmaxes share three steps: max-subtraction (stability), a base-2 or polynomial
exp for exp(x-max) <= 0, and an integer reciprocal for exp_i / sum(exp_j).

### 4.1 CMSIS-NN classic Q7/Q15 (base-2, shift-only exp, no fractional LUT)

`arm_softmax_q15` (CMSIS 5.8): exp is a plain power of two, so logits must already be in
octaves (scaled by log2(e)):

    shift = (uint8_t)__USAT(vec_in[i] - base, 5);
    sum  += 0x1 << shift;                       // exact 2^shift, integer shift
    int64_t div_base = 0x100000000LL;
    int output_base = (int32_t)(div_base / sum);      // one 64-bit divide per row
    // output: output_base >> (17 - (vec_in[i] - base)), saturating to q15

Q7 version: 3-bit/8-level saturating shift, `base = max - 8`, one integer division
`int output_base = (1 << 20) / sum;`, then `p_out[i] = __SSAT((output_base >> shift), 8);`.

Cheapest possible design (shifts + one divide), but it quantizes the logit to whole octaves -
too coarse for the 0.002/0.02 gate (Section 9) except for coarse classifier heads. Intended
for 8-bit heads, not high-accuracy attention.

### 4.2 Modern CMSIS-NN s8/s16 (`arm_nn_softmax_common_s8.c`, polynomial exp, no LUT)

    #define ACCUM_BITS 12
    diff = input[col] - max;
    if (diff >= diff_min)
        sum += DIV_POW2(EXP_ON_NEG(MUL_SAT(diff * mask, mult)), ACCUM_BITS);
    const int32_t headroom = CLZ(sum);
    ... ONE_OVER1(...)                            // Newton-style reciprocal
    // output: clamp(DIV_POW2(MUL_SAT(shifted_scale, EXP_ON_NEG(...)), bits_over_unit) + NN_Q15_MIN)

- `EXP_ON_NEG(x)` = `arm_nn_exp_on_negative_values` (in `arm_nnsupportfunctions.h`): Q31
  Taylor expansion plus masked power-of-two contractions:

      x = (val_mod_minus_quarter << 5) + (1<<28);
      x2 = MUL_SAT(x, x);
      result = 1895147668 +
        MUL_SAT(1895147668, x + DIV_POW2(MUL_SAT(DIV_POW2(MUL_SAT(x2,x2),2) + MUL_SAT(x2,x), 715827883) + x2, 1));
      SELECT_IF_NON_ZERO(1672461947) ... SELECT_IF_NON_ZERO(242)   // 2^(-1/4), 2^(-1/2), ... contractions

  `MUL_SAT` = saturating doubling-high Q31 multiply (VQRDMULH-like). No lookup table.
- `ONE_OVER1(x)` = `arm_nn_one_over_one_plus_x_for_x_in_0_1`: reciprocal by fixed-point
  polynomial + two Newton iterations.


### 4.3 TFLite / TFLite-Micro reference int8 softmax (gemmlowp fixed-point)

`tensorflow/lite/kernels/internal/reference/softmax.h`:

    // The representation chosen for the input to the exp() function is Q5.26.
    static const int kScaledDiffIntegerBits = 5;
    static const int kAccumulationIntegerBits = 12;
    ...
    input_diff = input_data[c] - max_in_row;
    if (input_diff >= diff_min) {
      input_diff_rescaled = MultiplyByQuantizedMultiplierGreaterThanOne(
          input_diff, input_beta_multiplier, input_beta_left_shift);
      sum_of_exps += Rescale<12>(exp_on_negative_values(FixedPointScaledDiff::FromRaw(input_diff_rescaled)));
    }
    FixedPoint0 shifted_scale = FixedPoint0::FromRaw(
        GetReciprocal(sum_of_exps.raw(), kAccumulationIntegerBits, &num_bits_over_unit));

Facts: the quantized logit is **Q5.26**; `input_beta_multiplier` folds in the attention scale
1/sqrt(d) (a multiply + left-shift, not a divide); `diff_min` discards underflowed exp terms -
this is also the causal-mask hook (Section 5); `exp_on_negative_values` and `GetReciprocal`
are gemmlowp primitives reused by CMSIS-NN. A LUT variant (`exp_with_lut`) exists separately.

### 4.4 I-BERT / PULP integer softmax (quadratic exp, `pulp_iSoftmax.c`)

Fetched body of `pulp-transformer/src/nn_fpu_requant_functions/src/pulp_iSoftmax.c`:

    int16_t xTilde; int8_t z, p;
    int8_t x_max = -128; uint32_t y_sum = 0; uint32_t y[rowDimension];

    for (i..rowDimension) x_max = max(x_max, pInBuffer[i]);

    for (i..rowDimension) {
      xTilde = pInBuffer[i] - x_max;        // i8 logits, max-subtract
      z = -(xTilde / log2);                 // integer part (log2 = 5 in the golden model)
      p = xTilde + z * log2;                // fractional part in [0, log2)
      y[i] = ((coeffA*(p+coeffB)*(p+coeffB) + coeffC) >> z) * (1 - (z>31 || z<0));
      y_sum += y[i];
    }
    for (i..rowDimension) pOutBuffer[i] = (uint8_t)((y[i]*(n_levels-1)) / y_sum);

Golden coefficients (`pulp_iSoftmax.py`): `coeffA=1, coeffB=7, coeffC=24, log2=5, n_levels=256`.
The quadratic `(p+7)^2+24` least-squares fits 2^p on p in [0,5); `>> z` applies the divide-by-2^z that the 2^z factor introduces.
Output is uint8 (0..255) via one integer division after the row sum: `(y*255)/y_sum`.


### 4.5 I-ViT (ShiftExp / ShiftMax)

Shift-only integer exp from the scraped I-ViT notes:

    I_p = I + (I>>1) - (I>>4)            // multiply by log2(e) ~= 1.4375, shifts only
    I_0 = 1 / S                          // precomputed (S = exp scale)
    q   = floor(I_p / (-I_0))            // integer part
    r   = -(I_p - q*(-I_0))              // remainder
    2^(S*(-r)) ~= ((-r)>>1) + I_0        // linear approx on the fraction
    I_exp = I_b << (N - q)               // final shift

- `ShiftMax` normalizes with `IntDiv(exp_i, sum, k_out)` = precomputed reciprocal
  `(1<<M)/sum` times an integer multiply.
- `ShiftGELU` uses `GELU(x) ~= x*sigma(1.702x)` with `1.702 ~= (1.1011)_2`, same ShiftExp +
  IntDiv (Section 6).

### 4.6 Softmax error-quality simulation (this survey)

Simulation: 500-2000 rows x 64 logits drawn N(0,1.2); integer base-2 softmax with logits in
octaves (round(logits*log2(e)*2^sf), sf fractional bits), exact 2^r exp, output quantized to
ob fractional bits:

| logit frac bits (octave) | output bits | max abs err | max rel err (p>=1e-3) | abs<0.002 | rel<0.02 |
|---|---|---|---|---|---|
| 7 (1/128 octave) | 15 (Q15) | 6.7e-4 | 1.7e-2 | yes | yes |
| 3 (1/8 octave) | 15 (Q15) | 1.1e-2 | 7.1e-2 | no | no |
| 7 | 8 (uint8) | 2.4e-3 | ~1.0 | no | no |
| 7 | 12 | 6.8e-4 | 1.1e-1 | yes | no |
| 7 | 15, 4-bit exp LUT | 4.0e-3 | 4.4e-2 | no | no |
| 7 | 15, 5-bit exp LUT | 2.0e-3 | 3.1e-2 | marginal | no |

Interpretation: output width dominates the absolute gate (8-bit probabilities step 1/255 ~=
3.9e-3 -> max rounding ~2.0e-3 plus exp error exceeds 0.002; probability outputs must be Q15
or at least 10-12 fractional bits). Logit width dominates the relative gate (>=7 fractional
bits safe). A 4-bit exp LUT is too coarse; prefer the polynomial exp
(`arm_nn_exp_on_negative_values`) or a >=8-bit LUT.

Note: relative error here is measured only on probabilities p >= 1e-3; tiny tail
probabilities (p ~ 1e-9) always have near-100% relative error after quantization and should be
judged by the absolute gate only.


## 5. Causal masking in integer attention

Two integer methods.

**A. Additive sentinel in the score matrix.** Before softmax, for each row i and column j>i:

    int32_t MASK = score_min;     // very negative sentinel in the score's fixed point
    if (j > i) score[i][j] = MASK;

Then run the normal integer softmax with `diff_min` / `__USAT` shift cap. Masked entries give
`exp(score - max) -> 0` (bit-shift underflow, or `diff >= diff_min` false in the TFLite/CMSIS
code of Section 4.3), so their probability rounds to 0. Rules:
- Do NOT let the masked value win the max pass: find max over j<=i only, or max() only causal
  positions.
- Pick MASK so `score_max - MASK` exceeds the exp cut-off (8 octaves for CMSIS Q15; the
  `diff_min` value for TFLite) - otherwise the masked column leaks tiny probability. It must
  not underflow the integer type (use the type's min, e.g. -(2^23), not a shift overflow that
  saturates to something small).

**B. Skip masked columns in the loops (no sentinel).** Iterate key columns only over 0..i,
keep a running max over those columns - no mask value, and the MACs are ~halved for triangular
attention. This is what streamed/decoder attention does naturally as keys/values are appended
from a KV cache one token at a time.

The PULP `matmulSoftmax_4x2_*` kernels compute dense SxS attention and apply no causal mask
(fetched file has no `j>i` test and no sentinel). They target a bi-directional ViT-style
encoder, so decoder attention must add method A or B around them, or use row-limited loops.

Recommended ESP32-C3 decoder recipe = method B: keep the Q row in int8, K/V tiles in a KV
cache, dot `q*k` into int32, `clip8((sum*mul)>>div)`, then integer softmax over the `i` valid
columns only. Causal masking costs zero sentinel logic.


## 6. Fixed-point GELU approximations

### 6.1 Exact and tanh forms (TFLite LUT generator)

`tensorflow/lite/kernels/internal/reference/gelu.h`:

    // Exact: GELU(x) = 0.5*x*(1 + erf(x/sqrt(2)))   (written with erfc for stability)
    inline float GeluTransform(float in) {
      return 0.5f * in * std::erfc(in * static_cast<float>(-M_SQRT1_2));
    }
    // Approximate: GELU(x) = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    inline float GeluTransformApproximate(float in) {
      return 0.5f * in * (1.f + std::tanh(kSqrt2dPi * (in + 0.044715f * in*in*in)));
    }

For int8/int16 inference TFLite computes one of these in float and stores a LUT (optionally the
approximate formula). On MCU: GELU = q15 LUT on x + optional linear interpolation, exactly
like CMSIS-NN activations. For the inner tanh, CMSIS-NN ships `tanhTable_q15` and
`sigmoidTable_q15` with linear interpolation in `arm_nn_activation_s16`
(`cmsis58_activations_q15.c`):

    value  = lookup_table[(uint8_t)(in >> shift_size)];
    value2 = lookup_table[(uint8_t)(1 + ((uint8_t)(in >> shift_size)))];
    out = ((full_frac - frac) * value + value2 * frac) >> shift_size;

Cheapest accurate integer GELU on Cortex-M: `x_q15`, `x3 = x*x*x` quantized,
`y = tanh_lut(c*(x + 0.044715*x3))`, `out = (x*(1+y)) >> 1`, with `sqrt(2/pi)=0.797884...`
and `0.044715` folded into the `mult` of the inner linear op.

### 6.2 Sigmoid form - reuses softmax code (MCUFormer / I-ViT)

MCUFormer (arXiv 2310.16898) uses `GeLU(x) ~= sigma(1.702*x)` (Eq. 7) and implements sigma
with the same integer exp/division as softmax. I-ViT (ShiftGELU) does the same with
`1.702 ~= (1.1011)_2` by shifts plus `IntDiv`. This is the preferred integer-transformer GELU
when integer softmax already exists: one more shift-exp call, no separate LUT.

### 6.3 PULP quadratic GELU (`pulp_gelu.c`, quoted)

    x = Im_in[i];
    sign  = (x > 0) - (x < 0);
    x_abs = sign * x;
    q = (x_abs > -b) ? -b : x_abs;        // clamp |x| (b is a negative parameter)
    d = q + b;                            // d in [0, -b]
    L = sign * (-(d*d) + one);            // quadratic branch
    y = ((x * (one + L)) >> 1);
    Im_out[i] = clamp((int32_t)(y * totScaler) >> log2D, -128, 127);

Two-piece quadratic approximation of the tanh-form GELU, requantized with
`(y*totScaler) >> log2D`. One multiply/square per element, no LUT; parameters from BN-layer
calibration.

### 6.4 Integer LayerNorm sqrt complement

Integer transformers also need 1/sqrt(Var):

- I-ViT I-LayerNorm: Newton iteration `I_{i+1} = (I_i + Var/I_i) >> 1`, seeded by bit-scan
  `P = ceil(log2 Var)` (I-BERT integer sqrt), 3-4 int32 iterations.
- MCUFormer (Eq. 6) uses the same I-BERT-style integer sqrt in LayerNorm.
- PULP provides `pulp_layerNorm.c` (`pulp_nn_iLayerNorm`) and `pulp_layer_norm.c`: mean via
  multiply accumulator, variance in a second pass, normalize `(x-mu)*invstd` fixed point.

Generic pattern:

    mu  = (sum x) >> log2(N)
    var = (sum (x-mu)^2) >> log2(N)
    inv_std = isqrt( (1<<B) * (1<<B) / var )   // ~= 2^B / sqrt(var)
    y = ((x - mu) * inv_std) >> B


## 7. Open-source integer transformer / attention kernels (survey)

| Project | License | What it gives you | Key files / functions |
|---|---|---|---|
| **CMSIS-NN** (ARM-software/CMSIS-NN) | Apache-2.0 | int8/s16 GEMM, Q7/Q15 legacy FC/conv, integer softmax (Q7/Q15, s8/s16), tanh/sigmoid LUT activations; int64 accum for s16 | `arm_nn_vec_mat_mult_t_s8`, `arm_nn_vec_mat_mult_t_s16_s16`, `arm_nn_mat_mult_kernel_q7_q15`, `arm_fully_connected_mat_q7_vec_q15`, `arm_softmax_q7/q15`, `arm_nn_softmax_common_s8`, `arm_nn_exp_on_negative_values` |
| **pulp-platform/pulp-transformer** (TinyFormer, arXiv 2404.02945) | Apache-2.0 | i8 GEMM + BN fused, fused matmul-softmax, iSoftmax, i8 GELU, i8 LayerNorm; PULP SIMD `sdotsp4`; golden Python models | `pulp_nn_linear_i8_i8_i8`, `matmulSoftmax_4x2_S/H`, `iSoftmax`, `pulp_nn_gelu_i8_i8`, `pulp_nn_iLayerNorm`, `pulp_iSoftmax.py` |
| **MCUFormer** (arXiv 2310.16898) | paper/method | int8 ViT: Q8 GEMM, sigma(1.702x) GELU, integer sqrt LayerNorm (Eqs. 6-7) | paper Eq. 6 (LN) / Eq. 7 (GELU) |
| **I-ViT** (arXiv 2205.11256) | paper + integer kernels | ShiftExp/ShiftMax/ShiftGELU/I-LayerNorm shift-only recipes | Section 4.5 |
| **TFLite-Micro** (tensorflow/tflite-micro) | Apache-2.0 | reference int8/int16 softmax, FC, activations, GELU LUT; drops to CMSIS-NN on Cortex-M | `kernels/internal/reference/softmax.h`, `.../reference/fully_connected.h`, `.../reference/gelu.h` |
| **gemmlowp** (google/gemmlowp) | Apache-2.0 | quantized-GEMM/requant framework borrowed by CMSIS-NN and TFLite (`FixedPoint`, `exp_on_negative_values`) | `fixedpoint/`, `output_stages/` |
| **int-llm** (nmicic/int-llm) | custom/paper | pure-integer LLM in C, Q16.48 fixed point, bit-exact vs float oracle; TinyLlama 1.1B integer math lib | github.com/nmicic/int-llm, HF blog nmicic/int-llm |
| **EdgeNN** (Dimitrios-Kafetzis/EdgeNN) | MIT | zero-arena C11 ViT; FP32 reference attention + LayerNorm (structure reference, not integer) | `src/ops/transformer/edgenn_attention.c`, `edgenn_layernorm.c` |
| **ATTN-11** (toy PDP-11 attention) | - | historical extreme: Q8 fwd / Q15 bwd, Q16.16 weight acc, 256-entry Q8 exp LUT `exp(-i/32)`, `ASHC #-8` back to Q15 | cached notes `attn11.md` |


## 8. Copyable details for ESP32-C3

**PULP GEMM body** (`pulp_nn_linear_i8_i8_i8`, scalar fallback of the SIMD file):

    for (j .. dim_vec>>2) { sum = SumDotps4(vecA, vecB, sum); ... }   // 4x s8 dot into int32
    // tail
    sum += inA * inB;
    ...
    *pOutBuffer = pulp_nn_quant_i8(sum, out_mult, out_shift);   // or bias<<shift then BN quant

Bias is `int16_t`; per-channel `pKappa`/`pLambda` implement fused BN (scale/offset) on the
linear output. The SIMD 4x2 dispatch processes 2 output neurons per 4 input bytes.

**PULP fused attention-score kernel** (`matmulSoftmax_4x2_H.c`): int8 Q*K^T dots accumulate in
int32 `sum`, requantized to int8 scores before softmax:

    sum += *pA * *pB;   // ... 8 interleaved accumulators sum..sum8
    *softmax_buffer_1 = clip8((sum*requant_mul)>>requant_div);
    ...
    // then iSoftmax(buffer, ..., coeffA/B/C, log2, n_levels)

Attention scores stay int8 with `requant_mul/requant_div` folding the 1/sqrt(d) scale, then
the integer softmax of Section 4.4 runs on the int8 scores.

**int-llm** (Q16.48) is the most complete standing integer transformer math library in C for a
bit-exact oracle; its README validates the Q16.48 softmax/attention path at 32-bit MCU scale.

**gemmlowp `FixedPoint`** is the pattern to copy for the Q15xQ12 int64 accumulator:
templated `FixedPoint<int64_t, F>` with `Rescale`/`SaturatingRoundingMultiplyByPOT` and an
`exp_on_negative_values` equivalent to CMSIS-NN's.

**CMSIS-NN s16 path** shows int64 accumulation with fixed-point requant (`arm_nn_vec_mat_mult_t_s16_s16`):

    int64_t result = 0;
    ...
    result = arm_nn_requantize_s64(result, dst_multiplier, dst_shift);
    tmp = MAX(tmp, activation_min); tmp = MIN(tmp, activation_max);
    *dst++ = (int16_t)tmp;

This is the template for a Q15xQ12 int64 GEMM on a no-FPU MCU: accumulate q15*q12 products in
int64, then one `(acc*mul + round) >> shift` saturating requant.


## 9. Quantization error versus the 0.002 abs / 0.02 rel gate

### 9.1 Error sizes by bit width

Rounding to n fractional bits has step `2^-n`; uniform rounding error is bounded by
`2^-(n+1)` (max) and has SD `2^-n / sqrt(12)`:

| Quantization | n (frac bits) | step | max rounding | SD |
|---|---|---|---|---|
| Q15 activation | 15 | 3.05e-5 | 1.53e-5 | 8.81e-6 |
| Q12 weight | 12 | 2.44e-4 | 1.22e-4 | 7.05e-5 |
| Q7 weight | 7 | 7.81e-3 | 3.91e-3 | 2.25e-3 |
| Q15 output | 15 | 3.05e-5 | 1.53e-5 | 8.81e-6 |
| uint8 probability output | - (256 levels) | 3.92e-3 | 1.96e-3 | 1.13e-3 |

### 9.2 GEMM / dot-product error (before requant)

For a dot product of N terms with independent rounding, output error SD is
`sqrt(N) * sqrt(sd_a^2 + sd_w^2)` (weight term dominates for Q12):

- N=64:  SD ~= 5.7e-4   -> 3-sigma ~= 1.7e-3; absolute OK (under 0.002 at 3-sigma);
  relative OK at 1-sigma whenever the true |sum| >= ~0.03 (5.7e-4 / 0.02), and at
  3-sigma whenever |sum| >= ~0.085 (1.7e-3 / 0.02).
- N=256: SD ~= 1.1e-3   -> 3-sigma ~= 3.4e-3; 3-sigma exceeds 0.002 pre-softmax, but this is
  the logit level (pre-softmax), where the 0.002 gate is not usually applied. Relative OK for
  |sum| >= ~0.17 at 3-sigma.
- The Q7-weight alternative (SD 2.25e-3/term): N=64 SD ~= 1.8e-2 -> fails absolute gate
  unless combined with per-channel scales; Q7 is the coarse option.

Additional requant to the output format adds one more uniform rounding of the output step
(1.5e-5 for Q15), negligible vs the inputs.

### 9.3 Softmax / probability gate summary (from Section 4.6)

- Q15 probability output + >=7 logit fractional bits + exact (or good) exp passes both gates:
  max abs ~7e-4 (< 0.002), max rel ~1.7% (< 2%) on probabilities p >= 1e-3.
- uint8 probability output fails the 0.002 absolute gate (max abs ~2.4e-3) and the relative
  gate (rounds small-but-relevant probabilities disproportionally). Do not use 8-bit softmax
  probabilities if the 0.002/0.02 gate is hard.
- A 4-bit exp LUT is too coarse (abs ~4.0e-3); 5-bit is marginal; use the polynomial exp or
  an >=8-bit LUT.

### 9.4 Overall verdict

| Path | 0.002 abs gate | 0.02 rel gate | Notes |
|---|---|---|---|
| int8 x int8 GEMM + Q15 softmax (>=7 logit bits) | PASS | PASS | production MCU config |
| Q15 x Q12 GEMM (int64 acc) + Q15 softmax | PASS | PASS | use int64 acc + 64-bit requant |
| Q15 x Q12 GEMM (int32 acc) | FAIL (overflow, not just error) | FAIL | overflow after ~16 worst-case / ~256 RMS terms |
| Q15 x Q7 GEMM (int32 acc) + Q15 softmax | marginal | FAIL (weight error dominant) | CMSIS classic; coarse |
| uint8 softmax output (any GEMM) | FAIL | FAIL (small probs) | step 1/255 too coarse |

The binding constraints are: (1) accumulator width for Q15xQ12, and (2) output fractional
width + logit fractional width for the relative/absolute gate on softmax probabilities.


## 10. Concrete ESP32-C3 recipe (no FPU, RV32IMC)

1. **Weights/activations int8** (per-tensor or per-channel scale, like TFLite quantization),
   inputs/outputs int8. This maps directly to CMSIS-NN s8 kernels or hand-written RV32
   dot-product code (`mul` + `add` in int32).
2. **GEMM**: `acc = bias; for k: acc += a[k]*w[k]; out = clamp((acc*mul + (1<<(shift-1)))>>shift)`.
   int32 acc is safe (max term 2^14). For Q15xQ12 instead: `int64_t acc` and a 64-bit requant.
3. **Attention score**: `score8 = clamp((qk_acc * requant_mul) >> requant_div)` where
   `requant_mul/requant_div` encode 1/sqrt(d_head). (PULP `matmulSoftmax_4x2_H` does exactly
   this.)
4. **Causal mask**: iterate key columns 0..i only (no sentinel); keep running max over causal
   columns for the softmax max-subtraction.
5. **Softmax**: max-subtract; exp via `arm_nn_exp_on_negative_values` (polynomial, no LUT) or
   an 8-bit (256-entry) exp LUT; reciprocal via `arm_nn_one_over_one_plus_x_for_x_in_0_1` or
   `(1<<31)/sum`; output **Q15** (not uint8) to meet the 0.002/0.02 gate. Keep the pre-exp
   logit at >=7 fractional bits (octave units) — scale by `log2(e)` with a `mult + shift`.
6. **GELU**: either reuse softmax (`GELU = x*sigma(1.702x)`, MCUFormer/I-ViT) or a
   512/513-entry q15 tanh LUT with interpolation for
   `0.5x(1+tanh(sqrt(2/pi)(x+0.044715x^3)))` (TFLite approximate form).
7. **LayerNorm**: two-pass integer mean/variance (int32), I-BERT integer sqrt
   (`I_{i+1}=(I_i+Var/I_i)>>1`, seeded by `ceil(log2 Var)`), 3-4 iterations, then
   `y = ((x-mu)*invstd)>>B`.

Memory: int8 activations for an SxS attention matrix (S tokens) cost S^2 bytes; a 256-token
sequence needs 64 KiB just for scores — use streaming/rowwise softmax or a triangular KV-cache
scheme on an MCU.


## 11. Sources

Primary code (downloaded to /tmp/research/sources/):
- CMSIS-NN (ARM-software/CMSIS-NN, CMSIS_5): `arm_nn_mat_mult_kernel_q7_q15.c`,
  `arm_fully_connected_mat_q7_vec_q15.c`, `arm_softmax_q7.c`, `arm_softmax_q15.c`,
  `arm_softmax_s8_s16.c`, `arm_nn_softmax_common_s8.c`, `arm_nn_vec_mat_mult_t_s16_s16.c`,
  `arm_nn_vec_mat_mult_t_s16.c`, `arm_nnsupportfunctions.h` (MAX_COL_COUNT, EXP_ON_NEG,
  ONE_OVER1, MUL_SAT).
- pulp-platform/pulp-transformer (TinyFormer, arXiv 2404.02945): `pulp_nn_utils.h`,
  `pulp_nn_kernels.h`, `pulp_nn_linear_i8_i8_i8.c`, `pulp_matmulSoftmax_H.c`
  (matmulSoftmax_4x2_H), `pulp_iSoftmax.c`, `pulp_iSoftmax.py`, `pulp_gelu.c`,
  `pulp_layerNorm.c`, `pulp_layer_norm.c`, `pulp_softmax_arm.c`, `pulp_fc_q7_opt.c`.
- tensorflow/tflite-micro: `tensorflow/lite/kernels/internal/reference/softmax.h`,
  `.../reference/gelu.h`.
- gemmlowp (google/gemmlowp): fixed-point output-stage documentation.
- EdgeNN (Dimitrios-Kafetzis/EdgeNN): `edgenn_attention.c`, `edgenn_layernorm.c`.

Papers / docs (scraped to /tmp/research/.firecrawl/):
- TinyFormer: arXiv 2404.02945 (also the PULP repo above).
- MCUFormer: arXiv 2310.16898 — LayerNorm integer sqrt (Eq. 6), GELU = sigma(1.702x) (Eq. 7).
- I-ViT: arXiv 2205.11256 — ShiftExp/ShiftMax/ShiftGELU/I-LayerNorm.
- I-BERT: integer BERT math (quadratic i-exp; integer sqrt seed P=ceil(log2 x)).
- Softmax fixed-point evaluation: arXiv 2501.13379 — 64-sample quadratic interpolation LUT
  (16-bit fixed point) softmax exp RMSE 2.31e-7; 3rd-order Taylor RMSE 4.18e-5; 12-bit
  fixed-point LeNet-5 softmax <= 0.2% top-1 degradation.
- gemmlowp quantization doc; HF blog nmicic/int-llm (Q16.48 bit-exact integer LLM).
- blog.ando.ai attention tutorial (float reference; limited integer-causal detail).
- Springer embedded-ViT survey note.
- ATTN-11 PDP-11 transformer note (Q8/Q15/Q16.16, 256-entry exp LUT, ASHC).

All numeric overflow/error tables in this report are analytic or small simulations and are
labeled as such; exported code is quoted verbatim from the downloaded files.
