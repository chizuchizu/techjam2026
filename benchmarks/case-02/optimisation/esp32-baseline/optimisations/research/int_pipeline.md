# Integer-only pipeline — killing the remaining soft-FP32 passes

Date: 2026-08-30 · Research survey (web + primary sources) for removing the residual
FP32 work left in the FAST build: **residual adds, LayerNorm re-quantize pass,
final norm, attention-context write**, i.e. most of the ~0.5 s/forward "LayerNorm"
block in the 4.02 s total.

Gate to protect: per-element `|Δ| ≤ 2.0e-3 OR |Δ| ≤ 0.02·|ref|` vs fp32 torch
reference. Q15 act + Q12 weight pipeline already holds worst-case ≈ 9.5e-4.

Reference frame: `src/kernels.c` — `tm_bn_q15_int` (integer stats but fp32
amax-scan + `sqrtf` + per-call fp32 Fk/Bk), `tm_add_inplace` (fp32 residual add),
`tm_layernorm` (fp32 final norm), attention ctx store via fp32 multiply + fcvt.

---

## 1. What is still FP32, and what it costs

Per forward the following touch soft-float library calls on RV32IMC (no FPU):

| site | op | ~cost |
|---|---|---|
| residual add before each LN (4×) | `tm_add_inplace` fp32 add + fp32 store at one of the two scales | ~0.1 s |
| LN input path (4×) | amax scan of fp32 + per-element fp32 multiply + fcvt→Q15 (16 384 elems) | part of 0.5 s |
| LN per-row | fp32 `sqrtf` + fp32 rstd accumulation + per-call fp32 Fk/Bk (128 rows + 128 ch) | small |
| final norm | fp32 `tm_layernorm` + fp32→Q15 quantize | ~0.09 s |
| attention ctx | `oq = fp32_mul(int64 scalar)` + fcvt + saturate per element (65 536) | ~0.1 s |

Everything else (GEMMs Q15×Q12 int32, attention QK/PV int64, exp/gelu LUT) is
already integer. The goal of all four research threads is to keep **activations
as int16 (Q15) end-to-end**, converting at block boundaries with integer
multiply–shift (dyadic) instead of through fp32.

---

## 2. (a) Integer-only LayerNorm — exact integer algorithms

### I-BERT (arXiv 2101.01321) — the reference recipe
All non-linear ops run in INT32 with *no accuracy loss*:
- MatMul/Embedding INT8; **LayerNorm/GELU/Softmax computed in INT32** on the
  already-accumulated value, then one Requantization brings INT32 → INT8.
- LN mean/var are computed straight from the quantized values with integer
  arithmetic (scaled numerators), and the inverse standard deviation uses an
  **integer square root by Newton iteration** on the INT32 variance; the
  normalization is an integer division of `/ (1/σ)` form.
- Reported GLUE: RoBERTa-Base 86.0 → **86.3**, RoBERTa-Large 89.0 → **89.5**
  (INT8 fully integer, i.e. *equal or higher* than FP32). 2.4–4.0× speedup on T4.
- Design point directly relevant to us: keep the LN at **high integer precision**
  (they chose INT32; we already use int32 stats / int64 accumulators → even

 *safer*) and bring the *output* down with one requantize.

### I-ViT (arXiv 2207.01405): I-LayerNorm — the compact form
Paper §3.3 (dyadic pipeline, everything `b/2^c`):
- mean `m = round(Σ x_i / N)` in integer;
- variance `v = round(Σ x_i² / N − m²)` in integer;
- **I-RSQRT** by fixed-iteration Newton on the integer variance:
  `s_{i+1} = (s_i + (v >> f) / s_i) >> 1`, 10 iterations (on 8-bit inputs this is
  branch-free and fast); the ± mean-shift reuses the same integers.
- Normalization `x̄_i = (x_i − m)·s` stays in integer; `γ,β` folded as integer
  per-channel factors.
- I-ViT accuracy (INT8, ImageNet, fully integer): ViT-S 81.39 → **81.27**,
  ViT-B 84.53 → **84.76**, DeiT-T 72.21 → **72.24**, Swin-T 81.35 → **81.50**;
  3.72–4.11× speedup via TVM integer units.

### Scale-side fixes for the LN input distribution (why LN is hard)
- **FQ-ViT (arXiv 2111.13824)**: PTF — *Power-of-Two Factor* — the hard part of
  quantizing LN is inter-channel variation of its input; absorb it with a
  per-channel power-of-two factor so the LN input keeps one per-tensor scale.
  LIS (Log-Int-Softmax): replace exp/sum by log-2 → `BITSHIFT` (budget: 4-bit
  attention maps). Lossless ~1% degradation first time for fully-quantized ViT.
- **RepQ-ViT (ICCV 2023)**: *scale reparameterization* — decouple the
  quantization scale from inference by reparameterizing the post-LayerNorm /
  post-Softmax extremes so per-tensor quantization works without a per-channel
  pass. (Both relevant if we ever hit amax/scale mismatches on non-seed inputs.)

### Concrete form for THIS code (Q15 input already at a known scale, int64 stats)
With the residual already an int16 Q15 buffer at per-layer scale `sx` (thread b),
`tm_bn_q15_int` needs **zero fp32**:
- stats from the int16 values directly: `s1 = Σ q_i` (int32), `s2 = Σ q_i²`
  (int64); per-row `mq = s1/D` (exact, D=128), `vq = (s2/D) − mq·mq` (int64);
- rstd: integer RSQRT on `vq>>15` by 4–10 Newton iterations (D=128 → ≤16 hints)
  instead of `sqrtf`;
- per-row `rs = rstd·2^15` (Q15), then per element
  `o_i = reqround(((q_i − mq)·rs) >> 15),  d=o`; then per-channel
  `y = sat((o·Fk + Bk) >> 15)` with **precomputed static `Fk,Bk,sa`** (see c).
- Because LN output is bounded (amax-bound fusion), the LN requantize scale `sa`
  is per-layer constant → `Fk/Bk` computed once at init, not per forward.
- Only two runtime integers per layer remain (rstd and amax-if-dynamic), both
  from int16/int64 data. Cost: pure 16/32-bit ALU + shifts — no soft-float.

---

## 3. (b) Residual re-scaling / block-float — one Q15 scale end-to-end

### Published: I-ViT `fixedpoint_mul` — residual add as dyadic requantize+add
Verified in `models/quantization_utils/quant_utils.py` (Block.forward →
`qact(x, act_scale, x_1, act_scale_1)`). The residual add is:
```
z   = round(x / sx);        new_scale = sx / sz;  [m,e] = frexp(new_scale)
out = round(z * m / 2^e)                      # block output rescale via b/2^c
z1  = round(x1 / sx1);      [m1,e1] = frexp(sx1 / sz)
out1= round(z1 * m1 / 2^e1)                   # residual rescale via b/2^c
y   = sat(out1 + out)                          # integer add at common scale sz
```
Both arms are rescaled to one output scale via a **dyadic multiply-and-shift**
(the scale ratio `b/2^c` is precomputable) and added in integer. This is exactly
the parent's "act-prescale": `sz` is known in advance from calibration.

### I-Segmenter (arXiv 2509.10334) — INT16 residual buffers
§3: "residual connections are computed by adding two INT16 tensors with an
INT32 accumulator, and the outcome is then requantized to INT16. Maintaining
residuals in INT16 within Transformer block preserves precision and ensures
consistent scaling at each skip connection." Shiftmax also in INT16. Claims
fully-integer ViT segmentation within 5.1 % mIoU of FP32; code
`github.com/ms245755/I-segmenter`.

### I-LW-DETR (arXiv 2607.24981) — common-scale dyadic for multi-branch
§3.2: each feature-fusion branch is rescaled to the max of the calibrated per-branch
scales; "the scale ratios are fixed after calibration, the rescaling reduces to a
constant affine transformation implemented as an integer dyadic multiply-and-shift".
(Our residual branch + block branch is the same multi-scale add pattern; their
combined QAT/PTQ numbers: 8/8/16 model size 3.6× smaller, BOPs >10× lower, PTQ
brings COCO mAP 48.0→41.6 on LW-DETR-Small, QAT recovers to 43.7.)

### Underlying concept — block floating point (per-buffer exponent)
Classic BFP / dynamic fixed point: **shared exponent per block + fixed-width
mantissa** (IBM DFP; Flexpoint). Here the natural block is "one residual /
one GEMM output buffer": mantissa always int16 Q15, per-buffer exponent chosen
so Q15 stays in range. Residual drift across layers is absorbed by the per-layer
(exponent) scale, not by widening the mantissa.

### Concretely for this code
- Store each residual buffer as **int16 Q15 with per-layer scale `sx_l`**
  (16 KB in SRAM vs fp32 anyway).
- Choose every GEMM's output requantize scale **ahead of the GEMM** (act-prescale)
  so that GEMM output lands directly at the residual scale `sx_l` → the block
  branch and residual branch come in at the **same scale** and
  `tm_add_inplace` degenerates to a **pure int16 saturating add** (no fp32, no
  per-element rescale). If a scale mismatch is ever unavoidable, do the I-ViT
  dyadic `round(z·b) >> c` on whichever side, with `b,c` precomputed per layer.
- amax: most of the scan disappears because a amax-bound scale is static per
  layer (see c); when we keep runtime adaptivity, scan the **int16** buffer
  (pure ALU abs+max) instead of fp32, ~1.2 ms for all layers — not a soft-float.

---

## 4. (c) Fused LN + residual + precomputed per-layer scale

- **amax-bound normalization (already in `tm_bn_q15_int`)** is the key trick and
  must be **lifted to calibration time**: because LN output is `(x−μ)·σ·γ + β`
  with `σ,γ,β` bounded, the output amax is bounded a priori → `sa` per layer is a
  **constant** → `Fk,Bk` and the following GEMM's requantize factor `b/2^c` are
  **precomputed at init** (stored in flash), so the LN hot loop has no fp32 and
  the GEMM epilogue is a fixed integer multiply-shift. `Fk/Bk` in
  `tm_bn_q15_int` are currently recomputed every forward with fp32 — that work
  is pure waste under a static scale.
- **Fold LN into the consuming GEMM (published pattern, ONNX Runtime "LayerNorm
  fusion" / I-BERT-style)**: `Y = LN(X; μ,σ,γ,β)·W + b` →
  `Y = σ·( (X·(γ⊙W)) − μ·(Σ_j γ_j W_jk) ) + (βW + b)`.
  Pre-store per layer (flash cost ≈ +1 weight set for the matrices right after
  LN: q,k,v,f1; the other GEMMs are unnecessary): `W' = γ⊙W` re-quantized
  per-channel and `b'_k = (β·W)_k + b_k` in Q12; at runtime compute `GX = GEMM(X,
  W')` (same cost, int), then a per-token affine on the GEMM output:
  `y_k = ((GX_k · rs) − (μ·rs)·G1_k + b'_k)` with `G1` the folded column sums.
  LN then reduces to **stats only** (S rows) — the normalize+requantize buffer
  pass over 16 384 elements per layer disappears entirely. Cheapest when done
  only for the two biggest consumers (f1 after norm2, qkv after norm1) and final
  norm; verify flash (currently ~83.6 % of the app partition — folded weights as
  Q12 add ≈ +0.5–1 MB, see §6 risk).
- Fused LN + residual: once the residual is int16 at scale `sx_l` (b), the add
  is `sat(res + layer_out)` in one pass and the **same pass** can accumulate the
  LN stats (`s1,s2`), turning 3 passes (add, stats, normalize) into ~2, with the
  normalize either in its own pass (b) or folded into the next GEMM (above).

---

## 5. (d) Published integer transformers with integer residuals — evidence

| System | arch/precision | residuals | reported accuracy | fully int |
|---|---|---|---|---|
| I-BERT 2101.01321 | RoBERTa INT8, LN/GELU/SM INT32 | requantized in INT32 | GLUE 86.0→86.3 (B), 89.0→89.5 (L) | yes |
| I-ViT 2207.01405 | ViT/DeiT/Swin INT8 | `fixedpoint_mul` dyadic add (INT32 acc, back to INT8) | 81.39→81.27 (ViT-S), 84.53→84.76 (ViT-B), 72.21→72.24 (DeiT-T), 81.35→81.50 (Swin-T) | yes |
| FQ-ViT 2111.13824 | ViT + PTF/LIS | INT (BitShift) | 84.89 ViT-L; ~1% lossless | yes |
| RepQ-ViT ICCV23 | ViT scale reparam | INT | SOTA PTQ image clf | yes |
| I-Segmenter 2509.10334 | Segmenter INT; **INT16 residuals, INT32 acc** | `res1+res2` at per-block scale ⇒ requantize INT16 | within 5.1 % mIoU (avg), one-shot PTQ ok | yes |
| I-LW-DETR 2607.24981 | LW-DETR 8/8/16 | common-scale dyadic per branch | COCO mAP −6.4 (PTQ) / −4.3 (QAT) on Small | yes |
| EIQ-DETR (Entropy 2025) | Swin-T DETR | integer residual connections | fully int detection | yes |
| TinyFormer 2311.01759 | sparse ViT on MCUs (1 MB flash / 320 KB SRAM) | sparse engine | 96.1 % CIFAR-10, ≤320 KB SRAM, 12.2× vs CMSIS-NN | yes (int8 kernel) |

Consistent pattern: every fully-integer transformer keeps **activations in high
precision (INT16/INT32) at the residual+LN boundary**, chooses **one scale per
block from calibration**, and implements all re-scaling as **dyadic
multiply-and-shift**. Our gate (Q15 / ≈2.4e-4 minimum resolution) is far finer
than ANY of these published INT8/INT16 pipelines, so the per-element error budget
is not the binding constraint — the binding constraint is *float-library calls*,
which the above removes.

---

## 6. Application plan (ranked by gain ÷ effort, mapped to current code)

### Tier 1 — biggest wins, lowest risk
1. **int16 Q15 residual buffer + act-prescaled GEMM epilogue (dyadic requant)**
   `tm_add_inplace` → `int16 sat-add`; all 4 GEMMs requantize `int32 acc → int16
   Q15@residual-scale` with a precomputed `b/2^c` inside the epilogue (no fp32
   store, no re-read, no fp32 add of the block branch). Kills ~0.25–0.35 s
   (most of the residual adds + the fp32 requant path of the LN input). Matches I-ViT/I-Segmenter.
2. **Integer-only LN with int64 stats + integer RSQRT (Newton) + static
   `Fk/Bk/sa` from amax-bound calibration** → `tm_bn_q15_int` becomes pure ALU;
   no amax-scan of fp32, no `sqrtf`, no per-call fp32 Fk/Bk. Kills ~0.15–0.2 s.
   (Optional: keep a cheap int16 amax scan for runtime robustness.)
3. **Integer final norm** — reuse tier-2 LN on the (now int16) residual, output
   Q15/fp32; kills the ~0.09 s final fp32 norm.

### Tier 2 — good, medium effort
4. **int64 × Q16 fixed-point attention-ctx requant** (`oq`): replace
   fp32-scalar-mul+fcvt with `(acc·rot_q16 + 2^15) >> 16` clip, one Q16 per row
   (rot = inv·scale). Kills ~0.08–0.1 s; no accuracy impact.
5. **Fold LN into next GEMM (`W'=γ⊙W`, `b'=βW+b`)** for norm2→f1 and norm1→qkv
   (and final norm→classifier if applicable). Removes the 16 384-element
   normalize+requantize pass per layer. Watch flash: folded Q12 weights ≈
   +0.5–1 MB (evaluate per-app-partition budget first).

### Tier 3 — keep on the shelf
6. **Runtime per-buffer block-float** (recompute per-layer exponent from the
   int16 buffer) only if static calibration fails outside seed-0.
7. **LIS 4-bit softmax** — swap LUT softmax for log-2 + BitShift if attention
   ever becomes the bottleneck again (small accuracy trade).
8. **RepQ-ViT / PTF reparameterization** — only if a scale mismatch on the LN
   input distribution appears in validation.

### Expected result
Removing tier 1+2 (≈0.5–0.65 s of the 0.5 s LayerNorm block + final norm) plus
attention-ctx (0.08–0.1 s) should take the forward from **4.02 s → ≈3.3–3.5 s**
(~1.15–1.2×) with integer-only residuals, and removes essentially every
soft-float call except the (optional) fp32 output stage — matching the published
"fully integer transformer" pattern with Q15 detail margin ≈ 4–8× finer than the
published INT8/INT16 pipelines.

---

## 7. Sources

- I-BERT: arXiv:2101.01321 (integer LN: mean/var INT32, Newton int sqrt §3.5,
  impl. details App. C.1) — https://arxiv.org/abs/2101.01321
- I-ViT: arXiv:2207.01405 (I-LayerNorm, I-RSQRT, `fixedpoint_mul` dyadic residual
  add) — https://arxiv.org/abs/2207.01405 · code zkkli/I-ViT (quant_utils.py)
- FQ-ViT: arXiv:2111.13824 (PTF, LIS) — https://arxiv.org/abs/2111.13824
- RepQ-ViT: ICCV 2023, github zkkli/RepQ-ViT (scale reparameterization)
- I-Segmenter: arXiv:2509.10334 (INT16 residuals, INT32 acc, λ-ShiftGELU)
  — https://arxiv.org/abs/2509.10334 · code ms245755/I-segmenter
- I-LW-DETR: arXiv:2607.24981 (common-scale dyadic, ShiftGELU, Shiftmax,
  scale-preserving split) — https://arxiv.org/abs/2607.24981
- TinyFormer: arXiv:2311.01759 (sparse transformer on MCUs, 320 KB SRAM)
  — https://arxiv.org/abs/2311.01759
