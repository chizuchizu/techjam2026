# Q15xQ12 Fixed-Point Transformer Forward Pass on ESP32-C3 — Research Report

Status: sources scraped/verified 2026, ESP32-C3 (160 MHz RV32IMC, NO FPU, 320 KB SRAM, 4 MB flash, Arduino env).
Target: B=1, S=128, D=128, H=4, HD=32, F=128, L=4, causal attention, correctness gate abs<=0.002 OR rel<=0.02 vs fp32 torch.

---

## 0. APPLY NEXT — prioritized action list (do in this order)

### Tier 1 — must do (biggest win/effort, removes the softmax+float bottleneck)
1. **Replace every fp32 float op with Q15/Q12 integer math; keep ALL tensors integer (Q15 activations).**
   Measured reason (Espressif, [2]): on ESP32-C3 every fp32 add = ~100 cycles, division = ~102, `cosf` = 2,377 cycles (vs 25/69/121 on the FPU-equipped ESP32-S3). A single per-token softmax (L=128, H=4) would burn ~512 `expf` calls => 0.8–1.6M cycles (5–30 ms at 160 MHz) IF exp were cheap; with soft-float it is much worse. Meanwhile QK^T for the same token is only ~16k MACs (~0.1–0.3 ms @ 1–2 cyc/MAC). Conclusion: softmax + layernorm are the bottleneck; integerizing them removes it. (See §2, §5, §7.)
2. **Implement integer softmax EXACTLY as TFLite's reference int16 softmax (§2.2).** 513-entry int16 `exp` LUT (domain [-10,0]) + 513-entry `1/(1+x)` LUT + `CountLeadingZeros`-based reciprocal. Zero divisions, zero transcendentals, ~4–6 int ops/element. This is the single most-debugged, most-portable integer softmax in existence (TFLite/TFLM use it in production).
3. **Implement GEMMs as Q15*Q15 -> int32 MAC with a single combined dequant scale (§5.1).** Per-GEMM/per-tensor symmetric (zero_point=0) Q15 for activations, Q15 (or Q12) static weights from flash. Inner loop: scalar `acc += (int32)a[i]*b[j]`, 4x-unrolled like CMSIS `arm_mat_mult_fast_q15` [18]; final `(acc*scale) >> shift` with rounding and saturation. Keep every intermediate in Q15 so softmax/ln inputs never pass through float.
4. **Use the TFLM-style LUT softmax + I-BERT-style integer LN/GELU (approx) for ALL nonlinearities (§2.3, §2.5).** Layernorm mean/variance and `1/sqrt(var)` via CLZ + LUT/polynomial refs; no division.

### Tier 2 — do next (memory + speed)
5. **KV cache MUST be 16-bit (Q15) or int8, not fp32.** Full S=128, D=128, L=4 KV = 4 x 2 x 128 x 128 elements. At 2 B/elem = 256 KB (fp32 would be 512 KB — does not fit 320 KB SRAM). Store KV as Q15 (or int8 = 128 KB) in SRAM; weights (only ~1 MB q15) live in flash. (§6)
6. **Causal efficiency: per-row online softmax; only compute the upper-triangle (S*(S+1)/2) MACs; avoid materializing the full SxS matrix (§6).** For decode step with prefilled KV, each new token touches exactly one new row: 4 heads x 128 keys x 32 dims.
7. **Resident-weight double-buffered row streaming from flash (ESP-LLM pattern [15], §4.2) + `PROGMEM`/aligned flash copies; process one weight row at a time** so GEMM never needs the whole 0.9 MB weight matrix in RAM.

### Tier 3 — precision/robustness
8. **Build the C reference on host first** with identical integer code, then diff against torch fp32 to the gate before flashing (§7). Verify LUT error < 1e-4 in exp domain and reciprocal error; confirm final abs error <= 0.002 / rel <= 0.02 per element across random seeds.
9. **Precompute all per-tensor scales on host, embed as integer multipliers+shifts** (MultiplyByQuantizedMultiplier, i.e., `QuantizeMultiplier` integer pair, [5][8]); avoid runtime float scale computation.
10. **Watch out for `exp` LUT range**: TFLM maps the full int16 diff to [-10,0] so exp(-10) ~ 0; replicate this range-clipping so nothing overflows Q16.15 sums. Add `diff_min`/clip as TFLM does. (§2.2, §7)

References [N] resolve in §8.

---

## 1. Platform facts (verified)
- ESP32-C3: single-core RV32IMC (no F, D, V, P), 4-stage in-order, up to 160 MHz, 32-bit multiplier + divider, 320 KB SRAM, 4 MB flash (datasheet v2.4 [3]). C-extension = 16-bit compressed instructions. NO FPU: `.float` ops compile to software calls.
- Peak arithmetic: 1 int32 MAC/cycle => ~160M MAC/s (160 MHz). There is no SIMD. Mantra: ALWAYS integer, minimize per-MAC overhead (unroll, 32-bit loads, avoid C-extension address arithmetic).
- Soft-float is catastrophic: measured by Espressif [2] (cycles/op, fp32): ADD 100, DIV 102, COS 2,377, MIX 3,659 (ESP32-S3 with FPU: 25 / 69 / 121 / 312). systemonchips [22] generalizes: soft-float is 10–100x slower than hardware FPU for div/sqrt.
- Instruction-set summary for optimization: RV32IMC only -> no `smlad`-type dual-MAC, no packed-SIMD, no P-extension; use plain `mul`+`adc`/`add` pairs and rely on C-extension for compact code.

---

## 2. Area A — Integer-only attention & softmax

### 2.1 The goal: keep attention 100% integer end-to-end
IntAttention (MLSys 2026, [1]) is the closest published pipeline to what we need, and its GitHub includes a de-facto reference implementation:
- Quantize Q/K/V to INT8 per-tensor symmetric (zero-point 0), do QK^T and PV in integer.
- After INT8 GEMMs, the **dequantize->softmax->requantize path is 57–65% of attention latency** (13–19% with FP32 GEMMs, 23–30% with FP16). In other words: once the GEMMs are integer, softmax is THE bottleneck.
- Fix: IndexSoftmax — 32-entry UINT8 exp LUT on the clipped range [0,c] after integer max-subtraction; direct integer normalization; direct requantization of the probability tensor. Result: 3.7x vs FP16 attention, 2.0x vs INT8-attention.
- Transfer to us: our logits are Q15 (from Q15*Q15 GEMM) not INT8, and we have H=4 HD=32 — but the *structure* (integer max-sub, small LUT exp, integer normalization, no float round-trip) is exactly plan (A)+(B).

### 2.2 The concrete blueprint: TFLite/TFLM reference int16 softmax (copy this)
TFLite reference `SoftmaxInt16` ([5]) and TFLM `softmax_common.cc` ([6]) implement the exact integer softmax we should port. Steps (all integer):
1. Find row max (int16 max).
2. Per element: `input_diff = x - max`. Scale so that `[-65535, 0]` maps to `[-10.0, 0.0]` using `MultiplyByQuantizedMultiplier(input_diff, input_multiplier, input_left_shift)` (integer mult+shift; no float).
3. `sym_scaled_diff = scaled_diff + 32767`, saturate to int16.
4. `exp = LUTLookup(sym_scaled_diff, exp_lut)` (LUTLookup below) -> Q0.15 exp, values in [0,32767].
5. `sum_of_exps` as int32 Q16.15.
6. Inverse without division: `headroom = CountLeadingZeros(sum)`; normalize; compute `x` and `1/(1+x)` via the second 513-entry LUT (`one_over_one_plus_x_lut`), index recentered to [-32768,32767].
7. Final: `result = (exp_value * reciprocal + round) >> right_shift`, clamp to [0,32767].

`LUTLookup` (reference, from TFLite common.h [5]) — 513-entry int16 LUT, 9-bit index + 7-bit linear interpolation:
```c
// index using high 9 bits, interpolate low 7 bits
uint16_t index = 256 + (value >> 7);   // 0..511 (lut[512] used only for slope)
int16_t offset = value & 0x7f;         // 0..127
int16_t base  = lut[index];
int16_t slope = lut[index+1] - lut[index];
int delta = (slope * offset + 64) >> 7;  // round to nearest
return base + (int16_t)delta;
```
Cheap: 1 shift, 1 and, 2 loads, 1 mul, 1 add + few ops. The exp LUT is generated on [-10, 0] at build time from `exp(x) * 32767` and the `1/(1+x)` LUT on [0,1]. This removes division AND transcendentals from softmax entirely. Because we only ever need per-row sums over L<=128 elements, Q16.15 accumulation cannot overflow.

### 2.3 Integer softmax everywhere else (LN / GELU / SiLU / RMSNorm)
- **I-BERT** (ICML 2021, [4]): integer-only BERT — softmax, GELU and LayerNorm replaced by low-order integer polynomials + iterative refinement (e.g. quadratic for GELU, `gelu(x) ~ 0.5x(1+tanh(...))` via integer tanh). Reported 2.4–4.0x speedup for INT8 inference. We need LN and GELU (or SiLU); use their recipes: mean+var with integer sums and `1/sqrt(var)` via CLZ+LUT/poly (gemmlowp `one_over_one_plus_x` / `sqrt` fixed-point helpers [7]).
- **I-ViT** (ICCV 2023, [9]): `Shiftmax` — exponential via **bit shifts + additions** instead of LUT (piecewise linear of 2^x). Note: uses QAT/calibration; on our tiny model this is optional, but the "2^x by shift" trick is the cheapest possible exp if our logit range is small.
- **I-LLM** (2024, [10]): `DI-ClippedSoftmax` — clipping + scaling done entirely in integer.
- **gemmlowp** `exp_on_negative_values` [7] is the math behind TFLM: polynomial on (-1/4,0] + barrel-shifter corrections (multiply by const exp(-1/4), exp(-1/2), ...) — the fully analytic alternative to the LUT if we ever want zero tables.
- Cascade into layernorm/RMSNorm: replace `(x-mean)/sqrt(var+eps)` with integer mean (sum>>k), integer variance from squares, reciprocal-sqrt via CLZ + `1/(1+x)` LUT [5][7], then `Q15 scale`.

### 2.4 Related hardware/quant-driven attention papers (citation + one-line)
- **TurboAttention** (MLSys 2025, [11]): LUT for integer part of exp + 3rd-order polynomial for fractional part; sparsification of negligible exponentials; ~90% of attention savings from softmax/quant; we adopt only the "small LUT + low-order poly" idea in fixed point.
- **Softermax** (DAC 2021, [12]): replace e^x with 2^x, integer shifters, online normalization; 2.35x energy efficiency @0.90x area on accelerator; conceptual support for shift-based exp.
- **ConSmax** (ICCAD 2024, [13]): fixed scaling constants remove max-find and normalization; table lookups + multiplies only.
- **EXAQ** (NeurIPS 2024 Wksp, [14]): dynamic optimal clipping ranges for attention scores, down to 3 bits — relevant if we later compress attention scores to int8.
- **softmax_eval** (arXiv 2501.13379, [21]): empirical open-source survey of fast softmax approximations — LUT with (quadratic) interpolation has lowest error; Taylor/Pade fastest — justification for our LUT+linear (or later quadratic) interpolation choice and quantitative error-risk discussion.
- **FlashAttention(-3)** [25], **SageAttention/SageAttention2** [26]: online softmax + integer attention kernels on GPUs; concept (row tiling, fusing) transfers to our per-row causal processing (see §6).

---

## 3. Area B — Fast exp / softmax approximations (options, ranked for us)
Ranked for RV32IMC no-FPU, with our >= 0.98-relative-accuracy gate:
1. **16-bit LUT + linear interpolation (TFLM style)** [5][6] — ~1e-4 accuracy in-domain, ~10 int ops/elem. Winner. Use two LUTs: exp on [-10,0], 1/(1+x) on [0,1]; clip inputs; index via `value>>7`.
2. **Small UINT8 LUT + integer normalization (IndexSoftmax)** [1] — fewer bits => smaller tables; on our Q15 logits a 256-entry exp2/exp table with 8-bit interpolation is the natural second step. 2.0x vs INT8 attention, 3.7x vs FP16 attention reported.
3. **Shift-based 2^x (Shiftmax/Softermax)** [9][12] — cheapest; risk: logit-scale sensitivity; keep as fallback if LUTs too big.
4. **Polynomial exp (I-BERT / gemmlowp)** [4][7] — zero tables; ~10-20 cycles; slightly more error control work; good fallback for LN (which takes continuous inputs, not clipped-range).
5. **Rejected: Schraudolph bit-trick (fastexp) [20]** — ~6% worst-case relative error, and it requires interpreting FP32 bits, which reintroduces the float path; Turbo-Softmax [23] is float-heavy poly; both fail our gate or defeat the point.
6. **Reciprocal without division** — always use CLZ+MultiplyShift (`1/sum` via `1/(1+x)` LUT) as TFLM does [5][7]; ESP32-C3 has a divider but it's ~100 cycles/op vs ~10 for CLZ+shift.

---

## 4. Area C — Transformer / LLM inference on MCUs (precedents & patterns)

### 4.1 ESP-LLM (bare-metal C++, ESP32 + ESP8266) [15] — closest match to our size
- Config almost identical to ours: D=128, H=4, HD=32, MLP_hidden=128, layers 6/4, **MQA (1 KV head)**.
- BitNet b1.58 ternary weights + INT8 activations; **per-token absmax dynamic quantization**; int32 accumulator; **single combined scale = scale_w * scale_x** applied once at the end; RoPE tables precomputed in flash; **PROGMEM/4-byte-aligned flash weights; zero heap allocation** static arena (88 KB); ~81 KB active inference SRAM; ~3 MB binary in flash.
- Directly validates: per-GEMM quant + combined scale (A, D), flash-resident weights (E), static buffers only.
- Adopt: static-arena design and single-read streaming of weight rows.

### 4.2 ruvllm-esp32 (Rust crate, ESP32) [16]
- INT8/INT4 quantization, "pre-computed tables for softmax/exp ... critical for ESP32 which lacks an FPU", libm only for the few FP cases; docs page explicitly justifies LUT softmax on no-FPU cores. Good supporting citation for our design decision.

### 4.3 NanoMind-S3 (ESP32-S3, 8 MB PSRAM / 16 MB flash class) [17] — upper-bound comparison only
- INT4 packing (2 weights/byte), 1-cycle nibble sign-extension unpacking, dual-core matmul, single-pool tokenizer (~150 KB vs ~1 MB BPE), flash MMU + PSRAM KV. Transfers conceptually (packing, tokenizer), but the S3 has 2 cores + PSRAM + FPU-free-but-RVV... treat as "what the bigger part can do", not our budget.

### 4.4 Fixed-point math library
- **MFixedPoint** [19]: header-only C++ fixed-point library for embedded (no FPU) — reference for Q-format helpers and rounding (round-half-away, saturation) that mimic our Q15/Q12 handling.

---

## 5. Area D — GEMM optimization on RV32 (no SIMD) / Cortex-M precedent

### 5.1 Integer GEMM structure
- Format: A in Q15 (activations, per-tensor), W in Q15 (weights, static). Each `acc += (int32)a[i] * w[j]` stays in int32; max |a|,|w|=32767 => product 2^30, sum over 128 terms ~2^37 — **overflows int32**! Two fixes used in production: (a) accumulate in **int64** (CMSIS `arm_mat_mult_q15` uses q63 for full precision [18]) or (b) **scale logits/activation to Q12** (then product ~2^24, sum over 128 ~2^31 still risky). Recommendation: use **int64 accumulation in C** (RV32 has no native 64-bit regs; compiler emits mul+add high-word pairs ~3-4 instr per MAC) OR **Q15*Q12 with int32 acc and client-side periodic rescale**. Best: follow CMSIS: wide accumulator + `sum >> 15` + saturate; if int32 acc desired, keep one operand at Q12 and accept clip guard.
- Scale handling: end-to-end integer: output = `(acc * scale) >> shift` with `QuantizeMultiplier`-style (multiplier, right-shift) pair computed host-side [8,5]. Combined scale: one per GEMM (per IntAttention and ESP-LLM [1][15]).

### 5.2 Loop-level techniques from CMSIS-DSP `arm_mat_mult_q15` / `_fast_q15` [18]
- Reference inner loop (Cortex-M0 recipe — our RV32 analog):
  ```c
  while (colCnt--) { sum += (q31_t)*pInA++ * *pInB++; }   // i.e. 64-bit acc, MMA
  *px++ = (q15_t)(sum >> 15);   /* saturate store */
  ```
- Fast variant `arm_mat_mult_fast_q15`: **4-way unrolled, 4 accumulators, 32-bit** — exactly the pattern to hand-write on RV32IMC:
  ```c
  sum += inA1*inB1; sum += inA2*inB2; sum += inA3*inB3; sum += inA4*inB4; // per acc
  ```
  with 4 independent accumulators (ILP hides `mul` latency), odd tail handled separately.
- On Cortex-M the `__SMLAD` dual-MAC exists; RV32IMC has none — that is precisely why unrolling + multiple accumulators (to hide the single `mul` pipe) is the main scalar optimization. Also prefer **32-bit (or 16-bit paired) loads**; C-extension means fewer cycles per packed 16-bit op but at the cost of extra address math — unroll to amortize.

### 5.3 Data layout / memory
- 320 KB SRAM cannot hold weight matrix ~0.9 MB. Stream rows from flash (PROGMEM) — ESP-LLM pattern [15]; 4 MB flash is ample. On C3, leverage 4-way set-associative 4KB direct data cache? (C3 has a 16KB? — verify on-chip: the RV32 core has 4KB/4KB? Actually ESP32-C3 has 400 KB SRAM total incl. RTC; *user* 320 KB with cache for flash. Keep weights flash-mapped and access streaming-friendly; see datasheet [3] and ESP-IDF memory layout.)
- KV cache resident in SRAM (see §6) — with Q15 KV = ~256 KB worst-case; consider int8 KV (128 KB) if needed.

---

## 6. Area E — Causal attention & KV-cache optimizations
- Causal mask => only compute upper triangle: for the sequence-token (prefill) each step n costs n x HD per head, not S x HD. Total MACs ~ H * HD * S*(S+1)/2 = 4*32*8256 = 1.06M (vs 2.1M full) for QK^T; PV has the same shape with one output per query (all-all V, but only S output rows each 128-element dot -> 4 heads * 128*128*32 = 2.1M).
- **Online/per-row softmax** with streaming KV, FlashAttention-style [25]: maintain per-row running max and running sum of exp (`streaming softmax`), so a row only ever holds L exp values buffer (128*2B = 256 B). One exp LUT + one reciprocal per row.
- KV memory (critical): L layers x (K+V) x S x D x bytes. With Q15 (2B): 4 * 2 * 128 * 128 * 2 = 256 KB. fp32 = 512 KB (does NOT fit). → Use 16-bit KV (design choice) or int8 KV (128 KB) to leave SRAM headroom. MQA-style single KV head (ESP-LLM) would cut KV to 64 KB if we are allowed to re-architect (flag as optional).
- Prefill vs decode: run full causal forward once for the "prompt" (1.06M MACs) then per-token decode appends one new Q row; all new-token work is O(S) not O(S^2) if KV is cached. On a 160 MHz core with ~1 MAC/cycle, QK^T+PV ~ 3-5M cycles ~ 20-30 ms for full prefill and ~0.2-0.4 ms per decode token (estimate, integer MAC).

---

## 7. Correctness gate & precision plan (abs<=0.002 OR rel<=0.02 vs fp32 torch)
- Port TFLM-LUT softmax and Q15 GEMM verbatim to C on host; run same random inputs through torch fp32 and our int path; assert gate elementwise, across seeds and sequence lengths.
- LUT domain/clip mirrors TFLM: input_diff scaled to [-10,0]; exp(-10) ~ 4.5e-5 < 0.0005 fraction -> below our gate's rel 0.02 for attention weights (and far below for output carry-through if output LN normalizes).
- Error budget notes (softmax_eval [21]): LUT+linear interpolation has smallest error among table/approx families; quadratic interpolation even better if needed later. Sample density matters: our int16 diff covers [-65535,0] with 513 points -> step ~2/65535 in exp-domain argument. For Q15 logits with typical attention-score spread < 10, in-domain accuracy is excellent.
- Watch int32 accumulation range (see §5.1): if using int32 acc with Q15*Q15, collapse at K=16 (rescale), or use int64 acc (slow on RV32 but correct), or fold scales so partial sums stay bounded; then compare to gate.
- `QuantizeMultiplier` (multiplier+shift) computed in double on host, embedded as int32 to avoid runtime float. All scales precomputed; zero runtime float in the whole forward.

---

## 8. Measured vs estimated numbers table
| Item | Value | Source |
|---|---|---|
| ESP32-C3 fp32 add | 100 cyc | [2] measured |
| ESP32-C3 fp32 div | 102 cyc | [2] measured |
| ESP32-C3 cosf | 2,377 cyc | [2] measured |
| ESP32-S3 (FPU) add/div/cos | 25 / 69 / 121 | [2] measured |
| soft-float vs HW FPU div/sqrt gap | 10-100x | [22] |
| softmax path share (after INT8 GEMM) | 57-65% | [1] measured |
| IntAttention vs FP16 attention | 3.7x | [1] |
| IntAttention vs INT8 attention | 2.0x | [1] |
| I-BERT INT8 inference speedup | 2.4-4.0x | [4] |
| Per-token QK^T+PV (est., int) | ~0.2-0.4 ms | estimate (this report) |
| Per-token softmax float (est.) | 5-30 ms | estimate from [2] |
| Full prefill cost (est., int) | ~20-30 ms | estimate (this report) |

---

## 9. References (URLs)
1. IntAttention — arXiv 2511.21513 — https://arxiv.org/abs/2511.21513 ; repo https://github.com/WanliZhong/IntAttention
2. Espressif Dev Blog, "Floating-Point Units on Espressif SoCs" — https://developer.espressif.com/blog/2025/10/cores_with_fpu/
3. ESP32-C3 Datasheet v2.4 (RV32IMC, 160 MHz, 320 KB SRAM, 4 MB flash) — https://www.espressif.com/sites/default/files/documentation/esp32-c3_datasheet_en.pdf
4. I-BERT — arXiv 2101.01321 — https://arxiv.org/abs/2101.01321
5. TFLite reference softmax (int16 LUT) — https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/internal/reference/softmax.h ; LUTLookup in .../kernels/internal/common.h
6. TFLite Micro softmax_common.cc — https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/kernels/softmax_common.cc
7. gemmlowp fixedpoint.h — https://github.com/google/gemmlowp/
8. Jacob et al., "Quantization and Training of NN for Efficient Integer-Arithmetic-Only Inference" — arXiv 1712.05877 — https://arxiv.org/abs/1712.05877
9. I-ViT (Shiftmax) — arXiv 2207.01405 — https://arxiv.org/abs/2207.01405
10. I-LLM (DI-ClippedSoftmax) — arXiv 2405.17849 — https://arxiv.org/abs/2405.17849
11. TurboAttention — arXiv 2412.08585 — https://arxiv.org/abs/2412.08585
12. Softermax — arXiv 2103.09301 — https://arxiv.org/abs/2103.09301
13. ConSmax — arXiv 2402.10930 — https://arxiv.org/abs/2402.10930
14. EXAQ — NeurIPS 2024 ML&Compression workshop — arXiv 2410.03185 — https://arxiv.org/abs/2410.03185
15. ESP-LLM — https://github.com/ahmedbarakat207/espllm
16. ruvllm-esp32 — https://docs.rs/ruvllm-esp32/latest/ruvllm_esp32/ ; https://github.com/ruvnet/ruvector
17. NanoMind-S3 — https://github.com/imFARSI/NanoMind-S3
18. CMSIS-DSP matrix functions (arm_mat_mult_q15 / arm_mat_mult_fast_q15) — https://github.com/ARM-software/CMSIS-DSP
19. MFixedPoint — https://github.com/gbmhunter/MFixedPoint
20. fastexp (Schraudolph trick) — https://github.com/psherman42/fastexp ; N.N. Schraudolph, Neural Computation 11(4), 1999
21. "Softmax approximations evaluation" — arXiv 2501.13379 — https://arxiv.org/abs/2501.13379
22. System on Chips — soft-float vs HW FPU tradeoffs — https://www.systemonchips.com/soft-float-vs-hardware-floating-point-tradeoffs-on-microcontrollers/
23. Turbo-Softmax (float poly exp; rejected for no-FPU) — https://github.com/LongWeihan/Turbo-Softmax
24. AttentionLego — fixed-point softmax LUT hardware (256x32-bit exp LUT, 32-bit acc, normalize) — https://deepwiki.com/bonanyan/attentionlego
25. FlashAttention-3 — arXiv 2407.08608 — https://arxiv.org/abs/2407.08608
26. SageAttention2 — arXiv 2411.10958 — https://arxiv.org/abs/2411.10958

Notes: the two verified anchors for "softmax is the bottleneck after integer GEMMs" are [1] (57-65% share) and [2] (cycle costs on the same chip).
