# ESP32-C3 Fastest-Kernel Research Brief (C, no floating-point hardware)

**Target:** Seeed Studio XIAO ESP32-C3 (160 MHz, single-core RV32IMC, 400 KB SRAM, 4 MB flash)
**Project context:** TikTok Tech Jam — port the torch transformer + benchmark harness to ESP32s in C.

---

## 0. THE critical hardware fact (verified, changes everything)

**The ESP32-C3 has NO hardware floating-point unit.** FP32 math is *software-emulated*.

Evidence (all primary/authoritative):
1. **ESP32-C3 Technical Reference Manual v1.4**, `misa` CSR: F (single-precision FP extension) = 0 (hardwired RO), D = 0, Q = 0; only I, M, C set. Machine-mode `mstatus` has **no FS field** — a standard RISC-V FPU would require one.
2. **ESP32-C3 datasheet (Espressif, official PDF)**: CPU = "RV32IMC ISA", no F, no FPU bullet.
3. **Espressif Developer Portal blog, 13 Oct 2025, "Floating-Point Units on Espressif SoCs"** (authors: A. Spagnolo + F. Bez, Espressif): *"On others, like the ESP32-C3, floating-point math is executed in software"* and *"On a CPU without an FPU, like the ESP32-C3, there are no native float instructions."* Table: ESP32 / ESP32-S3 / ESP32-P4 have FPUs; **ESP32-C3 / C6 / H2 do not**.

Consequences:
- Every `float` op → calls `__addsf3`, `__mulsf3`, `__divsf3`, `__eqsf2`, … (libgcc soft-float, ~50-100+ cycles each) or `libm` (`expf`, `sqrtf`, … ~1,500-3,500 cycles each).
- **No FMA instruction exists** (base ISA has none; no F ext). "FMA loops" and hardware-FPU register blocking from x86/ARM literature do NOT apply.
- A naive fp32 GEMM for case 2 (B=1,S=128,D=128,4 layers) costs roughly **~1 minute** of pure soft-float math (about 134M soft float ops × ~70 cycles).
- **Integer hardware (RV32I + M)** is fast: `mul` is native. This makes *integer/fixed-point kernels* the real optimization lever.

Verify on the bench before tuning (5 min):
1. Compile a `volatile float` loop; disassemble with the bundled `riscv32-esp-elf-objdump -d` and confirm you see `call __mulsf3` and **no** `fmadd.s`/`fmul.s`.
2. Cycle-count it with `esp_cpu_get_cycle_count()` and compare against an `int32_t` loop. Expect int to be an order of magnitude faster.

---

## 1. GEMM (the hot loop)

Two viable routes, chosen by build-time switch:

### Route A — soft-float fp32 (correct, simple, ~1 min/forward for case 2)
For v1 correctness this is the baseline-and-reference path (matches torch fp32 within tolerance). Keep it *simple*; it only needs to be the A/B "baseline".
- Straight i/j/k loops; per-dot accumulation in a register.
- `-ffp-contract=off` to keep rounding matches torch (torch GEMM does not FMA-contract on this shape; FMA contraction changes results by ~1 ulp — fine either way, but keep it deterministic).
- Do NOT use `double` anywhere in the hot loop (double soft-float is ~4x slower).

### Route B — integer/fixed-point (fast, ~5-10x, the real winner)
The M extension gives hardware `mul`. Scale weights+activations to a fixed-point Q-format and do MAC in int32, accumulate in int32 (or int16 with Q15 + fp32 accum).
- **Q15/16===16 architecture**: activations Q15 (int16), weights Q15, accumulate int32. ~50-70% speedup over fp32-emul.
- **Register blocking STILL matters** (validated in FastSoftmax + AMD/BLIS materials): 4x4 or 4x8 int blocks in registers to reuse loaded values, break accumulate dependency chains by using ≥4 independent accumulators per block, unroll ×4.
- Keep B-operand (weight) blocks in registers/SNRAM and re-read A rows; pack so inner loops touch contiguous int16/int32.
- Watch out: quantization introduces error vs the fp32 reference. For these small models (dim 128, 4 layers) Q15 + fp32-accum typically stays well inside rtol 0.02 / atol 0.002, but **this must be validated per case** — it is the riskiest knob. Provide a build-time `USE_Q15` switch and compare both against the gate.

### Kernel-principles that transfer (from AMD AOCL-BLAS "small matrices" + BLIS + FastSoftmax):
- For small matrices, *packing overhead can hurt* — skip packing, read straight from buffers, and fuse the bias into the epilogue.
- Keep the micro-kernel loop **contiguous and inlined** (static inline, `__attribute__((always_inline))`), avoid per-tile function-call overhead — this is the "I-cache/LSD" lesson from the AMD small-GEMM article, and it applies to the C3's small instruction cache.
- Break FMA/accumulator dependency chains: 4-8 independent int accumulators.

---

## 2. Causal softmax (avoid materializing, avoid expensive expf)

Sources: FastSoftmax (SzymonOzog) step-by-step; ShivasNotes "Flash Attention on CPU" (online softmax + cache discipline + C kernel); the Milakov–Gimelshein online-softmax writeup.

- **Stream it (flash-attention style):** never allocate the full `S×S` score matrix. For each query row `i`, iterate causal keys `j <= i`, fuse: score → exp → running max/sum rescale → accumulate weighted V. State is O(S) per row. (At S=128 the whole 64 KB score mat *would* fit SRAM, but streaming removes the extra 64 KB and halves traffic — and it's what the reference's semantics need.)
- **Online softmax with rescale:** keep `m` (running max) and `d` (running denominator); when a block max `m' > m`, rescale old numerator/denominator by `exp(m - m')` (multiplier always ≤ 1, no overflow). Normalize once at the end.
- **Use a FAST exp, not `expf()`:** soft-float `expf` is ~2,000-3,500 cycles — the dominant softmax cost. Options:
  - Schraudolph-style bit trick: `exp(y) ≈ 2^((y*ln2scale) + bias)` via integer add + shift on the float bits (fast, ~10 cycles, max rel err ~1e-3 — plenty for softmax denominators).
  - Small LUT (e.g., 256-512 entries) in SRAM.
  - Guard: only recompute exponents when `maxval` actually increases (skip `if (newmax > maxval)` branch), as FastSoftmax stresses (exps are expensive).
- **Scale Q by 1/sqrt(d) once** (pre-multiply each query value) instead of dividing every score; use an approximate reciprocal (`__divsf3` is costly; a 1-2 Newton step on an int-approx or a hardware-friendly float recip with one refine is enough for tolerance).
- **Causal mask = loop bound** (`for j = 0; j <= i; j++`), not a separate -inf fill — free.
- Only stream the needed KV slice into SRAM (the layer's KV is in flash; read once, reuse per query block).

---

## 3. LayerNorm

Sources: karpathy/llm.c docs (forward details + `rstd = 1/sqrtf(v+eps)`), fused residual+LN in llm.c.

- **Two-pass, fp32:** mean → var → `rstd = 1/sqrt(var+eps)`, then `(x-mean)*rstd*w + b`. Two-pass is safest vs the reference (torch computes mean/rstd over the row; E[x²]-E[x]² single-pass has catastrophic cancellation risk — avoid).
- `eps` must match the torch model (check the repo default; typically 1e-5). **Matches reference exactly → lowest gate risk.**
- Use an approximate `rsqrtf` (int-first + 1 Newton iter) instead of soft `sqrtf`+`divsf3` for the non-critical path; soft-float LN on 16K rows is cheap regardless (~0.1 s).
- **Fuse residual add + LN** (llm.c `fused_residual`): `y = LN(x + residual)` in one pass over the row — saves a full activation buffer and one SRAM round-trip. Big win for our tight SRAM budget.

---

## 4. Exact GELU (the one that must NOT be approximated)

- The repo uses **exact** GELU (`erf`-based): `GELU(x) = 0.5*x*(1 + erf(x/sqrt2))`, and hosts use `torch.nn.functional.gelu(x)` default = `approximate='none'` (exact erf).
- Therefore **do NOT switch to the tanh approximation** (that's `approximate='tanh'` and drifts ~1e-3 rel — risky against a strict gate even inside rtol 0.02; and it wastes the "exact" selling point). Implement erf.
- Implement erf with an accurate poly/rational approx (Abramowitz & Stegun 7.1.26 class, max abs err ~1.5e-7; or Cephes `erff` port; or newlib's `erff` if the toolchain ships it — check, then fall back to a table+fitted poly).
- Cost is modest: 65,536 GELUs for case 2; even at ~500-1,500 cycles (soft) that's ≤ ~0.6 s. Fine on Route A; on Route B implement erf in fixed-point or keep this function in soft-float (still cheap).
- Reference for approximation quality and the exact-vs-tanh tradeoff: John D. Cook "GELU: Gaussian Error Linear Units" (2025); Wikipedia «Error function» approximations.

---

## 5. Residual add / bias
- Trivial: loop `dst[i] += src[i]`; fuse bias into the GEMM epilogue and residual into the LN pass. No trick needed; keep memory contiguous and avoid extra buffers.

---

## 6. Cross-cutting C/build choices
- `-O2` + targeted `-funroll-loops`; keep the hot micro-kernels `static inline` in one translation unit; avoid aliasing (`restrict`).
- Put hot kernels' data in **fast internal SRAM**; keep weight matrices in flash-MUU-cached const arrays and copy a layer's matrix into SRAM scratch right before its GEMM (stage ~64 KB at a time) to avoid per-element flash-load stalls.
- `esp_cpu_get_cycle_count()` (or `arduino` `micros()`) for timing; benchmark = warmup + N repeats + median, matching the host harness structure.
- Provide a single header with `#ifdef KERN_USE_Q15` (Route B) vs `KERN_USE_FLOAT` (Route A) so the A/B speedup (naive float vs optimized int) is exactly the contest's baseline-vs-optimized story.

---

## 7. Sources
- Espressif blog (FPU): https://developer.espressif.com/blog/2025/10/cores_with_fpu/
- ESP32-C3 TRM v1.4 (misa/mstatus): https://documentation.espressif.com/esp32-c3_technical_reference_manual_en.pdf
- ESP32-C3 datasheet (Espressif PDF): https://www.espressif.com/sites/default/files/documentation/esp32-c3_datasheet_en.pdf
- FastSoftmax (online softmax step-by-step): https://github.com/SzymonOzog/FastSoftmax
- Flash Attention on CPU (online softmax, cache discipline, C kernel): https://www.shivasnotes.com/blog/5914
- Online softmax paper (Milakov & Gimelshein): https://arxiv.org/pdf/1805.02867
- llm.c LayerNorm docs: https://deepwiki.com/karpathy/llm.c/4.2-layer-normalization
- AMD AOCL-BLAS small-GEMM metrics (loop/I-cache/accumulator-chain lessons): https://www.amd.com/en/developer/resources/technical-articles/2025/aocl-blas-boosting-gemm-performance-for-small-matrices-.html
- John D. Cook, GELU exact vs approximations: https://www.johndcook.com/blog/2025/03/06/gelu/
- GELU background: https://en.wikipedia.org/wiki/Gaussian_error_linear_unit ; https://www.datacamp.com/tutorial/gelu

---
*Compiled 2026-08-28 from Firecrawl research (local copies in .firecrawl/research-kernels/).*
