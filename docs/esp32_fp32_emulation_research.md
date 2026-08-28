# "Most Optimized" FP32 Emulation for the ESP32-C3 — Paper-Grounded Research Note

**Date:** 2026-08-28  **Status:** recommendation (needs on-bench validation)
**Context:** XIAO ESP32-C3, RV32IMC, no FPU, soft-float today; accuracy gate rtol 0.02 / atol 0.002 vs a torch reference.

> Note: the Firecrawl *Agent* endpoint is still disabled server-side ("Agent beta is not enabled") on this self-hosted instance, so this research used the Firecrawl search + scrape engine (same content layer). Primary sources scraped into `.firecrawl/research-kernels/` (s-softfloat.md, s-schraudolph.md, s-qfmath.md, s-no_fpu_cortexm.md, s-erf_cheb.md, s-erf_arx.md, paper-*.json).

---

## 1. The hard ceiling of exact IEEE soft-float (why "faster emulation" alone loses)

Exact software IEEE fp32 has a measured performance ceiling regardless of implementation:

- **Berkeley SoftFloat 3e** official speed table (0pt., gcc -O2): on a **1-GHz in-order ARM Cortex-A8**, single-precision = **8.1 Mop/s add, 10.0 Mop/s mul, 7.0 Mop/s div** → ~100–125 cycles/op. Scaled to 160 MHz, that is **~1.3–1.6 Mop/s**.
- **libgcc soft-float** (what your toolchain uses) lands in the same ~50–100+ cycles/op class.
- Implication for our workload: case 2 needs ~134M soft float ops → **~1–2 minutes per forward** even with a *perfect* exact implementation. SoftFloat's design goal is **IEEE conformance, not speed** (it even carries exception flags, denormals, all rounding modes — all of which we do not need and which slow it down further).
- There is **no paper** that makes exact IEEE soft-float an order of magnitude faster on a scalar in-order core: the algorithms (align, operate, normalize; multiword division/sqrt) are already near the instruction-count floor. Speedups in the literature come from (a) dropping IEEE features you don't need (~20–40%: no denormals, no flags, near/fast rounding) or (b) leaving IEEE entirely.

**Conclusion: the fastest "FP32 emulation" for this project is a hybrid — keep an fp32-compatible API, but implement the high-volume math with fast integer/approximate kernels, and use exact soft-float only for low-volume, correctness-critical spots.** The gate (rtol 0.02) is exactly what makes this legitimate, and it is the same conclusion the embedded-math literature (qf_math/fr_math, CMSIS-NN, ARM's own "Options for FP Math Without FPUs") reaches for FPU-less targets.

## 2. The layered solution (per-kernel, papers cited)

| Kernel | Volume (case 2) | Recommended implementation | Evidence / paper | Expected vs soft-float |
|---|---|---|---|---|
| **GEMM (the 90%)** | 134M ops | Fixed-point: Q15×Q15→int32 MAC, register-blocked 4×4, ≥4 independent accumulators, unroll ×4, no packing at D=128 | CMSIS-NN (ARM, int16/int8 MAC); qf_math/fr_math benchmark (0.01–0.4% rel log/exp accuracy); AMD/BLIS small-GEMM (loop/accumulator/dependency-chain lessons) | **5–10× faster** |
| **Softmax exp** | ~262K | Schraudolph bit-math exp: `(int)u = EXPA*y + (1072693248-EXPC)`; `EXPA=1048576/ln2 (1512775 int)`, `EXPC=60801`; `~4 instr`, ~5–10 cyc | Schraudolph, *"A Fast, Compact Approximation of the Exponential Function"*, Neural Computation 11(4):853–862, 1999 (PDF scraped; Fig.2 macro). Max rel err ≈2.5% (tunable to ~1%) | **200–500× faster** than `expf` |
| **Attention scale / LayerNorm rstd** | ~ thousands | Fast inverse sqrt: magic-constant guess (0x5f3759df family) + 1–2 Newton iterations; avoid divisions | qf_math doc (`qf_sqrt` = FISR + Newton); Newton–Raphson division surveys (MDPI 2025; blog.segger) | 5–20× faster |
| **GELU / erf** | ~65K | Keep **exact-erf semantics**; use Cody rational-Chebyshev erff (1969): `erf(x)=x·R(x²)` on |x|≤0.5, `erfc(x)=e^-x²·R(x)` beyond; max rel err ~1e-7 (**not** the tanh GELU — host uses `approximate='none'`) | Same/slightly faster; exact |
| **LayerNorm mean/var** | 16K·2 | Exact soft-float fp32 two-pass (`sqrt` via FISR rstd) — low volume, keep correct | llm.c LN docs; qf_math | parity |

Notes:
- Q15×Q15→int32: `(w16*x16*global_scale + bias)>>Q_shift`, saturate, convert back to float at the activation boundary. Standard no-FPU ML path (CMSIS-NN validation).
- Conversions float↔int are cheap and happen once per tensor boundary, not per element.

## 3. Why not the other "emulation" options

- **Berkeley SoftFloat 3e**: gold-standard exactness; **not faster** than libgcc for basic ops; adds flags/denormals we don't need. Use only if we ever need *provable* IEEE conformance. (jhauser.us/arithmetic/SoftFloat.html)
- **Custom stripped exact soft-float** (no denormals/flags, fast rounding): ~20–40% win only; not worth the risk/effort vs the int path, which is 5–10×.
- **Block-float / half-float**: half precision loses too much for our gate; block-float is real complexity. Defer.
- **POSIT**: no advantage for dense GEMM + softmax; adds toolchain/validation risk. Defer.
- **External FPU (STM32 FMAC/CORDIC)**: not applicable to ESP32-C3 — no such accelerator.

## 4. Diet (build/run procedure)

1. Implement all kernels with an `ACCURACY_MODE` switch (EXACT = libgcc soft-float path; FAST = int/approx path), **per kernel**, behind one header. This is also your contest "naive vs optimized" A/B.
2. Run the host exporter/verifier (25 seeded trials, elementwise `|a-b| ≤ 0.002 OR ≤ 0.02·|b|`, zero failures).
3. Validate **each kernel swap in isolation** (EXACT-GEMM + FAST-softmax first, then FAST-GEMM, etc.) — gives per-kernel worst-case error and lets you revert any kernel whose error breaks the gate.
4. If a specific kernel's FAST path fails the gate: keep that one on EXACT (soft-float) — it will still be faster overall than today, and the failing component is contained.
5. Time with `esp_cpu_get_cycle_count()`; report median of 20 runs.

## 5. Expected result
- **GEMM (FAST):** ~67M int MACs × ~6–10 cyc ≈ 0.3–0.7 s (vs ~60 s soft-float).
- **Softmax (FAST exp):** ~262K × ~10 cyc ≈ 0.02 s (vs ~4 s).
- **GELU/LN:** < 0.5 s combined.
- **Total forward: roughly 1–2 s** — back in the original "realistic" ballpark, but now with a measured A/B story (soft-float baseline ~60 s vs hybrid ~1–2 s ≈ **30× speedup**).

## 6. Sources (scraped copies in `.firecrawl/research-kernels/`)
- Schraudolph 1999 paper PDF: nic.schraudolph.org/pubs/Schraudolph99.pdf
- Berkeley SoftFloat: jhauser.us/arithmetic/SoftFloat.html ; ucb-bar/berkeley-softfloat-3
- deftio qf_math "Float math tradeoffs — libm, qf_math, fixed-point": deftio.github.io/qf_math/float-math-tradeoffs.html ; fr_math (fixed-point sibling)
- ARM "Options for Floating Point Math on Cortex M Without FPUs": systemonchips.com
- CMSIS-NN (int16 MAC validation): arm-software.github.io/CMSIS-NN
- Cody, *Rational Chebyshev Approximations for the Error Function*, Math. Comp. 23(107) 1969 (AMS PDF scraped)
- Global erf approximations for vectorized computation (arXiv 2504.05068, 2025)
- Newton–Raphson division/reciprocal: MDPI Electronics 15(13):2899 (2025); segger blog; fp32.org

---
*Prepared from Firecrawl research; the FAST-vs-EXACT claims must be verified on the bench against the tolerance gate before being trusted.*
