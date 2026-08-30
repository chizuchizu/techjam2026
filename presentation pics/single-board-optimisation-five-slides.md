# Case 2 single-board optimisation — five-slide presentation content

## Slide 1 — Remove the no-FPU attention bottleneck

### Main message
The original ESP32-C3 spent most of the forward pass doing software floating-point attention. The first two optimisations changed the expensive inner loops to fixed-point integer arithmetic.

### Baseline problem
- Hardware: one XIAO ESP32-C3 at 160 MHz, RV32IMC, no hardware FPU.
- Every FP32 add, divide, conversion, and `expf()` call is implemented by software.
- Attention consumed **30.09 s of the 42.15 s forward** — about **71%** of total runtime.
- The attention workload included millions of QK products, softmax exponentials, and PV products.

### Opt 1 — Integer QK, softmax, and PV
- QK dot products changed from FP32 dequantised multiplication to Q15×Q15 integer products.
- `int64` accumulation was used because 32 products can exceed the signed 32-bit range.
- Softmax became a numerically stable two-pass algorithm: find the exact maximum, then exponentiate `score − max`.
- PV accumulation became integer multiply-accumulate; only one dequantisation step remained at the end of each row.
- This removes FP32 work from the hottest attention loops while preserving softmax stability.

### Opt 2 — Integer exponential lookup table
- Replaced software `expf()` with a 513-entry int16 lookup table over approximately `[-10, 0]`.
- The index uses fixed-point arithmetic and linear interpolation between neighbouring table entries.
- A Q24 index was rejected because its coefficient lost precision; the final Q32 index preserved the required accuracy.
- This converts an expensive transcendental function into integer indexing, shifts, and interpolation.

### Result
```text
Complete forward: 42.15 s → 15.21 s → 13.70 s
Overall speedup:       1.0× → 2.77× → 3.08×
Attention:             30.09 s → 3.15 s → 1.64 s
```

**Takeaway:** the largest win came from matching the arithmetic to the processor: integer operations instead of software floating point.

---

## Slide 2 — Reduce memory traffic and remove intermediate tensors

### Main message
After attention was fixed, GEMM, GELU, quantisation, and data movement became the next bottlenecks. These optimisations reused registers, fused passes, and kept data in integer form.

### Opt 3 — Register-tiled GEMM
- The scalar GEMM repeatedly reloaded flash-resident weights.
- `core2` computed two output rows together; `core3` computed four rows together.
- A weight loaded once could be reused across several output accumulators.
- This reduced expensive flash/XIP weight traffic without changing the accumulation order.

### Opt 4 — Scale-exact integer GELU LUT
- Replaced the FP32 degree-11 GELU polynomial with a 513-entry runtime-generated LUT.
- A fixed LUT was inaccurate because Q15 values do not reveal the activation's real scale.
- The table is rebuilt from the layer's actual activation maximum, so the GELU lookup remains scale-correct.
- GELU was folded into the FFN2 quantisation path, removing an FP32 output buffer and an extra full pass.

### Opt 5 — Attention and quantisation micro-optimisations
- Replaced slow `llrintf()` rounding with integer rounding and clipping.
- Packed two QK products into one integer operation where possible.
- Used parallel PV accumulators to avoid spilling an int64 accumulator array to memory.
- Corrected a stale profiling-buffer alias so measured timings represented the real f2 cost.

### Opt 6 — Fused QKV quantisation
- Q/K/V projections now accumulate directly into integer scratch storage.
- The implementation tracks the activation maximum during projection instead of writing FP32 output, rereading it, scanning it, and quantising it.
- Q30 coefficients preserve precision even when accumulators approach the 32-bit limit.
- This removes FP32 staging buffers and separate amax/quantisation passes.

### Opt 7 — Fully integer exponential index
- Removed the remaining int64→FP32→integer conversions used to index the exponential LUT.
- A Q32 scale is computed once per head; each score then uses one fixed-point multiply and shift.
- The exponential stage fell from approximately **194.8 µs to 23.2 µs per call**.

### Opt 8 — `core4` j-outer GEMM
- Changed the traversal to output-column outer, with an eight-row inner tile.
- Each flash weight column is read once for eight output rows.
- Only eight accumulators are used so the tile fits the RV32 register file without spills.
- Applied to the output projection, FFN1, and FFN2 GEMMs.

### Result
```text
Complete forward: 13.70 s → 6.91 s → 6.56 s → 5.27 s
Overall speedup:       3.08× → 6.10× → 6.43× → 8.0×
```

**Takeaway:** the implementation stopped moving FP32 tensors between stages and instead reused integer data in registers and existing scratch buffers.

---

## Slide 3 — Fuse LayerNorm, context, and quantisation

### Main message
The next gains came from removing complete-buffer passes. Instead of writing FP32 data and converting it later, each stage computes its scale and emits the final Q15 representation directly.

### Opt 8b — Optimised head GEMM and integer amax
- Added a j-outer head-projection kernel for the smaller Q/K/V head GEMMs.
- Moved the amax calculation to integer operations.
- This reduced the remaining QKV/head-projection overhead.

### Opt 9 — Fused LayerNorm → Q15
- The old path performed LayerNorm, wrote FP32 output, scanned for amax, and quantised in separate passes.
- A safe analytic bound estimates the maximum possible LayerNorm output from the row statistics and learned gain/bias.
- The Q15 scale is therefore known before the normalisation pass finishes.
- Normalised values are written directly to Q15, eliminating the FP32 staging buffer and two later scans.

### Opt 9b — Integer LayerNorm pass and direct Q15 output
- LayerNorm statistics and quantisation were moved to fixed-point arithmetic.
- Integer bit comparisons replace FP32 absolute-value scans.
- A split mantissa/exponent representation provides accurate multiply-and-shift conversion.
- The final output is rounded and clipped directly into Q15.

### Opt 10 — Q15 attention-context fusion
- Attention now writes Q15 context directly instead of producing FP32 context for the output projection.
- A two-phase design first projects all V heads and computes a safe global context bound.
- The output projection consumes Q15 context directly, so its FP32 amax scan and input quantisation disappear.
- The context bound is valid because attention context is a weighted average of V rows.

### Opt 11 — Integer PV and context epilogue
- Softmax weights are rescaled so each row sums to approximately 32767.
- This lets PV use int32 accumulation instead of expensive int64 arithmetic.
- The context scale is calculated once per head rather than once per element.
- The context epilogue becomes an integer multiply, rounding shift, and clamp.

### Opt 12 — Two-column tiling and K-pair prefetch
- `core4_v2` processes two output columns per tile.
- Activation loads are reused across both columns.
- K-pair prefetching places flash loads ahead of the multiply sequence to hide XIP latency.
- Odd-size tails remain supported even though the target shape is 128×128×128.

### Opt 13 — Integer LayerNorm statistics
- Replaced FP32 `fabsf()`/comparison logic with integer bit-level maximum detection.
- Replaced per-element software-FP scale conversion with exact fixed-point mantissa/exponent arithmetic.

### Opt 14 — Integer GEMM amax and Q15 quantisation
- GEMM input amax scans became integer operations.
- Per-element input quantisation now uses the same exact fixed-point helper as LayerNorm.
- This removes duplicate FP32 scan and conversion loops.

### Result
```text
Complete forward: 5.27 s → 4.862 s → 4.160 s → 3.969 s → 3.706 s → 3.664 s → 3.205 s
Overall speedup:       8.0×                         →                         13.1×
```

**Takeaway:** fusion saved more time than arithmetic optimisation alone because it removed repeated writes, reads, scans, and conversions across the entire model.

---

## Slide 4 — Fit the kernels to the RV32 register file

### Main message
The processor has limited registers and an in-order pipeline. Larger tiles are not automatically faster: once registers spill to the stack, the optimisation reverses direction.

### Opt 15 — `core5` register-pressure fix
- `core4_v2` used 16 accumulators plus prefetched operands and spilled registers.
- `core5` uses a four-row × two-column tile with eight accumulators.
- The smaller tile fits in the available RV32 registers while retaining K-pair processing.
- The kernel improved from about **7.48 to 6.21 cycles/MAC**.

```text
Forward: 3.184 s → 2.982 s  (−202 ms, −6.3%)
```

### Opt 16 — Hand-scheduled head GEMM assembly
- Replaced the Q/K/V head GEMM inner loop with RISC-V inline assembly on the device.
- Loads, eight multiplies, and eight additions are scheduled so multiply latency is hidden.
- The C implementation remains for host validation.

```text
Head GEMM: 17.1 ms → 13.43 ms per call
Forward:    2.982 s → 2.838 s
```

### Opt 17 — Direct FFN1 Q15 output
- FFN1 no longer writes FP32 output and then performs a separate amax scan and Q15 conversion.
- Bias is folded into the integer accumulator.
- The accumulator tracks amax while computing and emits Q15 in place.
- GELU and FFN2 can consume that Q15 buffer immediately.

```text
FFN1:    81.5 ms → 54.0 ms per call
Forward: 2.838 s → 2.701 s
```

### Opt 18 — Faster requantisation, bias folding, and limb-based QK
- KB1 replaces the slow int64/soft-FP requantisation path with hand-scheduled RISC-V multiply-high arithmetic.
- Bias folding removes one FP32 addition from core5 GEMM epilogues.
- QK's 64-bit dot product is represented as two exact 32-bit limbs with explicit carry/borrow handling.
- A signed-carry bug was found and corrected using millions of random int64 comparisons.

```text
KB1:     3.95 ms → 0.925 ms per call
Forward: 2.701 s → 2.447 s
```

### Opt 19 — QK unrolling and PV accumulators
- Four causal QK rows are computed together while sharing query loads.
- PV uses eight independent accumulators to reduce probability-load traffic.
- QK improved by approximately **22%**; PV was already multiply-latency bound and changed little.
- Accumulation order remained unchanged, so attention stayed bit-exact.

```text
Forward: 2.444 s → 2.386 s
```

### Opt 20 — Measure before adding more code
- This was a profiling and feasibility stage rather than a new fast kernel.
- It showed that approximately 99.9% of usable SRAM was already occupied.
- IRAM placement and larger tiles would overflow memory or cause register spills.
- It established that the remaining bottlenecks were instruction latency and FP32 epilogues, not simply flash bandwidth.

**Takeaway:** the fastest kernel was the one that fit the hardware—not the one with the largest tile.

---

## Slide 5 — Make the residual path integer and validate the final kernel

### Main message
The residual stream connected every Transformer layer. Keeping it in FP32 forced repeated conversions and memory passes, so the final optimisations made the whole FAST path integer end-to-end.

### Opt 21 — Integer residual FAST path
- The residual is stored as fixed-scale int32 in the same 64 KB union used by the FP32 input buffer.
- Output-projection and FFN2 epilogues add integer results directly into the residual.
- Separate FP32 residual passes disappear.
- LayerNorm reads the int32 residual and performs a local integer rescale before Q15 normalisation.
- The local rescale preserves Q15 precision without sacrificing the wider int32 residual range.
- The final norm also uses integer statistics before emitting the FP32 result.

```text
Forward: 2.385 s → 2.122 s
Saving:  −263 ms (−11.0%)
Removed: 86 ms of residual passes, plus faster epilogues and norms
```

### Opt 22 — Compose independent optimisations
- Combined the integer residual path with the hand-assembled KB0 head GEMM.
- The two changes are orthogonal: one improves residuals/epilogues, the other improves QKV projection MACs.
- Both can be used without changing the other kernel's data representation.

```text
Forward: 2.122 s → 2.056–2.057 s
```

### Opt 23 — Correct and validate core5 assembly
- The first core5 assembly version reused an activation register for the second output column.
- Column 0 was correct, but column 1 could calculate `(activation × weight0) × weight1`.
- The fix gives each product its own destination register and restores the missing local loop label.
- A dedicated probe compared both assembly columns against the C reference: `bad=0`, `worst=0`.
- All 25 device seeds then passed the numerical gate.

```text
Forward: 2.056 s → 1.996 s
Final speedup: 42.15 / 1.996 = 21.1×
```

### Final significance
- **95.3% less time per forward**.
- Approximately **40.15 seconds saved per input**.
- Final validation: **25/25 device seeds** and **54/54 host checks** passed.
- Worst reported absolute error was approximately **1.24e−3**, below the **2e−3** limit.
- The result is a complete four-layer Transformer forward on one ESP32-C3, not an attention-only benchmark.

**Takeaway:** the final speedup came from a complete integer data path, hardware-aware assembly, aggressive fusion, and correctness checks after every major step.
