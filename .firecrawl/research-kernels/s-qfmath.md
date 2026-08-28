[qf\_math](https://deftio.github.io/qf_math/index.html)

[Home](https://deftio.github.io/qf_math/index.html)
[Quick Start](https://deftio.github.io/qf_math/quickstart.html)
[API](https://deftio.github.io/qf_math/api.html)
[Algorithms](https://deftio.github.io/qf_math/algorithms.html)
[Integration](https://deftio.github.io/qf_math/integration.html)
[Tradeoffs](https://deftio.github.io/qf_math/float-math-tradeoffs.html)
[fr\_math](https://deftio.github.io/qf_math/fr-math.html)
[GitHub](https://github.com/deftio/qf_math)

Float math tradeoffs
====================

Choosing between vendor libm, approximate float, and fixed-point for embedded firmware.

Most embedded projects need a handful of math functions — `sin`, `sqrt`, `log`, `exp`, maybe `atan2` — and the question is never _whether_ an implementation exists but _which one_ to link. The answer depends on the target silicon, the error budget, the flash budget, and whether the project can tolerate linking a vendor library whose internals are opaque. This page lays out the landscape and the reasoning behind each choice.

Three families of implementation
--------------------------------

Firmware math on 32-bit MCUs falls into three broad categories:

| Approach | Representation | Typical source | Accuracy class | Code size |
| --- | --- | --- | --- | --- |
| **Vendor libm** | IEEE-754 `float`/`double` | Toolchain (newlib, musl, ARM AEABI, Xtensa libm) | 1–2 ULP (correctly rounded for most ops) | Variable; often pulled in piecemeal by the linker |
| **Approximate float** (qf\_math) | IEEE-754 `float` | Application-linked single TU | ~0.002% FS trig, ~0.001% rel log/exp | ~9–10 KB `.text` |
| **Fixed-point** (fr\_math, libfixmath) | `int32_t` with implicit radix | Application-linked single TU | ~0.008% FS trig, ~0.01–0.4% rel log/exp | ~5–10 KB `.text` |

None of these is universally better. Each trades something for something else. The rest of this page explains what gets traded and when it matters.

IEEE-754 float in thirty seconds
--------------------------------

A 32-bit IEEE-754 `float` stores a sign bit, an 8-bit exponent, and a 23-bit mantissa (plus an implied leading 1). This gives roughly 7 decimal digits of precision across a range of about ±3.4 × 1038. The format is standardized, portable, and understood by every modern compiler.

The catch: operating on floats requires either a hardware Floating-Point Unit (FPU) or a software emulation layer. When an FPU is present — Cortex-M4F, M7, M33, ESP32-S3, RP2350 — basic arithmetic (`+`, `-`, `*`, `/`) runs at near-integer speed. When it is absent — Cortex-M0, M0+, M3, many 8/16-bit targets — every float add or multiply becomes a multi-instruction software routine, typically 10–100× slower than the equivalent integer operation.

Transcendental functions (`sin`, `log`, `exp`, `sqrt`) are never single instructions, even with an FPU. They are always implemented in software, and that is where the three families diverge.

Vendor libm: the safe default
-----------------------------

The toolchain’s math library — typically a variant of newlib, musl, or a vendor-tuned libm — provides `sinf`, `cosf`, `logf`, `sqrtf`, and so on. These implementations aim for correctly-rounded results (1–2 ULP error) and handle every IEEE-754 edge case: NaN propagation, infinities, signed zeros, denormals, `errno` setting.

Advantages:

*   Accuracy is as good as `float` precision allows.
*   No source to maintain — the toolchain ships it.
*   On desktop and server CPUs, these are aggressively tuned (SIMD, branch prediction, large polynomial coefficients precomputed to Chebyshev optimality).

Disadvantages:

*   **Code size is unpredictable.** Linking `sinf` may pull in argument-reduction tables, polynomial coefficients, exception handling, and `errno` glue that add several KB to `.text`. On flash-constrained targets (< 64 KB) this can matter.
*   **Execution cost is opaque.** The function body is inside a prebuilt archive; without disassembling it, the cycle count per call is unknown. On a Cortex-M0 without an FPU, `sinf` via newlib can take thousands of cycles because every float multiply inside the polynomial evaluation is itself a soft-float call.
*   **Over-precision for the application.** Many embedded pipelines work with 10–12-bit ADC readings, 8-bit PWM outputs, or 16-bit sensor data. Sub-ULP accuracy in the math layer is wasted when the signal itself has 0.1% noise.

When to use it: libm is the right answer when accuracy requirements are tight, code size is not the binding constraint, and the target has a hardware FPU. Most of the time, starting with libm and measuring is sound practice — only replace it when measurement shows a problem.

Approximate float: qf\_math’s approach
--------------------------------------

qf\_math keeps the `float` representation but replaces the transcendental implementations with simpler, smaller, faster approximations. The basic techniques:

### Table-driven trig

A 512-entry sine lookup table covers one full period. Input angles are reduced to a Binary Angular Measure (BAM) phase, the table is indexed, and linear interpolation fills the gaps between entries. This trades 2 KB of ROM for a constant-time `sin`/`cos` that never branches on input magnitude. Worst-case error is about 0.002% of full-scale output (±1).

For comparison, a typical libm `sinf` uses a minimax polynomial (5–7 terms) with Cody-Waite argument reduction. The polynomial gives sub-ULP results, but each term is a float multiply-accumulate, and the argument reduction itself involves a conditional branch tree and a table of π-related constants. On a soft-float target, every one of those multiplies is expensive.

### Polynomial log and exp

qf\_math splits the input into an integer exponent (extracted directly from the IEEE-754 bit pattern) and a fractional mantissa, then evaluates a short polynomial on the mantissa. For `log2`, this means a 4th-order polynomial on the range \[1, 2). For `pow2`, a similar polynomial maps a \[0, 1) fractional exponent back to a mantissa, then the integer part is reassembled into the IEEE-754 exponent field. The result is sub-0.001% relative error with no lookup table at all.\
\
### Newton-Raphson sqrt\
\
`qf_sqrt` uses the classic fast-inverse-square-root trick (a magic constant applied to the IEEE-754 bit pattern produces a rough initial estimate) followed by Newton-Raphson refinement iterations. Two iterations give about 0.0005% relative error. This is slower than a hardware `VSQRT` instruction on Cortex-M4F but faster than a soft-float `sqrtf` on Cortex-M0.\
\
### What gets given up\
\
Accuracy, obviously — but in a controlled way. The worst-case errors are known and documented per function. More subtly:\
\
*   **No IEEE edge-case compliance.** Domain errors return a sentinel (`QF_DOMAIN_ERROR`) rather than setting `errno` or returning a quiet NaN. Denormal inputs may not be handled identically to libm.\
*   **No `double` path.** Everything operates in `float`. Applications that need 15-digit precision need a different library.\
*   **Table ROM cost.** The 512-entry sine table and 512-entry tangent table occupy about 4 KB of flash. On a 16 KB part, that is significant. The `QF_MATH_LEAN` build flag trims features (no `log10`/`pow10`, no waves, no ADSR) but the core tables remain.\
\
Fixed-point: the integer alternative\
------------------------------------\
\
Fixed-point math stores numbers as integers with an implicit scale factor (the _radix_): the value 3.5 at radix 16 is the integer 229376. All arithmetic is integer arithmetic. There is no exponent field, no implicit leading one, no denormals, and no FPU dependency.\
\
[fr\_math](https://github.com/deftio/fr_math)\
 is the fixed-point sibling of qf\_math, sharing the same algorithmic heritage (same table layouts, BAM phase system, piecewise hypot, wave generators, ADSR envelope) but operating entirely in `int32_t`. [libfixmath](https://github.com/PetteriAimonen/libfixmath)\
 is another well-known fixed-point library using a Q16.16 format.\
\
For a detailed introduction to fixed-point concepts — radix selection, overflow hazards, Q notation, and worked examples — see the [fr\_math fixed-point primer](https://deftio.github.io/fr_math/pages/guide/fixed-point-primer.html)\
.\
\
Advantages:\
\
*   **No FPU required.** Integer multiply and shift are available on every target, including 8-bit and 16-bit parts.\
*   **Deterministic and bit-exact.** The same input always produces the same output, across compilers, optimization levels, and architectures. This matters for control loops, protocol checksums, and regression testing.\
*   **Small code footprint.** fr\_math’s lean build is about 5 KB. No soft-float runtime gets linked.\
\
Disadvantages:\
\
*   **Limited dynamic range.** A Q16.16 `int32_t` covers roughly ±32767 with 16 fractional bits. Signals outside that range require radix changes or wider storage. `float` spans ±3.4 × 1038 without manual scaling.\
*   **Overflow is silent.** C does not trap integer overflow on signed types (it is undefined behavior in theory, wrapping in practice). The programmer must track value ranges and guard against it. fr\_math provides saturating variants, but using them correctly requires understanding the radix math.\
*   **Bridge cost when mixing with float.** If the rest of the firmware uses `float` (sensor drivers, display code, communication stacks), converting back and forth at every boundary adds overhead and code. The benchmark tables in [BENCHMARK\_CROSSPLATFORM.md](https://github.com/deftio/qf_math/blob/main/compare/BENCHMARK_CROSSPLATFORM.md)\
     show this: fr\_math’s “speed vs libm” ratios include float↔fixed conversion cost that would not exist in a pure-integer pipeline.\
\
The decision in practice\
------------------------\
\
A few rules of thumb that hold across most projects:\
\
| Situation | Likely best choice | Reason |\
| --- | --- | --- |\
| Cortex-M4F / M7 / M33 / ESP32 with single-precision FPU | Start with **libm**; switch to **qf\_math** if code size or cycle count is the bottleneck | Hardware float makes arithmetic cheap; the question is only the transcendental implementation cost. |\
| Cortex-M0 / M0+ / M3 (no FPU) | **fr\_math** or **libfixmath** | Every float operation is soft-float. Fixed-point avoids that entirely. |\
| 8/16-bit MCU (AVR, PIC, MSP430) | **fr\_math** (with appropriate radix) | 32-bit integers are the widest native type; float is prohibitively expensive. |\
| Mixed pipeline: float sensors + integer control loop | **qf\_math** for the float side, **fr\_math** for the integer side, bridge macros at the boundary | `QF_TO_FR` / `FR_TO_QF` macros convert between representations. |\
| Bit-exact reproducibility required (protocol, golden-vector test) | **fr\_math** | Integer arithmetic is deterministic across toolchains. Float results can vary with `-ffast-math`, FMA contraction, or different libm implementations. |\
| Wide dynamic range (audio dB, scientific telemetry) | **qf\_math** or **libm** | Float’s exponent field handles 1038 without manual scaling. |\
\
Error budgets: how much accuracy is enough\
------------------------------------------\
\
The answer depends on the application, not the math library. A few reference points:\
\
*   A **12-bit ADC** has an LSB of ~0.024% of full scale. Math that introduces less error than one LSB is effectively transparent.\
*   A **16-bit DAC** has an LSB of ~0.0015% FS. qf\_math trig (0.002% FS worst case) is right at this boundary; libm is safely below it.\
*   **Motor commutation** typically needs < 1° of electrical angle error (~0.3% FS in sine terms). Both qf\_math and fr\_math are well inside this.\
*   **Audio synthesis** at 16-bit depth has a noise floor of about −96 dB. qf\_math’s trig accuracy (−94 dB or better) is borderline; for production audio paths, libm or a dedicated wavetable with interpolation may be warranted.\
*   **LED dimming**, **PID loops**, **display graphics**, and **sensor fusion** are typically well within the 0.01% class and tolerant of either library.\
\
The general principle: start from the output resolution and work backward. If the final actuator has 10-bit precision, spending cycles on sub-ULP math is waste.\
\
Where the speed difference comes from\
-------------------------------------\
\
On a desktop x86-64, vendor `sinf` and qf\_math `qf_sin` are often within 2× of each other because the CPU has deep pipelines, branch prediction, out-of-order execution, and hardware FMA that make polynomial evaluation cheap. The libm implementation is tuned for that hardware.\
\
On an embedded Cortex-M or Xtensa, the balance shifts:\
\
*   **Simpler pipelines.** Branch mispredictions are cheaper but still nonzero, and most MCU cores are in-order. A branchless table lookup with linear interpolation has more predictable throughput than a polynomial with conditional argument reduction.\
*   **No hardware FMA.** Many M4F cores have a fused multiply-add instruction, but the libm shipped with the toolchain may not exploit it (newlib-nano, for instance, is compiled with conservative flags). qf\_math’s simpler expressions may actually pipeline better.\
*   **Fewer cache levels.** On a desktop, a 4 KB sine table fits in L1 and is fast. On an MCU with single-cycle flash or tightly coupled memory, it is equally fast. The table approach does not get worse on smaller targets the way branch-heavy polynomial code can.\
*   **Soft-float amplification.** On a Cortex-M0, every float multiply inside a libm polynomial is a call to `__aeabi_fmul`. A 7-term polynomial means 7 soft-float multiplies plus accumulates. qf\_math’s table lookup with one interpolation multiply reduces this to 1–2 soft-float operations.\
\
The ESP32-S3 benchmark data illustrates this concretely: `qf_sin` is **4.3×** faster than `sinf` on the same chip, even though both use hardware float. The speedup comes from replacing a ~20-operation polynomial with a table fetch and one multiply.\
\
Code size: what actually ships\
------------------------------\
\
On flash-constrained targets, the relevant number is not the library’s total size but the `.text` that makes it into the final binary after the linker discards unused symbols.\
\
| Library | Variant | Bytes (`.text`) | Notes |\
| --- | --- | --- | --- |\
| qf\_math | full | ~10 KB | All functions, tables, waves, ADSR |\
| qf\_math | lean (`-DQF_MATH_LEAN`) | ~9 KB | Core math only; no `log10`/`pow10`, waves, ADSR |\
| fr\_math | full | ~10 KB | All functions including print helpers |\
| fr\_math | lean (`-DFR_LEAN`) | ~5 KB | Core math only; no degree/BAM wrappers, waves, ADSR, print |\
| libfixmath | bench subset | ~3.6 KB | Trig + sqrt + exp + core only |\
| Vendor libm | (varies) | 2–20+ KB | Depends on which functions are referenced; hidden transitive pulls are common |\
\
One subtlety with libm: calling `sinf` may pull in argument-reduction code that references `__ieee754_rem_pio2f`, which in turn pulls in a table of π/2 coefficients, which pulls in `__kernel_cosf`, and so on. The total linked size can be 3–5× what a single function’s source would suggest. qf\_math and fr\_math avoid this by design — everything is one translation unit with no internal cross-references to other compilation units.\
\
Mixing libraries in one project\
-------------------------------\
\
There is no rule against using more than one math source in the same firmware image. A common pattern:\
\
*   Use **vendor libm** for initialization-time calculations (calibration curves, coordinate transforms computed once) where speed is irrelevant and accuracy matters.\
*   Use **qf\_math** in the real-time loop (motor commutation, audio synthesis, display rendering) where throughput and determinism are the constraints.\
*   Use **fr\_math** for any integer-only subsystem (protocol encoding, fixed-radix PID, bit-exact logging).\
\
The `QF_TO_FR(float_val, radix)` and `FR_TO_QF(fixed_val, radix)` macros in qf\_math.h bridge the two representations at the boundary. The conversion cost is a float-to-int cast and a shift (or the reverse), which on an FPU-equipped core is one or two instructions.\
\
Postscript: DSP libraries, SIMD, and vendor-tuned math\
------------------------------------------------------\
\
The three families above cover general-purpose scalar math. Several other categories exist and are worth understanding even if they serve different niches.\
\
### CMSIS-DSP (ARM)\
\
ARM’s [CMSIS-DSP](https://arm-software.github.io/CMSIS-DSP/latest/)\
 ships as part of the CMSIS ecosystem and provides optimized math for Cortex-M targets. It includes fast trig (`arm_sin_f32`, `arm_cos_f32`), vector operations, FFT, FIR/IIR filters, and matrix routines. The scalar trig functions use table-lookup with cubic interpolation, similar in spirit to qf\_math but tuned for the Cortex-M instruction set.\
\
CMSIS-DSP is large (~40 KB+ for the full library) but modular — individual function groups can be linked selectively. Its real strength is **batch processing**: the filter, FFT, and matrix functions operate on arrays and exploit the Cortex-M’s hardware loop and saturation instructions. For scalar one-off `sin(x)` calls, it offers little advantage over qf\_math or libm.\
\
### ESP-DSP (Espressif)\
\
[ESP-DSP](https://github.com/espressif/esp-dsp)\
 provides signal-processing primitives for ESP32 targets, including dot products, FFT, and FIR/IIR filters. Some functions use the Xtensa HiFi DSP extensions where available. Scalar math functions are minimal — the benchmark harness in this repo tests ESP-DSP’s `dsps_sqrtf_ansi` as a standalone wrapper, but most of the library’s value is in its batch/vector operations.\
\
### SIMD and vectorized math\
\
On larger ARM cores (Cortex-A with NEON) and on x86 (SSE/AVX), SIMD intrinsics or auto-vectorized loops can compute 4 or 8 `sin` values in parallel. Libraries like [SLEEF](https://sleef.org/)\
 and Intel’s SVML provide vectorized transcendentals. These are extremely fast for batch workloads (signal processing, graphics pipelines, physics engines) but are irrelevant on Cortex-M — there is no SIMD unit.\
\
The Cortex-M4’s “DSP extensions” (SIMD-style 8×8 and 16×16 multiply-accumulate) operate on integer data and are useful for fixed-point FIR filters but do not help with scalar float transcendentals.\
\
### Vendor-tuned libm variants\
\
Some toolchain vendors ship math libraries optimized for their specific silicon:\
\
*   **ARM Optimized Routines** (`libmathlib`) — replacements for newlib’s `sinf`, `cosf`, `logf`, etc., tuned for AArch64 and increasingly for Cortex-M. These can be 2–5× faster than stock newlib while maintaining correctly-rounded accuracy.\
*   **TI C2000 FPU FastRTS** — ROM-resident fast-math routines for TI’s real-time MCUs, callable as drop-in replacements for libm symbols.\
*   **Microchip dsPIC libm** — tuned for the dsPIC’s 16-bit DSP engine with hardware loop and barrel shifter.\
\
These are worth investigating when targeting a specific vendor’s silicon and toolchain. The tradeoff is vendor lock-in: the optimized routines are not portable across architectures.\
\
### Hardware math instructions\
\
A few MCU-level instructions deserve mention because they affect the cost baseline:\
\
| Instruction | Available on | Effect |\
| --- | --- | --- |\
| `VSQRT.F32` | Cortex-M4F, M7, M33 | Single-precision square root in ~14 cycles. Faster than any software `sqrt`. |\
| `VDIV.F32` | Cortex-M4F, M7, M33 | Single-precision divide in ~14 cycles. Faster than Newton-Raphson reciprocal. |\
| `VFMA.F32` | Cortex-M4F, M7, M33 | Fused multiply-accumulate. Eliminates one rounding step in polynomial evaluation. |\
| None (soft-float) | Cortex-M0, M0+, M3 | All float ops are function calls (~20–70 cycles each). Fixed-point wins here. |\
\
When hardware `VSQRT` is available, `qf_sqrt` will not beat it — the benchmark data confirms this (`qf_sqrt` is ~0.39× libm speed on a desktop, where `sqrtf` compiles to a single SSE instruction). The value of `qf_sqrt` is on targets _without_ that instruction. The same reasoning applies to `VDIV`: if division is a single-cycle instruction, Newton-Raphson reciprocal approximation is slower.\
\
### When to look beyond scalar libraries\
\
If the workload is inherently **batched** — computing `sin` for 256 samples per audio frame, applying a 64-tap FIR filter, running a 1024-point FFT — then CMSIS-DSP, ESP-DSP, or hand-written SIMD/DMA routines will outperform any scalar library by a wide margin. qf\_math, fr\_math, and libm are designed for **scalar, per-sample** operations: one `sin` call per control-loop tick, one `atan2` per sensor reading, one `sqrt` per distance calculation.\
\
In many embedded projects, the two coexist naturally: a DSP library handles the signal-processing pipeline, and a scalar math library handles the control logic, UI calculations, and calibration routines.\
\
qf\_math — BSD-2-Clause — © M. A. Chatterjee[GitHub](https://github.com/deftio/qf_math)\
 · [fr\_math](https://github.com/deftio/fr_math)