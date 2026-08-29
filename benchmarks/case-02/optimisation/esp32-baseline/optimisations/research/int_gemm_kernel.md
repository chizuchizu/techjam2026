# int16×int16 fixed-point GEMM kernels on a single-issue in-order RV32IMC core

Date: 2026-09-01 · Branch: `esp32-baseline` @ `f67a101` (opt10b, 3.97–4.02 s/forward, 10.6× vs fp32 reference)
Scope: research + implementation sketches for speeding up the **Q15×Q12 (int16×int16 → int32)**
GEMM kernels (the six per-layer projections, 100.66 M-MAC/forward) and the **Q15×Q15 attention
QK/PV** dots on the ESP32-C3 (RV32IMC 160 MHz, no FPU/SIMD, 4-stage in-order scalar core, 320 KB SRAM).
Gate to protect: per-element `|Δ| ≤ 2e-3 OR |Δ| ≤ 2%·|ref|` vs fp32 torch (current FAST worst ≈ 1.1e-3).

---

## TL;DR — the three highest-value, C-implementable techniques

1. **Pack two adjacent int16s into one 32-bit load** (`lw` instead of 2×`lhu`), then extract
   with shifts — the RISC-V translation of CMSIS-DSP's `read_q15x2_ia` / `__SMLAD` dual-MAC.
   Halves the load-instruction count in the GEMM inner loop (the issue-slot bottleneck on this
   core) and in the QK dot. Expected: GEMM inner ~7.7 → ~5.5–6.0 cyc/MAC, ≈ 0.4–0.7 s/forward.
2. **Software-pipeline the GEMM/QK inner loop** — preload next-k A/B words ahead of use and keep
   8 (GEMM) / 4–8 (QK) independent int32 accumulators. On the 4-stage in-order core every `lhu`
   has exposed load-use latency (zero-wait SRAM data, but the load→MUL stall is real); unroll ×2
   on k with a 2-iteration preload hides it. Pairs with (1): each prefetch is one `lw` for two
   operands. Expected: effective ~3.5 → ~2 cyc/MAC·issue + hidden stalls.
3. **Drop int64 accumulation where model data allows, using measured per-slot overflow margins** —
   QK dot: keep four independent int32 accumulators (8 dims each, measured worst lane 1.76e9 =
   0.82·2³¹ over 25 seeds) instead of the current serial int64 `dot` (compiler emits `mul`+`mulh`+
   two-word adds ≈ 9 instr/dim → ≈ 4 instr/dim). GEMM (K=128, measured worst 0.29·2³¹) and PV
   (worst 0.50·2³¹, already int32-4-lane) are already safe without int64. This makes *all* inner
   loops pure int32: 1 MUL + 1 ADD per MAC, no 64-bit carries.

A fourth, smaller win: i-tile 8 × j-tile 2 register blocking with a **transposed-B** operand in
SRAM (block-fill W once per layer instead of re-reading flash-XIP weight columns) — the structure
of CMSIS-DSP `arm_mat_mult_(fast_)q15.c`; cuts B-side loads and flash traffic in the current
j-outer core4.

---

## 0. Target platform facts (verified from primary sources)

ESP32-C3 Technical Reference Manual (v1.4), §1.1/§3.1:
> "ESP-RISC-V CPU is a 32-bit core based upon RISC-V ISA comprising base integer (I),
> multiplication/division (M) and compressed (C) standard extensions. The core has **4-stage,
> in-order, scalar pipeline** optimized for area, power and performance."
> "Zero wait cycle access to on-chip SRAM and Cache for program and data access over IRAM/DRAM
> interface."

- RV32IMC ⇒ 1 `mul` (32×32→64, low 32) / cycle issue; **no FPU** (misa F=0, D=0), **no `div`-relevant
  cost** (divides are slow but rare), no hardware SIMD/P-extension (`misa` K=0 ⇒ no PMULH.H/PMACC.W).
- Practical scalar ceiling: **160 MMAC/s** ⇒ 100.66 M-MAC of GEMMs alone claim ≥ 0.63 s of pure
  `mul` issue; attention QK (4.23 M) + PV (8.4 M) add ~0.08 s ⇒ **kernel-MAC floor ≈ 0.71 s**
  if every MAC were 1 cycle with no loads. Current 3.97 s ⇒ ~5.6 cyc/MAC aggregate for kernels,
  7.7 cyc/MAC in the GEMM inner loop ⇒ ~3–4× headroom to the practical ~1.7–2 cyc/MAC
  (MUL+ADD issue) floor. This bounds what any kernel rewrite can give.

## Current kernel inventory (src/kernels.c, src/model.c, FAST mode)

| kernel | shape/loop | acc | loads | measured |
|---|---|---|---|---|
| QKV head GEMM `tm_gemm_head_q15` | j-outer (HD cols) × 8-row i-tile, K-inner, 8 int32 accs (same shape as core4) | int32 | `lhu` per element | QKV slot ≈ 1.31 s (incl. quant) |
| oproj/f1 `tm_gemm_core4` (K=128) | j-outer × 8-row i-tile, K inner, 8 int32 accs | int32 | `lhu` per element | oproj 101 ms/call → ≈0.42 s + f1 ≈0.2 s |
| f2 `tm_gemm_core3` (K=128) | 4-row × 2-col, 8 accs | int32 | `lhu` | ≈ 0.3 s |
| QK `attn_head` | per (i,j≤i) int64 dot over HD=32, unroll 2 | **int64** | 4×`lhu` per 2 dims | attn ≈ 1.21 s total (QK ≈0.3–0.4, exp 0.02, PV 0.8) |
| PV | 4 int32 accs over d-blocks of 4, p15-rescaled | int32 | `lhu` per vj | in attn slot |

Epilogues (per-GEMM dequant `(int64)acc*GX+BX >> 30`, fp32→Q15 casts) are already near-minimal
integer work; they are not the target here.

---

## 1. Register blocking & accumulator scheduling

**What.** Choose (i-tile, j-tile, K) so that (a) the inner loop is a rectangular micro-tile with
independent accumulators per output element, (b) each weighting operand is loaded once per k and
reused across the tile, (c) the B operand is traversed contiguously (transpose B into SRAM ahead
of time), and (d) accumulator count ≤ register budget so no accumulator spills to SRAM.

**Evidence from primary sources.**
- CMSIS-DSP `arm_mat_mult_fast_q15.c` (Arm): transposes B first (`pSrcBT`), then processes **2 A-rows
  × 2 B-cols** per k-step with 4 accumulators (`sum, sum2, sum3, sum4`) and 4 `__SMLAD` (each = dual
  MAC) per 4 MACs. Rationale: maximum reuse of the 4 loaded q15-pairs with the fewest instructions;
  contiguous B rows so the inner loop has no stride multiplies.
- CMSIS-DSP `arm_mat_mult_q15.c`: same transpose-B; non-DSP path still unrolls `colCnt = numColsA >> 2`
  (×4) on the k-loop.
- TFLite Micro `tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h`:
  reference int8/int16 FC is a clean `for d: acc += filter[d]·(input[d]+offset)` with one
  `BiasType` (int32) accumulator and a single final `MultiplyByQuantizedMultiplier(acc, mult, shift)`
  requant — the canonical "int32 accumulate + one-shot requantize" structure, *no* inner-loop 64-bit.
- muRISCV-NN (tum-ei-eda, RISC-V port of CMSIS-NN): keeps the same 2×2 micro-tile + q7x4/q15x2
  packed loads and 4-accumulator structure, just with RVV/P packed instructions — confirms the tile
  shape is ISA-independent.

**Where it lands for this model.**
- Current core4 is j-outer × 8-row i-tile × K-inner with 8 int32 accs — already in the right family.
  The concrete deltas vs. sources:
  1. **j-tile 2 (or 4)**: with 8 i-rows × j-tile 2 = 16 int32 accs, the single `b=wr[k]` load is
     reused by 8 MACs *and* a second weight row is loaded+MACed in the same k-step — 2 weight loads
     per 16 MACs instead of 1 per 8 (halves B-side `lhu` per MAC) and doubles independent-accum
     ILP to hide latency.
  2. **Transpose-B into SRAM per layer** (weight block for all 6 mats is 0.79 MB in flash-XIP;
     column-strided j-outer re-reads flash columns). A 128×128 Q12 block is 32 KB — copy Wᵀ to a
     32 KB SRAM scratch once per layer, then the i×j tile inner loop reads both A and B
     forward-contiguously at 1 `lw` per 2 operands. Flash-XIP reads at 160 MHz also have a fixed
     per-access cost; halving B-side access count helps beyond SRAM latency.
  3. **K-outer vs j-outer**: with the full Wᵀ block resident, an M-tile × N-tile × K-blocked GEMM
     (CMSIS-style) becomes possible; the current j-outer exists to stream W from flash. Sources
     indicate the SRAM-resident transpose is the better trade once the block fits (it does: 32 KB).

**Expected gain.** GEMM issue ~3.6 → ~2.2–2.5 instr/MAC (loads halved, b-reuse doubled) and stall
fraction reduced; realistic GEMM block ≈2.2 s → ≈1.3–1.5 s. **Accuracy risk: none** — same int32
accumulators, same final requant; only operand layout + tile shape change.

**Implementation sketch.**
```c
/* per layer: Wtblocks[128][128] int16 Q12, transposed once */
for (int nb = 0; nb < Nt; nb += 2)          /* 2 output cols */
  for (int mb = 0; mb < M; mb += 8) {       /* 8 output rows  */
    int32_t c[8][2] = {{0}};
    const int16_t* a0 = A + mb*K; ... a7 = A + (mb+7)*K;
    const int16_t* w0 = Wt + nb*K;           /* transposed: contiguous */
    const int16_t* w1 = Wt + (nb+1)*K;
    for (int k = 0; k < K; k += 1) {
      int32_t b0 = w0[k], b1 = w1[k];
      int32_t a0v = a0[k]; ... a7v = a7[k];
      c[0][0]+=a0v*b0; c[1][0]+=a1v*b0; ... c[7][0]+=a7v*b0;
      c[0][1]+=a0v*b1; ... c[7][1]+=a7v*b1;
    }
  }
```
(Loads: 8 A + 2 B per 16 MACs — 0.625 loads/MAC before pairing; with §2 pairing → 0.4 `lw`/MAC.)

---

## 2. 32-bit "SIMD-in-register" pairing of two int16s

**What.** Two *adjacent* int16 operands live in one 32-bit register lane: one `lw` fetches both
(identical cost/dispatch to one `lhu`), the two halves are sign-extracted with 1 `slli`+1 `srai`
(or `srai`+`sext`), and a `MUL`+`ADD` pair per operand keeps 16-bit×16-bit→32-bit MACs at exactly
the scalar 1 MAC/cycle — no `mulh` ever needed because Q15×Q15 and Q15×Q12 products fit int32.

**Evidence.**
- CMSIS-DSP `arm_mat_mult_fast_q15.c`: `read_q15x2_ia(&p)` is a single 32-bit load producing a
  `q31_t` with two q15 halves; `__SMLAD(inA, inB, sum)` is one ARM instruction doing
  `sum += lo(A)·lo(B) + hi(A)·hi(B)` (dual MAC). The *instruction* doesn't exist on RV32IMC, but
  the **memory-side** idea (1 load = 2 operands) does, and it is exactly what `lw` + two shifts
  reproduces.
- RISC-V P-extension spec (riscv-p-spec): `PMULH.H`, `PMACC.W.H00` implement packed dual 16×16→32
  MACs in one instruction — the ISA's own acknowledgment that pairwise is the granularity the
  architecture targets. RV32IMC must do it with 2 `mul` + 1 `add` + 2 extract, which is still fewer
  *instructions* than 2 separate `lhu`+`mul`+`add` when the loads are paired.
- muRISCV-NN: `pack_q7x4` / `pack_q15x2` helpers exist precisely to feed packed dual-MAC ops.

**Where for this model.**
- **GEMM inner loop** (all six projections, 100 M-MAC): currently per k-step core4 issues 8 A-`lhu`
  + 1 B-`lhu` — pairing each pair of adjacent A values (rows are contiguous in memory, and the
  i-tile rows a0..a7 are adjacent rows ⇒ A[d],A[d+1] are adjacent *columns* of the same row, only
  if we process 2 K per step). The cleanest pairing is **K-pairing**: process k and k+1 together —
  `lw` reads a0[k..k+1], a1[k..k+1], …, w0[k..k+1], w1[k..k+1]; each pair gives 2 operands from
  1 load. This is exactly what §1's tile becomes. Loads drop from (8+2)=10 `lhu` → 5 `lw` per
  16 MACs = 0.31 `lw`/MAC.
- **QK dot**: qi16[d],qi16[d+1] are adjacent ⇒ 1 `lw` for the q pair, 1 `lw` for the k pair per
  d-step (instead of 4 `lhu`), 2 `mul`, 2 `add` into 2 int32 lanes.
- Alignment note: A rows are 128×int16 = 256 B, and Wᵀ rows 256 B ⇒ `lw` pairs land on 2-byte
  alignment only; a 32-bit load from an even halfword boundary is legal on RV32 (unaligned halves
  are fine as long as the address is byte-aligned; the two halves are both sign-extracted by hand).
  Use `memcpy`-style or `__attribute__((packed))` cast or just `int16_t*` + `*(uint32_t*)` with
  `-fno-strict-aliasing` / `memcpy` to avoid UB; many production kernels do the packed cast.

**Expected gain.** Removes ~half the load instructions in the hot GEMM loop. Load-issue cost is
currently the dominant component (≈ 10 of 29 instrs/8-MAC-iter). Combined with §1/§4, GEMM inner
→ ≈1.9–2.3 instr/MAC. Independent estimate for GEMM block: ≈2.2 s → ≈1.2–1.4 s.
**Accuracy risk: none** — bit-identical products as the current scalar `(int16)(int16)` loads;
only the way the two values land in registers changes.

---

## 3. Mixed 16/32-bit accumulate — avoiding int64 (QK, PV, GEMM)

**What.** Keep the *accumulator* int32 and the *products* int32 (16×16→32) everywhere; the only
reason the QK dot uses int64 today is the theoretical worst case `32×(2^15)² = 2^35`. The right
fix is not wider accumulation but *measured, model-specific* overflow margins + a small number of
independent int32 accumulators whose union is folded to one int64 only at the end of the dot.

**Measured margins (this model, fp32-torch walk of the full quantized pipeline, 25 seeds 1234..1258;**
i.e. real activation sequences through LN→Q/K/V→attention→oproj→FFN, with per-buffer Q15 and per-matrix
Q12 quantization identical to the C pipeline — see methods note at the end):

| accumulate site | dims | worst \|Σ\| over 25 seeds | 2^31 fraction | verdict |
|---|---|---|---|---|
| QK full dot (one int32 acc) | 32 | 2.83e9 (8 overflow events) | 1.32 | **unsafe** |
| QK lane-4 (8 dims / int32 acc) | 4×8 | 1.76e9 | 0.82 | **safe** (18% margin) |
| QK lane-8 (4 dims / int32 acc) | 8×4 | 1.90e9 | 0.88 | safe (thin) |
| PV full sum (int32, raw Q15) | 128 | 1.07e9 | 0.50 | safe (matches 32767² bound) |
| GEMM Q15×Q12 K=128 (one int32 acc) | 128 | 0.63e9 | 0.29 | safe |
| GEMM partial K=16 (sub-accumulator) | 16 | 0.25e9 | 0.12 | safe |

Notes:
- QK int64 is currently justified by the *theoretical* 2^35 bound, but on this model's data the
  empirical max |dot| = 2.83e9 with tiny overflow tail; **4-lane int32 is safe with 18% margin**
  for the whole seeded corpus. A cheap belt-and-braces: after the fold, clamp/one-time int64 reassign
  only if a lane saturates (never observed here) — or scale Q15 by 0.9 in the fixed exp path.
- PV: `p15·v16 ≤ 32767·32767 = 1.0737e9` is provable after the kernel's `f15` rescale (Σp′ ≡ 32767),
  and the measured worst sum ≈ the single-product bound ⇒ **int32 PV is unconditionally safe**, no
  measurement needed. (The current code already does this — this documents *why* it is correct.)
- GEMM: K=128 theoretical max `128·2^15·2^11 = 8.6e9` exceeds int32, but the *model's* Q15 activations
  and Q12 weights are not all at max; measured worst 0.29·2^31 with 0 failures across the corpus.
  Guard cheaply: split K into 2×64 or 4×32 sub-accumulators (still int32, returned to int64 only in
  the dequant epilogue) — couple of int64 adds total per output, not per MAC.

**Evidence.**
- TFLite Micro reference FC (above): int32 `BiasType` accumulator + single `MultiplyByQuantizedMultiplier`
  requant — the industry-default way to avoid int64 per-MAC.
- CMSIS-DSP `arm_mat_mult_fast_q15.c`: int32 accumulator, `sum >> 15` per output (faster, "less
  accurate" variant) — proof that 32-bit accumulate with q15-normalized inputs is the accepted
  default; the _fast_ variant additionally truncates products *per step* to control range, which is
  the aggressive version of the same idea.
- I-BERT (arXiv:2101.01321): all nonlinearities (GELU, Softmax, LayerNorm) computed **in INT32**
  with no accuracy loss vs fp32 — the rest of this pipeline already follows that recipe
  (exp/GELU LUT, int-LN); making QK accumulation int32 completes the "no int64 in the hot path"
  pattern that I-BERT and TFLM both use.

**Where for this model.**
- QK `attn_head`: replace `int64_t dot` + 16-add loop with `uint32_t dot0..dot3`, 8 dims each
  (d-block of 4 pairs), fold `dot = dot0+dot1+dot2+dot3` once per (i,j). Removes `mulh` and the
  64-bit carry ADDs from the inner loop (~9 → ~4 instr/dim) and restores register pressure for
  §4 pipelining.
- GEMM epilogue: keep the current Q30 int64 dequant *outside* the loop (already the case); if
  sub-accumulators are introduced, fold them to one int64 before the epilogue multiply.

**Expected gain.** QK is ≈0.3–0.4 s of the attention block today (int64 two-word adds dominate its
inner loop); int32-4-lane + paired loads (§2) + pipelining (§4) → ≈0.15–0.2 s. GEMM already int32;
no direct gain there from this item, but it removes the *reason* anyone might split PV/GEMM loops.

**Accuracy risk.** None for the gate: 4-lane fold reproduces the int64 dot exactly (no saturation
observed; a `|lane| > 2^31` check with an int64 fallback is a 4-instruction guard per dot if we want
zero surprise). PV/GEMM int32 folding is exact for the observed corpus.

---

## 4. Unrolling + software pipelining to hide LHU latency

**What.** On a 4-stage in-order scalar core, every `lhu`/`lw` result is used by a MUL/ADD that the
pipeline cannot reorder past: the load→use distance (2–3 cycles, zero-wait SRAM but still
pipeline-stage latency) stalls the loop unless ALU work from a *previous* iteration fills the slots.
Software pipelining = load k+1's A and B words before issuing k's MACs; with ≥8 independent
accumulators the compiler can usually do this if the loop is unrolled ×2 (k-pair) with
`a_next`/`b_next` temporaries.

**Evidence.**
- ESP32-C3 TRM (above): 4-stage, in-order ⇒ no OoO to hide stalls; the *programmer/compiler* must.
- CMSIS-NN kernels are written with exactly this shape: `read_q15x2_ia` loads are batched at the top
  of each loop body, then 4 `__SMLAD`s consume them with zero load→use adjacency — the source
  explicitly separates loads from MACs.
- CMSIS-DSP `arm_mat_mult_q15.c` non-DSP path unrolls the k-loop ×4 (`colCnt = numColsA >> 2`),
  the classic "unroll to give the pipeline independent work" move; its DSP path relies on the paired
  loads + dual MAC doing the same.
- muRISCV-NN keeps the identical schedule for RVV-less builds.

**Where.** Both the GEMM inner loop and the QK d-loop. Concretely for QK: prefetch `lw` of qi[k+2/4],
kj[k+2/4] pairs while MACing the current pair into 4 int32 lanes. For GEMM: k-pair unroll where the
`lw` of A[k+1], B[k+1] happens before the `mul`+`add` of k.

**Expected gain.** Removes most load-use stalls. Current 7.7 cyc/MAC vs ~3.6 instr/MAC ⇒ ~2.1 cyc
overhead per instr attributable to stalls + static hazards; a clean ×2-pipelined loop typically
removes 60–80% of those. Realistic: GEMM inner → ~4.5–5.5 cyc/MAC, aggregate forward → ≈3.1–3.4 s
if applied only to GEMMs. **Accuracy risk: none** (pure scheduling).

**Implementation sketch (QK, int32-4-lane + paired, pipelined):**
```c
/* d-loop unrolled 2 steps ahead; qi/kj are the lw'd 32-bit pairs */
uint32_t qp = *(uint32_t*)&qi[0], kp0 = *(uint32_t*)&kj[0];
for (d = 2; d < HD; d += 2) {
  uint32_t qn = *(uint32_t*)&qi[d];      /* preload next before using cur */
  uint32_t kn = *(uint32_t*)&kj[d];
  lane0 += (int32_t)(int16_t)qp * (int32_t)(int16_t)kp0;
  lane1 += (int32_t)(qp >> 16) * (int32_t)(kp0 >> 16);
  lane2 += (int32_t)(int16_t)qn * (int32_t)(int16_t)kn;
  lane3 += (int32_t)(qn >> 16) * (int32_t)(kn >> 16);
  qp = qn; kp0 = kn;
}
dot = (int64_t)lane0 + lane1 + lane2 + lane3;
```
(4 lanes × 8 dims = 32; the preload of `qn/kn` 1 step early hides the LHU→MUL distance.)

---

## 5. Published embedded int16 GEMM kernels to steal from

1. **CMSIS-DSP `arm_mat_mult_q15.c` / `arm_mat_mult_fast_q15.c`** — transpose-B, `read_q15x2_ia`,
   `__SMLAD` dual-MAC, 2×2 micro-tile, int32 accumulator, final `>>15`. Primary source captured.
   URL: github.com/ARM-software/CMSIS-DSP (Functions/MatrixFunctions).
2. **TFLite Micro `reference/integer_ops/fully_connected.h`** — int16 (and int8) FC with int32
   accumulate + `MultiplyByQuantizedMultiplier` requant; also the int16 softmax used here (513-entry
   LUT). Primary source captured. URL: github.com/tensorflow/tflite-micro.
3. **muRISCV-NN (tum-ei-eda)** — the CMSIS-NN structure transplanted to RISC-V (RVV/P only, but the
   pack/dual-MAC scaffolding and 2×2 tile map 1:1 to RV32IMC manual sequences). Captured.
   URL: github.com/tum-ei-eda/muriscv-nn.
4. **I-BERT (Kim et al., arXiv:2101.01321)** — end-to-end integer-only transformer; INT32 GELU/
   Softmax/LayerNorm, no fp32 — the accuracy-preservation reference for every "make it int32"
   change here.
5. **RISC-V P-extension spec (riscv-p-spec, github.com/riscv/riscv-p-spec)** — documents the packed
   dual-MAC instructions (PMULH.H, PMACC.W.H00, KADD16, PSADD.H). Not available on RV32IMC, but
   defines the exact instruction budget a hand-packed kernel is emulating (2 MUL+1 ADD per 2 MAC).

---

## 6. Implementation order + expected headroom

| step | change | est. GEMM block Δ | est. forward Δ | risk |
|---|---|---|---|---|
| A | §3 QK → 4-lane int32 (drop mulh/int64 adds) | — | −0.15–0.2 s | none (measured safe) |
| B | §2+§1 GEMM: K-pair `lw` + j-tile 2 (16 accs) | −0.5–0.7 s | −0.5–0.7 s | none |
| C | §4 software-pipelined ×2 k-loop on GEMM+QK | −0.2–0.4 s | −0.2–0.4 s | none |
| D | Transpose-B Wᵀ blocks to SRAM per layer | −0.2–0.4 s | −0.15–0.3 s | none (32 KB block fits) |

Honest bottom line: kernel-MAC floor ≈0.71 s means even a perfect kernel leaves ~0.7–0.9 s of
GEMM/attention plus LN/GELU/epilogues (~0.5 s) — i.e. **realistic best ≈1.3–1.6 s/forward
(~2.5–3×)**, and A+B+C together plausibly reach ~2.5–3.0 s without touching numerics beyond what
was measured safe above. A and B are the safest and give >60% of the win; do them first.

---

## Methods note (for the magnitude table)

Numbers come from a Python (torch) re-walk of the *exact* case-2 pipeline: fp32 weights init
(seed 1234), input seeds 1234..1258, per-buffer Q15 (amax/32767) and per-matrix Q12 (2047/amax)
quantization identical to `src/`, messages propagated through all 4 layers including residual adds;
QK/PV/GEMM integer accumulations are computed in int64 in the emulator solely to *measure* what an
int32 accumulator would hold. Worst values are max-abs over the full corpus. Reproducer kept at
`/tmp/jam26/quant_mag3.py` (not in the repo; the export/verify scripts in `tools/` remain
authoritative for the gate itself).

## Citations
- Espressif, ESP32-C3 Technical Reference Manual v1.4 (§1.1 CPU, §3.1 System and Memory).
- ARM CMSIS-DSP `arm_mat_mult_q15.c` / `arm_mat_mult_fast_q15.c` (raw source captured to /tmp/jam26).
- TensorFlow Lite Micro `reference/integer_ops/fully_connected.h` (raw source captured).
- tum-ei-eda/muRISCV-NN (README/source captured).
- riscv/riscv-p-spec (P-extension packed-SIMD spec captured).
- S. Kim et al., I-BERT: Integer-only BERT Quantization, arXiv:2101.01321.
