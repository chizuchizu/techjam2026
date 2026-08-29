# 02 · GEMM register tiling + scale-exact int GELU + quant/attn micro-opts

Date: 2026-08-29 · Series of small kernels landing between 13.70 s and 6.91 s.

## Goal
After opt 1+2 the profile looks like (per forward, FAST, device):

| section | s/fwd | % of 6.91 s | note |
|---|---|---|---|
| attention | 1.21 | 17.6% | qk 0.29 / exp 0.40 / pv 0.52 (inner) |
| QKV trio (3 gemms + 3 quant_head) | 0.53 | 7.7% | outside attn, incl. per-head int16 quantize |
| quant (shared A-quantize, 4) | ~0.24 | 3.4% | |
| oproj GEMM | 1.05 | 15.2% | Q15×Q12, N=128 |
| f1 GEMM | 1.05 | 15.2% | |
| f2 GEMM | 0.75 | 10.9% | fused amax+quant+gelu |
| GELU (int LUT) | 0.27 | 3.9% | inside f2 span |
| LayerNorm (fp32 2-pass) | 0.72 | 10.4% | |
| res1/res2/final | 0.29 | 4.2% | |

GEMM block (QKV+oproj+f1+f2 ≈ 39.8 M MACs) dominates. Target ~20 cyc/MAC
was measured for the scalar int32 loop; flash-XIP weight reads are the
limiter (weights live in the 3 MB app partition, cached through the 16 KB i-cache).

## Opt 3 — GEMM register tiling (core → core2 → core3)

Each MAC in the int16×int16 path needs 2×`l.h` (A, W) + `mul` + `add`; W rows
are re-fetched from flash once per output row with the plain kernel.

- **core2**: 2-row i-tile + 2-way j-unroll → 8 registers of int32 accumulators,
  8 MACs per inner step, and **2× fewer weight bytes** fetched from flash
  (each W element read once for 2 output rows).
- **core3**: 4-row i-tile + 2-way j-unroll → 4× fewer weight bytes.
  Register pressure is manageable on RV32 (32 regs): ~8 acc + 2 b + 2 a live.

All four fast-path calls (QKV, oproj, f1, f2) switched to core3.
Measured: 8.51 s → 7.85 s (core2 swap, other fixes in build) → final step
to 6.91 s with the micro-opts below; the f2 span dropped 0.91 → 0.75 s.

## Opt 4 — scale-exact integer GELU with runtime LUT (`tm_gelu_q15_lut`)

fp32 GELU (deg-11 poly, 1.43 s) was 3.4% of baseline. A fixed LUT fails:
`gelu(x) = 0.5·x·(1+erf(x/√2))` needs x in **real units**, but a LUT indexed
by `v16 = Q15(x)` only knows `x = v16·amax/32767` — the erf argument scales
with the *runtime* activation amax for the layer, so a fixed table cannot be
exact (measured error up to 2.04 raw units / 5569 Q15 units, worst on the
negative side, where `gelu(-amax) ≈ -amax/2`). Sigmoid/tanh approximations
max-error 0.0203 → exceeds the post-f2 2e-3 gate. Rejected both.

**Working design**: per layer, rebuild a **513-entry LUT from the actual amax**
(≈ 4×~67 ms total device, 0.27 s/fwd — 5.3× faster than fp32):
`u = -1 + 2k/512 = x/amax;  lut[k] = round(0.5·u·(1+erf(u·amax/√2))·32767)`
indexed by 7-bit linear interp of Q15(x). Output stays *on the input Q15
grid* (|gelu| ≤ |x|), so the fused f2 path is:
```
sa2 = tm_gemm_amax(g_buf2);        // 1 pass
tm_gemm_quantA_into(a2, ..., sa2); // quant
tm_gelu_q15_lut(a2, TM_S*TM_F, TM_QACT_MAX/sa2);
tm_gemm_core3(a2, 1.0f/sa2, W2q, ..., g_buf1);
```
GELU slot: 1.43 s → 0.27 s. Host max_abs after: ~6.4e-4–7.9e-4 (gate 2e-3).

## Opt 5 — quant & attention micro-opts
- `quant_head`: replace soft-float `llrintf` with `(int)(v ± 0.5)` fast
  rounding + clip (quant slot ≈ 1.0 → 0.83 s).
- QK dot: two int16 products packed per int32 add (dot = 16 int64 adds/dim).
- PV: int32 product + int64 accumulator add (`attn` 1.45 s → 1.21 s);
  later restructured to 4 parallel register accumulators (d-block of 4) to
  avoid an int64 acc-array spill — cleaned up after the first (slower)
  transpose attempt measured at 7.00 s vs 6.91 s with the register version.
- PB(P_F2) fix: a stale PE pair corrupted the f2 slot; renamed so f2 timing
  is truthful.

## Opt 6 — fused QKV quantization (Q30 fixed-point, no fp32 staging)

The Q/K/V per-head path was: 3× `core3 → g_buf2(see fp32 staging)` +
`quant_head` (soft-float amax + quantize over each 128×32 slice),
contributing ~0.63 s (incl. the fp32 GEMM store + re-read + the fp32→int16
quantize). Replaced with `tm_gemm_head_q15`:
- **Pass 1** gemms straight into an int32 accumulator scratch (reuses `g_buf1`,
  which is free during the QKV loop — the attention output accumulates in
  `g_buf2`; per-head acc slice fully consumed before the next head), tracking
  the *global* real-value amax via fp32 fold `acc·g + bias` per output so the
  dequant scale matches `quant_head` exactly (single per-buffer scale → no
  attention changes).
- **Pass 2** fixed-point quantize: `q15 = round((acc·GX + BX[d])/2^30)` in
  int64 with **Q30 coefficients** — a Q15 coefficient loses precision when acc
  is large (`acc·ΔGX` grows with acc; first attempt at Q15 failed host with
  0.22 max_abs, fixed by Q30). Handles acc near 2^31 safely.
- Gotcha: two scratch aliasing bugs — a 16 KB static overflowed DRAM, and
  `g_buf2` was wrong (it accumulates attention outputs across heads); `g_buf1`
  is the correct alias. Host 50/50 + device 5/5 after fix, FAST max_abs
  actually improved (~6.4–9.5e-4).

### Result
| metric | before (opts 3–5) | + opt 6 |
|---|---|---|
| QKV slot (incl. attn) | 1.75 s | 1.39 s |
| forward | 6.91 s | **6.56 s** |
| vs baseline | 6.10× | **6.43×** |


## Opt 7 — all-integer exp LUT index (attention exp 0.40 → 0.05 s)

Before: the FAST exp pass converted the integer dot to the LUT index via
`logit = (float)diff * gsc` then `logit * 6553.5f` — two soft-float
multiplies **and** an int64→float conversion **and** a float→int conversion
per element (~450+ cycles). Now a single fixed-point multiply per element:

- `g_exp_c = round(gsc * 6553.5 * 2^32)` computed once per head.
- `mag = |score - maxs|` (int64) → `y = (mag * g_exp_c) >> 32` = |y16|,
  exactly the trunc-to-zero the fp32 path produced, then the same LUT
  interpolation.
- Precision: the LUT domain caps |diff·gsc·6553.5| ≤ 65535, so
  `mag·g_exp_c < 2^47` — uint64-safe. Q32 coefficient error is ≤ 0.5/2^32
  relative → sub-unit y16 error where it matters (near the max logit).
- First attempt used **Q24**: gsc = g_qs·g_ks·0.1768 varies over ~1e-9..1e-3
  (NOT the ~0.18 of the raw attention scale), so the small-shift coefficient
  lost precision and FAST jumped to 0.22–0.42 max_abs (10k+ fails). Q32 fixed
  it (host 50/50, worst ≈ 1.1e-3).
- Measured: attn_exp 194.8 → **23.2 µs/call** (7×), attention total
  1.22 → **0.85 s**.

## Opt 8 — core4 GEMM: j-outer, 8-row i-tile (oproj/f1/f2)

Audit found oproj and f1 were still calling the **original per-element
`tm_gemm_core`** (4-way k-unroll, one output at a time) — only f2 had been
moved to the tiled core in opt 4. That is why oproj/f1 stayed ~20 cyc/MAC
while f2 dropped to ~9.

`tm_gemm_core4` inverts the traversal: j-outer, then an 8-row i-tile inside,
with 8 int32 register accumulators. Each flash weight column is read once per
8 rows → weight-bytes per gemm = M·N·K·2/8 (half of core3's i4j2), at the
cost of re-reading A (SRAM) per output column. Routed the K=N=128 gemms
(oproj via tm_gemm_q12, f1 via tm_gemm_q12, f2) to core4.

- Only 8 accs fit RV32 registers; i8j1 avoids the 16-acc spill a 4×4 tile
  would force. The int32 accumulation order is unchanged → identical
  results to core3 (verified: all results bit-equal on the host).

### Result
| metric | opt 6 ('6.56 s build') | + opts 7–8 |
|---|---|---|
| attention (QK+exp+PV) | 1.22 s | 0.85 s |
| oproj | 1.05 s | 0.61 s |
| f1 | 1.05 s | 0.61 s |
| f2 | 0.75 s | 0.68 s |
| forward | 6.56 s | **5.27 s** |
| vs baseline | 6.43× | **8.0×** |

## Trajectory (device, seed-0 FAST, wall)
| build | s/forward | vs baseline |
|---|---|---|
| baseline fp32 | 42.15 | 1.00× |
| + integer attention (opt 1) | 15.21 | 2.77× |
| + exp LUT (opt 2) | 13.70 | 3.08× |
| + GEMM/gelu/quant/attn (opts 3–5) | 6.91 | 6.10× |
| + fused QKV quantization (opt 6) | 6.56 | 6.43× |
| + integer exp index + core4 GEMM (opts 7–8) | **5.27** | **8.0×** |


Gate throughout: host 25/25 FAST + 25/25 EXACT (all-zero-filtered), and after op 7's Q24 misstep the Q32 version passed host 50/50; the shipping 5.27 s build (TM_PROFILE off) passed device seeds 0–4 5/5, worst max_abs 9.414e-4 (gate 2e-3), RAM 270,860 B (82.7%), Flash 2,629,884 B (83.6%).
(An intermediate build with broken LN-pair precompute measured 18.6 s f2
slot during an 8.5 s wall — the slot-vs-wall sanity check caught the OOB
write; reverting LN to the single-pass form restored clean numbers.)
