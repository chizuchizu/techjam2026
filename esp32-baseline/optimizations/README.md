# esp32-baseline/optimizations — performance engineering log

Goal: make the case-2 transformer forward as fast as possible on the ESP32-C3
(160 MHz RV32IMC, no FPU, 320 KB SRAM) while keeping the benchmark gate
(|Δ| ≤ 0.002 OR |Δ| ≤ 0.02·|ref| per element vs the fp32 torch reference).

Every entry: what, why, host + device measurements, gate results, flash/RAM cost.

## Result (device, seed-0 FAST, per forward)
| build | s/forward | speedup | gate |
|---|---|---|---|
| baseline fp32 | 42.15 | 1.00× | 25/25 host, 5/5 device |
| integer attention (opt 1) | 15.21 | 2.77× | 25/25 + 25/25 EXACT |
| + exp LUT (opt 2) | 13.70 | 3.08× | 25/25 + 25/25 EXACT |
| + GEMM tiling / int GELU / quant+attn micro-opts (opts 3–5) | 6.91 | 6.10× | host 50/50, worst § 8e-4 |
| + fused QKV int16 quantization (opt 6) | **6.56** | **6.43×** | host 50/50, device 5/5, worst § 9.5e-4 |

## Files
| file | contents |
|---|---|
| 00_baseline_profile.md | baseline compute + memory profile (start here) |
| 01_integer_attention.md | integer QK + exact-max softmax + int PV + exp LUT |
| 02_gemm_gelu_quant.md | GEMM register tiling, scale-exact int GELU, quant/attn micro-opts, fused QKV int quantization |
| research.md | firecrawl/web literature skim (research/techniques, kept at /tmp/jam26_opt) |

## Key techniques applied (performance-engineer view)
1. Q15 activation × Q12 weight int16×int16 GEMM (39.8 M MACs) with the
   per-buffer Q15 scale (sq·sk·scale) folded into a single fp32 multiply —
   only a handful of soft-float ops remain, all outside the hot loops.
2. Register-tiled GEMM (i-tile 4 × j-unroll 2, 8 int32 accumulators) to cut
   flash-XIP weight bytes 4× (GEMM block 4.7 → ~3.4 s).
3. Exact-max two-pass softmax in integer QK logits + 513-entry interp LUT
   for exp (attention 30.09 → 1.21 s).
4. Scale-exact runtime-LUT GELU fused into the f2 quantize (fp32 gelu
   1.43 → 0.27 s) — fixed LUTs are mathematically impossible here because the
   erf argument needs the layer's runtime amax.
5. Soft-float `llrintf` → fast `(int)(v±0.5)` rounding; PV with 4 register
   accumulators; count correctness (PB/PE pairing) kept exact after fixing a
   stale PE that corrupted the f2 slot.
6. Fused QKV quantization: the per-head GEMM stores int32 accumulators
   (pp. scratch = g_buf1, free during the QKV loop) + fixes a global Q15 scale
   in Q30 fixed-point int64 — replaces the fp32 ctx staging + soft-float
   quant_head and drops the QKV slot 1.75 → 1.31 s without touching attention.

## Final device validation (TM_PROFILE off)
```
[device] TM 1 128 128
seed 0: fails=    0 max_abs=8.1015e-04 6.558s fwd PASS
seed 1: fails=    0 max_abs=9.4807e-04 6.561s fwd PASS
seed 2: fails=    0 max_abs=6.8700e-04 6.559s fwd PASS
seed 3: fails=    0 max_abs=7.3624e-04 6.559s fwd PASS
seed 4: fails=    0 max_abs=6.5146e-04 6.556s fwd PASS
ALL PASS
RAM 270,860 B (82.7% of 320 KB) · Flash 2,629,884 B (83.6% of 3 MB)
```