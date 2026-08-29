# Case-02 single-board optimisation log

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
| + GEMM tiling / int GELU / quant+attn micro-opts (opts 3–5) | 6.91 | 6.10× | host 50/50, worst 8e-4 |
| + fused QKV int16 quantization (opt 6) | 6.56 | 6.43× | host 50/50, device 5/5, worst 9.5e-4 |
| + integer exp index, core4 GEMM (opts 7–8) | 5.27 | 8.0× | host 50/50, device 5/5, worst 9.4e-4 |
| + j-outer head GEMM, integer amax (opt 8b) | 4.862 | 8.67× | host 50/50, worst 9.4e-4 |
| **+ fused LN→Q15 amax-bound (opt 9)** | **4.784** | **8.82×** | host 50/50 (FAST+EXACT), worst 1.14e-3 |
| + integer LN pass: int-stats, direct Q15 emit (opt 9b) | 4.784 → 4.160 | 10.13× | host 50/50, worst 1.19e-3; device 5/5 |
| **+ oproj Q15-ctx fusion: two-phase V proj + one K=128 core4 (opt 10)** | **4.160 → 3.969** | **10.62×** | host 50/50 FAST+EXACT, worst 1.12e-3; device 5/5, worst 1.14e-3 |
| + integer-only attention PV + integer ctx epilogue (opt 11) | 3.969 → **3.688/3.706** | ~10.7× | host 50/50, worst 1.20e-3; device 5/5, worst 1.11e-3 |
| + core4_v2 GEMM j-tile-2 + K-pair prefetch (opt 12) | 3.706 → **3.664** | ~11.5× | host 50/50, worst 1.20e-3; device 5/5 |
| + integer LN pass, int amax+quant (opts 13–14) | 3.664 → **3.205** | ~13.1× | host 50/50, worst 1.27e-3; device 5/5, worst 1.088e-3 |
| **+ core5 GEMM: j-tile-2×IBLK=4 (opt 15)** | **3.205 → 2.982** | **14.1×** | host 50/50, worst 1.277e-3; device 5/5, worst 1.088e-3 |
| **+ head_q15 8-MAC hand-asm (opt 16)** | **2.982 → 2.838** | **14.8×** | host 50/50, device 5/5, worst 1.088e-3 (bit-exact) |
| **+ FFN1 fixed-point Q15 epilogue (opt 17)** | **2.838 → 2.701** | **15.6×** | host 50/50 (9.05e-4..1.12e-3), device 5/5 (9.2e-4..1.1e-3) |
| **+ KB1 int32 bias-fold + asm requant, core5 bias-fold, int32-limb QK (opt 18)** | **2.701 → 2.447** | **17.2×** | host 50/50 (FAST ≤9.8e-4, EXACT ≤6.8e-5); device 5/5 (≤1.29e-3) |
| **+ QK j-unroll-4, PV 8-accumulator (opt 19)** | **2.447 → 2.386** | **17.6×** | host 50/50 (≤9.99e-4); device 5/5 (≤1.29e-3), bit-exact attention |
| **+ integer-residual FAST path (opt 21)** | **2.386 → 2.122** | **19.8×** | host 50/50 (worst 1.03e-3); device 25/25 test seeds PASS (worst 1.24e-3), ExScore 0.267 → 5.30 |
| **+ KB0 head-GEMM asm on R1 (opt 22)** | **2.122 → 2.056** | **20.5×** | host 50/50 (worst 1.03e-3); device 25/25 PASS (worst 1.24e-3, kb0 bit-exact), ExScore 5.48 |
| **+ core5 4×2 asm fix (opt 23, col1 product-reuse)** | **2.056 → 1.996** | **21.1×** | host 54/54 (worst 1.03e-3); device 25/25 PASS (worst 1.24e-3); probe bad=0; ExScore 5.65 |

## Files
| file | contents |
|---|---|
| 00_baseline_profile.md | baseline compute + memory profile (start here) |
| 01_integer_attention.md | integer QK + exact-max softmax + int PV + exp LUT |
| 02_gemm_gelu_quant.md | GEMM register tiling, scale-exact int GELU, quant/attn micro-opts, fused QKV int quantization, integer exp index, core4 GEMM |
| 03_layernorm_fused_quant.md | fused LN→Q15 amax-bound + int-stats Q15 emit (removes a16 amax+quant passes) |
| 04_oproj_ctx_fusion.md | two-phase V projection, per-layer global ctx Q15 scale, single K=128 core4 oproj |
| 11_int32_attention_pv.md | integer-only PV (Q15 row-rescale + int32 acc) + integer ctx epilogue (m/2^sh) |
| 12_gemm_jtile2.md | core4_v2 GEMM: j-tile-2 + K-pair prefetch, bit-exact vs core4 |
| 18_kb1_core5_qkv2.md | KB1 int32 bias-fold + asm requant; core5 bias-fold epilogue; int32-limb QK |
| 23_core5_asm_fix.md | core5 4×2 asm col1 product-reuse fix + probe `1:` label repair (2.056 → 1.996 s) |
| 19_attn_qk_unroll_pv8.md | attention QK j-unroll-4 (3.25 instr/MAC) + PV 8 accumulators; core5-flash & IRAM negative results |
| 21_integer_residual_fast_path.md | int32 exact residual + fused fixed-point epilogues + integer-stats norms (R1): 2.386 → 2.122 s, −11%, no new RAM |
| 22_r1_kb0_composition.md | R1 + kb0 head-GEMM asm merge: 2.122 → 2.056 s, device 25/25 PASS |
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
7. All-integer exp LUT index: one Q32 fixed-point multiply per element
   (replaces int64→float + 2 fp32 muls + float→int) — attn_exp 194.8 → 23.2 µs
   call; Q24 was lossy (gsc ≈ 1e-9..1e-3), Q32 is exact.
8. core4 GEMM (j-outer, 8-row i-tile, 8 accs): halves flash weight reads vs
   core3. Also fixed the discovery that oproj/f1 were still on the legacy
   per-element core (only f2 had the tiled core) — oproj/f1 1.05 → 0.61 s.

## Final device validation (TM_PROFILE off)
```
[device] TM 1 128 128
seed 0: fails=    0 max_abs=1.1065e-03 2.3839s fwd PASS
seed 1: fails=    0 max_abs=1.0107e-03 2.3840s fwd PASS
seed 2: fails=    0 max_abs=1.0414e-03 2.3836s fwd PASS
seed 3: fails=    0 max_abs=1.2925e-03 2.3838s fwd PASS
seed 4: fails=    0 max_abs=1.0145e-03 2.3834s fwd PASS
ALL PASS
RAM 270,860 B (82.7% of 320 KB) · Flash 2,629,884 B (83.6% of 3 MB)
```
