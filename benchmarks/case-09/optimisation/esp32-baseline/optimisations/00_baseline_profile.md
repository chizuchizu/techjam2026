# Baseline profile — case 2 (B=1 S=128 D=128 H=4 F=128 L=4)

Date: 2026-08-29 · Branch `optimize-attention` · FW = stock `competition-cleanup` code + on-device per-kernel timers (TM_PROFILE, esp_timer µs).

## Compute profile (on-device, ESP32-C3 160 MHz, FAST mode, input seed 0)

Measured with per-kernel timers around each section in `tm_forward`; two full forwards; values are per-forward.

| section | s/forward | % of 42.15 s | notes |
|---|---|---|---|
| **attention (attn_head: QK+softmax+PV)** | **30.09** | **71.4%** | fp32 soft-float throughout: QK dequant mults, 512k exps, PV mults |
| GEMM QKV (3 Q15×Q12) | 5.76 | 13.7% | scalar int32 inner loop |
| GEMM oproj | 1.23 | 2.9% | |
| GEMM f1 | 1.23 | 2.9% | |
| GEMM f2 | 1.23 | 2.9% | |
| GELU (fp32 deg-11 poly) | 1.43 | 3.4% | runs on F-length buffer, then F2 re-quantizes → foldable |
| quant_head (Q15 per-head slices) | 0.80 | 1.9% | could merge into GEMM quant |
| LayerNorm (fp32, all) | 0.82 | 1.9% | |
| residual adds | 0.09 | 0.2% | |

**Kernel-level facts**
- Total MACs/forward ≈ 39.8 M (4 layers × (3·S·D·HD + 3·S·D² + S·D·F)). 
- GEMMs run at ~**36 cycles/MAC** measured (ideal int32 MAC loop would be ~5–8).
- Every fp32 op in attention is a soft-float library call on this no-FPU core
  (Espressif reference: C3 soft-float add ≈ 100 cyc, div ≈ 102 vs HW-FPU ≈ 25/69).

## Memory profile

| metric | value | limit |
|---|---|---|
| SRAM used (static arena + firmware) | 268,220 B (**81.9%**) | 327,680 B (320 KB) |
| Flash used (app partition) | 2,623,978 B (**83.4%**) | 3,145,728 B (3 MB partition) |
| embedded weights.bin (fp32) | 1,594,368 B | |
| embedded weights_q12.bin (Q12 int16) | 786,624 B | |
| core stats | 160 MHz RV32IMC, no FPU, 16 KB i-cache / 8 KB D-cache | |

Static arena in model.c/kernels.c (per forward, single-buffered):
- g_x, g_buf1, g_buf2: 3 × 64 KB fp32 = 192 KB
- g_qh/g_kh/g_vh: 3 × 8 KB int16 Q15 = 24 KB
- a16 (tm_gemm_q12 scratch): 32 KB int16
→ ~248 KB model workspace; remaining RAM = firmware statics.

## Baseline correctness (pre-optimization)
- Host 25 seeds FAST: all PASS (0 failures), worst |Δ| ≈ 8e-4 (gate atol=0.002 / rtol=0.02).
- Device seed0-4 FAST: 5/5 PASS, measured 42.15 s/forward (this build).

## Strategy (target order)
1. **attention → integer QK + fixed-point softmax/PV** (attack 71.4%).
2. **exp via LUT / bit-trick** (reduce soft-float exp count inside attention).
3. **GEMM inner-loop unroll + 32-bit loads** (attack ~36 cyc/MAC).
4. **fold GELU into F2 quantization** (saves a full fp32 pass + merge).
5. LN integerization only if gate margin allows.

Baseline numbers committed here so every optimization below can be compared
against this same measurement method.
