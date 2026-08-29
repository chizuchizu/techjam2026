# 01 · Integer attention (QK + fixed-point softmax + PV)

Date: 2026-08-29 · Commit chain on `optimize-attention` branch.
Target: attention was **71.4%** of the baseline forward (30.09 / 42.15 s).

## Why

Every float op on this RV32IMC (no-FPU) core is a soft-float library call
(C3 soft-float add ≈ 100 cyc, div ≈ 102; HW FPU parts do 25 / 69).
Per forward, attention ran 16.7 M QK dots (each an fp32 dequant-multiply over
HD=32), 1 M softmax exps (`expf` soft-float), and 8.4 M PV MACs — all in fp32.
`riscv32-esp-elf-gcc -S` checks confirmed int16×int16 → int64 dot compiles to
`mul`+`mulh` + two-word adds (~9 instr/dim), i.e. ~5× fewer cycles than the
float path. (Red flag: `riscv32-esp-elf-objdump` mis-decodes this ELF/ISA pair
and is untrustworthy for timing claims.)

## Opt 1 — all-integer QK dot + two-pass exact-max softmax + int PV

- **QK**: `s = Σ_d q16[d]·k16[d]` accumulated in `int64` (Q15·Q15 ≤ 2^30;
  32 terms could reach 2^35, so int64 is required). Dequant (`sq·sk·scale`)
  folded into one fp32 multiply per logit — the only float op left per score.
- **softmax**: two-pass with an *exact* max (no online-rescale drift):
  pass 1 computes all scores + max; pass 2 `diff = score − max` then exp.
- **PV**: `acc[d] += p15 · v16[d]` in `int64`; one fp32 dequant per row-end
  (`o[d] = acc[d]·(sv/lsum15)`), zero float ops in the inner accumulation.

### Result (device, seed 0, FAST)
| metric | baseline | + int attention | Δ |
|---|---|---|---|
| forward | 42.15 s | 15.21 s | **2.77×** |
| attention | 30.09 s | 3.15 s | **9.5×** |

Gate: host 25/25 FAST PASS, 25/25 EXACT PASS after this change.

## Opt 2 — integer exp via 513-entry LUT (TFLM-style)

`p15 = exp(diff·gsc)` replaced by a precomputed `int16 g_exp_lut[513]` on
`[-10,0]` (9-bit index = `idx = vv>>7`, 7-bit linear interp `off = vv & 127`),
the same structure TensorFlow Lite Micro uses for its int16 softmax.

**Bug found during bring-up (kept for the log):** the first scheme scaled the
LUT entry with `M = round(gsc·6553.5·32768)` and `y16 = diff·M >> 15`. It fails
catastrophically (max_abs ≈ 1.2, 15 k gate failures) because `gsc ≈ 1e-9`
(logits are `sq=amax/32767` tiny) while Q15 `diff` values span 1e8–1e9 — M
rounds to 1 and the whole mapping collapses. Fix: `logit = (float)diff·gsc`
(ONE soft-float multiply handles the 15-order dynamic range),
`y16 = (int)(logit·6553.5)` clipped to [-65535,0], then pure-integer LUT
interp. EXACT path keeps `expf`.
Debug note: the original probe printed nothing because `(i==0 && j==1)` never
executes with causal masking; moved to `(i==2 && j==1)`.

### Result
| metric | + int attention | + exp LUT |
|---|---|---|
| forward | 15.21 s | **13.70 s** |
| attention | 3.15 s | 1.64 s |

Gate after all of opt 1+2: host FAST worst |Δ| ≈ 1.05e-3 (atol 0.002),
25/25 FAST + 25/25 EXACT PASS; device seeds 0–4 PASS.

## KV cache notes (why int16 Q15 everywhere)
fp32 K/V for all heads would be 3×8 KB×… — full fp32 KV doesn't fit the 320 KB
SRAM; int16 Q15 (or int8) KV is the standard PD-disagg / on-device tradeoff
(I-BERT ICLR'21, I-ViT, TurboAttention). We use int16 Q15 + per-buffer scales.
