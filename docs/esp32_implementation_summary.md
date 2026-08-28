# ESP32 Baseline — Implementation Summary & Status

**Updated:** 2026-08-28
**Track:** Active TechJam ESP32 track
**Hardware scope:** One-board numerical baseline, then multi-board scaling

---

## 1. Goal

Run the Transformer benchmark on ESP32-C3 boards using portable C/C++ kernels,
independent numerical validation, and measured multi-board execution.

The baseline target is **case 2 on one board**, end to end with a host-side
verifier. Cluster experiments then distribute independent attention work across
additional boards.

Case 2 config: `B=1, S=128, D=128, H=4, L=4, F=128`, causal attention,
398,592 parameters.

## 2. Verified hardware facts (primary sources)

| Fact | Value | Source |
|---|---|---|
| CPU | **RV32IMC**, single-core, 160 MHz, 4-stage in-order | ESP32-C3 datasheet |
| **FPU** | **NONE** — soft-float only | TRM v1.4 (`misa.F=0`, no `mstatus.FS`); Espressif blog 2025-10 (C3/C6/H2 have no FPU) |
| SRAM / flash | 400 KB internal / 4 MB (≈3.5 MB usable) | datasheet |
| Native USB | USB-Serial/JTAG, detected at `/dev/cu.usbmodem2101`, VID:PID `303A:1001` | `pio device list`, `system_profiler` |
| PlatformIO | 6.1.19 installed (`/opt/homebrew/bin/pio`), connection verified | local |

**Key consequence:** every FP32 op is a software call (`__addsf3`/`__mulsf3` ~50–100+ cycles,
`expf` ~2,000–3,500 cycles). Integer math (`RV32M`) is the fast path.

## 3. Feasibility results (computed)

- **20 boards:** 11/14 cases feasible (cases 1,5,13 need flash-spill/sharing).
- **28–32 boards:** 12/14 (adds case 8).
- **~300 boards:** 13/14 (adds case 6).
- **Case 14:** impossible (≈19.7 GB working set, 2.7 PFLOPs; needs ≥5,618 boards for storage).
- **Single board:** only **case 2** runs cleanly (~160 KB working set, ~1–2 s target).
  Case 3 is borderline (384 KB vs 400 KB SRAM). All others exceed SRAM (activations alone ≥512 KB).

## 4. Scope decision (minimal baseline)

- Case 2, one board, C + Arduino framework, platform `seeed_xiao_esp32c3`.
- **fp32 weights (1.59 MB) in flash** (fits 3.5 MB usable); activations in SRAM;
  **KV stored fp16 (66 KB/layer)** to save SRAM.
- Simple serial protocol for host compare; no Wi-Fi/mesh/flash-spill in v1.
- Accuracy gate: elementwise `|a−b| ≤ 0.002 OR ≤ 0.02·|b|`, zero failures, 25 seeded trials.
  Timing: median of 20 runs.

### Memory map (case 2)
| Item | fp32 | fp16 |
|---|---|---|
| Weights (p = 398,592) | 1.59 MB (flash) | 0.80 MB (flash) |
| One activation [1,128,128] | 66 KB | 33 KB |
| KV per layer | 131 KB | **66 KB** |
| Live SRAM budget | ≈262 KB < 400 KB ✅ | |

### Planned layout
```
esp32-baseline/
  platformio.ini          # board seeed_xiao_esp32c3, Arduino framework
  src/main.cpp            # load weights, run, print results over serial
  src/model.c/.h          # forward(): layer loop
  src/kernels.c           # gemm_f32, softmax_causal, layernorm, gelu_erf, res_add
  src/weights.h           # generated static const float W[398592]
  tools/export_case2.py   # torch: weights.bin, inputs.bin, ref_output.bin
  tools/compare.py        # parse serial, elementwise diff, verdict
```

## 5. Kernel + FP32-emulation strategy (paper-grounded)

See `docs/esp32_fastest_kernels_research.md` and `docs/esp32_fp32_emulation_research.md`
for full citations. Decisions:

1. **Exact IEEE soft-float has a hard ~100 cycles/op ceiling** (SoftFloat 3e's own benchmark:
   ~8–10 Mop/s @ 1 GHz ≈ 100–125 cycles/op). Optimizing it further is a dead end
   → **hybrid FP32 emulation**: fp32-compatible API, but high-volume math in fast
   integer/approximate kernels; exact soft-float only for low-volume, correctness-critical spots.
2. **GEMM (90% of FLOPs):** fixed-point Q15×Q15→int32 MAC, register-blocked 4×4,
   ≥4 independent accumulators, unroll ×4, no packing at D=128. (CMSIS-NN / qf_math /
   AMD-BLIS lessons.)
3. **Softmax exp:** **Schraudolph bit-trick** (`EXPA*y + (1072693248−EXPC)` ≈ 4 instruction)
   instead of `expf`. ~200–500× speedup on the dominant softmax cost.
4. **Attention = fused/flash-style streaming** (single pass QKᵀ→softmax→PV, no 64 KB score
   buffer, KV read once) **combined with** fast exp (they are complementary, not alternatives).
   A/B test streaming two-pass vs online-rescale at S=128; causal = loop bound.
5. **LayerNorm:** exact two-pass fp32, rstd via fast-inverse-sqrt + Newton (no soft `sqrtf`/`div`).
6. **GELU:** keep **exact erf** (host uses `approximate='none'`); Cody rational-Chebyshev `erff`
   (1969, ~1e-7 rel err). Do NOT use the tanh approximation.
7. Every kernel behind an `ACCURACY_MODE` (EXACT vs FAST) switch → per-kernel A/B validation
   against the handle; also the contest's "naive vs optimized" story.

**Expected:** pure soft-float ≈ 60 s/forward → hybrid ≈ 1–2 s (≈30×).

## 6. Deliverables so far (files)

| File | What |
|---|---|
| `docs/esp32_implementation_summary.md` | This file |
| `docs/esp32_fastest_kernels_research.md` | Kernel techniques research brief (C) |
| `docs/esp32_fp32_emulation_research.md` | Paper-grounded FP32-emulation recommendation |
| `h200/model_architecture_research.md` | Original H200 model architecture used by the exporter |
| `.firecrawl/` | Raw research sources (scrapes, search JSON) |
| `esp32-baseline/` | **Not scaffolded yet** (code) — next step |

## 7. Status vs plan

`plan` (7 phases): target set ✅ → toolchain ✅ (PlatformIO verified, device detected) →
host reference/exporter (TODO) → model port (TODO) → accuracy gate (TODO) →
on-device benchmark harness (TODO) → multi-board scaling / write-up (v2).

**Done:** hardware verification, feasibility math, scope, all kernel/emulation research, acceptance criteria.
**Next:** scaffold `esp32-baseline/`, port kernels, validate vs torch reference, wire timing.
See `TODO.md` for the current ESP32 priorities. The inactive GPU backlog is in
`h200/TODO.md`.

## 8. Open risks
- Accuracy gate vs torch reference (softmax/LN/GELU ordering, `__expf` vs torch math) — mitigate
  by testing kernels against the reference early, one swap at a time.
- Q15 GEMM + Schraudolph exp error must be validated per case; any failing kernel reverts to
  EXACT soft-float for that kernel only.
- Firecrawl **Agent** endpoint is disabled server-side ("Agent beta is not enabled") —
  research used Firecrawl search+scrape (same content engine) instead.
