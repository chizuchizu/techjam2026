# ESP32-C3 case-2 transformer baseline (Tech Jam 2026)

Single-board C implementation of the case-2 transformer
(B=1, S=128, D=128, H=4, HD=32, F=128, L=4, causal, ~399K params) tuned for the
**Seeed XIAO ESP32C3** (400 KB SRAM, 4 MB flash, 160 MHz RV32IMC, **no FPU**).

The model is the reference `BaselineTransformer` from
`../h200/torch_transformer_benchmark.py` (fp32 weights init with seed 1234, fp32
reference forward).

## Two numeric modes

* **EXACT** (`TM_MODE_EXACT`): every GEMM and op in IEEE fp32 via libgcc
  soft-float. Bit-for-bit torch-like; ~50-90 s/forward on the C3 (fine for a
  reference check, not for throughput).
* **FAST** (`TM_MODE_FAST`, default): the six per-layer projection GEMMs
  become **Q15 x Q12 fixed-point** (int16 x int16 -> int32 saturating),
  activations quantized per-tensor, weights pre-quantized offline. Attention
  QK/PV, LayerNorm and the deg-11-poly GELU stay in fp32. **Measured on
  device: ~42.1 s/forward** (the earlier ~2-4 s target was wrong - the fp32
  attention/LN/GELU parts run as software float on the FPU-less core and
  dominate the time).

FAST is validated against the real benchmark gate (|a-b| <= 0.002 OR
|a-b| <= 0.02*|b|): **0 failures over 25 random seeds** (seeds 1234..1258),
worst max_abs error 9.6e-4. On-device: **5/5 seeds pass**, max_abs
6.7e-4..8.1e-4.

## Repository layout

    platformio.ini        two espressif32 envs (XIAO C3 + generic devkit)
    src/tm_config.h       model geometry, numeric modes, weight layout
    src/kernels.h/.c      fp32 GEMM, Q15xQ12 GEMM, LayerNorm, GELU, fast exp
    src/model.h/.c        forward pass + streaming causal attention
    src/main.cpp          Arduino firmware (serial protocol, timing)
    tools/export_case2.py torch artifact exporter (system python3)
    tools/host_test.c     host validation vs torch references (25 seeds)
    tools/compare.py      verify a raw device output dump vs torch refs
    tools/score.py        MFU + bandwidth-aware execution-score calculator
    tools/runs.json       measured device forward times (one case per run)
    scores.json           score.py output (per-case MFU, ExScore, weighted sum)
    weights.bin           flat fp32 weights (1.59 MB, embedded)
    weights_q12.bin       Q12 weights + scales (0.79 MB, embedded)
    testdata/             per-seed input_<s>.bin / ref_<s>.bin

## Weight layout

Flat fp32 (`weights.bin`, 398,592 floats); per layer the 16 blocks
norm1(w,b), q/k/v/o(w,b), norm2(w,b), f1(w,b), f2(w,b), then final_norm(w,b).
See `TM_W_BLK_*` / `woff()` in src. `weights_q12.bin` is 24 matrices
(layers x q,k,v,o,f1,f2), each `{u32 count}{f32 w_scale}{i16 data}`.

## Build & validate on the host

    python3 tools/export_case2.py --outdir . --seeds 25   # torch artifacts
    make -C tools host_test && ./tools/host_test all --both --reps 5

## Build & run on the ESP32-C3

    pio run -e esp32-baseline -t upload        # weights embedded in flash
    pio device monitor -b 115200               # then, from the monitor or a
                                               # script: send 'M' (mode),
                                               # 'R' + 65536 input bytes,
                                               # 'T' + count (timing)

Serial protocol (main.cpp): `M` prints mode; `R` reads 16384 floats,
runs one forward, streams 16384 output floats then `END`; `T <n>` does
n timed forwards and prints `TM <mode> <us>...`.

## Numbers

Param count 398,592 = 1.59 MB fp32. Live SRAM ~272 KB (fits 400 KB).
Flash for weights ~2.38 MB (fits 4 MB). Host-validated accuracy: FAST
0/25 seed failures (worst max_abs 9.6e-4); EXACT 0/25 (worst 3.6e-5).

## Measured execution (on-device, FAST mode)

| Seed | Forward time (s) | max_abs vs torch | Gate |
|---|---|---|---|
| 0 | 42.130 | 8.07e-4 | PASS |
| 1 | 42.152 | 8.06e-4 | PASS |
| 2 | 42.134 | 7.17e-4 | PASS |
| 3 | 42.151 | 8.12e-4 | PASS |
| 4 | 42.138 | 6.75e-4 | PASS |
| T-sweep ×2 | 42.103 / 42.073 | - (timing only) | - |

Mean forward time **42.09 s** -> achieved **2.91 MFLOP/s** aggregate.

## Scoring (evaluation methodology)

The benchmark evaluates a submission as a **weighted sum of per-test-case
model FLOPs utilization (MFU)**; the execution score also accounts for the
board's memory bandwidth (roofline). Numbers below are primary-source backed
(Espressif datasheet + TRM; firecrawl-verified) and reproduced by
`python3 tools/score.py`.

Model work per forward (2 FLOP/MAC), **122.57 MFLOP**:
- **100.66 M** int16 GEMMs (q,k,v,o,f1,f2 projections), FAST path;
- **16.91 M** fp32 causal attention (QK^T + PV, dequant-on-read);
- **~5.0 M** fp32 LayerNorm (x17) + deg-11 polynomial GELU + residuals.

Board peaks (no FPU; all fp32 is libgcc soft-float):
- **P_INT = 320 MFLOP/s** = 160 MMAC/s int16 ceiling (RV32IMC scalar,
  1x 32-bit MUL/cycle, no SIMD) — architectural ceiling.
- **P_SOFTFP ≈ 2.0 MFLOP/s** (range 1.5-2.7) — fp32 soft-float peak, from
  Espressif's ~100 cyc/fp32-add measurement (~160 cyc per fused MAC).
- **BW ≈ 320-640 MB/s** — scalar SRAM load bandwidth @160 MHz (4 B/cycle
  theoretical; sustained is estimate).

Definitions per test case i:
- `MFU_i = achieved_FLOPs / (t_i * peak_mix)` where peak_mix aggregates the
  mixed workload against its own ceiling: int GEMMs vs P_INT and fp32 parts
  vs P_SOFTFP.
- `ExScore_i = roofline_time / t_i`, roofline per op-class peak =
  `min(compute_peak, BW * AI)`. The GEMM kernel is **unblocked (4 B/MAC,
  AI ~0.5 FLOP/B)**, i.e. it sits right at the memory/compute ridge; the fp32
  parts are compute-bound (AI far above their ridge).
- Submission score = `sum_i w_i * MFU_i` (equal weights by default; pass
  `--weights` to override).

### Current result (case 2, single board)

| Case | t (s) | MFU (mix) | MFU (raw-int) | ExScore |
|---|---|---|---|---|
| seed 0 | 42.130 | 26.7% | 0.91% | 26.7% |
| seed 1 | 42.152 | 26.7% | 0.91% | 26.7% |
| seed 2 | 42.134 | 26.7% | 0.91% | 26.7% |
| seed 3 | 42.151 | 26.7% | 0.91% | 26.7% |
| seed 4 | 42.138 | 26.7% | 0.91% | 26.7% |
| T-sweep 1 | 42.103 | 26.8% | 0.91% | 26.8% |
| T-sweep 2 | 42.073 | 26.8% | 0.91% | 26.8% |

**Weighted-sum score: MFU = 26.7%**, **ExScore = 26.7%** (bandwidth does not
bind: the weighted roofline equals the mixed-compute peak; a blocked GEMM with
4x operand reuse would move the GEMM clearly below its ridge).
Robustness: at P_SOFTFP = 1.5..2.7 MFLOP/s, ExScore = 35.4..20.0%; at
BW = 320 MB/s, ExScore = 27.5%.

Context: against the strict raw scalar ceiling (everything as int16 MAC),
MFU = 0.91% — that framing is unfair because 18% of nominal FLOPs are fp32 and
cannot run on the int peak; those fp32 parts actually account for **97% of the
peak-equivalent time**, so the practical lever is fixed-point
attention/LayerNorm/GELU, not the GEMMs.
