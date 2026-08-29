# ESP32-C3 case-02 single-board Transformer

Single-board C implementation of the case-2 transformer
(B=1, S=128, D=128, H=4, HD=32, F=128, L=4, causal, ~399K params) tuned for the
**Seeed XIAO ESP32C3** (400 KB SRAM, 4 MB flash, 160 MHz RV32IMC, **no FPU**).

The model is the reference `BaselineTransformer` from
[`../../../reference/torch_transformer_benchmark.py`](../../../reference/torch_transformer_benchmark.py)
(the official competition benchmark, with fp32 weights initialised from seed
1234 and an fp32 reference forward).

## Two numeric modes

* **EXACT** (`TM_MODE_EXACT`): fp32 GEMMs and reference softmax arithmetic via
  libgcc soft-float. It retains shared quantized attention staging, so it is a
  slower reference-quality mode rather than a bit-exact all-fp32 path.
* **FAST** (`TM_MODE_FAST`, default): Q15 activations and Q12 weights feed
  fixed-point GEMMs; attention uses integer QK/PV paths and an exp lookup
  table; GELU and quantisation are fused where possible. **Measured on device:
  5.27 s/forward**, down from the 42.15 s starting implementation.

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

## Current measured execution (on-device, FAST mode)

| Seed | Forward time (s) | max_abs vs torch | Gate |
|---|---|---|---|
| 0 | 5.268 | 8.16e-4 | PASS |
| 1 | 5.271 | 9.41e-4 | PASS |
| 2 | 5.268 | 6.83e-4 | PASS |
| 3 | 5.269 | 7.31e-4 | PASS |
| 4 | 5.266 | 6.20e-4 | PASS |

Median forward time is approximately **5.27 s**, an **8.0x speedup** over the
42.15 s measured starting point. The complete progression and kernel profiles
are in [`optimisations/README.md`](optimisations/README.md).

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

### Baseline scoring snapshot (case 2, single board)

The checked-in `tools/runs.json` and `scores.json` intentionally retain the
42.1 s pre-optimisation measurements so the scoring baseline remains
reproducible. They are not the latest latency capture.

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

### 2-node / inter-node bandwidth

Relevant when the workload is split across 2+ small boards: each node must
exchange a per-forward data amount (`node_traf`/2 each way) with its partner,
and the measured node-to-node link peak (`node_bw`) can become the **binding
resource**. `tools/score.py` supports this with an overlap model:

- `t_transfer = node_traf/node_bw (+ host_traf/2 / host_bw)`
- per-case link-bound score = `score / max(1, t_transfer/t_measured)` — a link
  faster than the compute leaves the score unchanged; a slower link scales the
  score down proportionally (compute/communication assumed overlapped).

Measured links (source of truth: [`../../multiboard/esp32-linkbench/`](../../multiboard/esp32-linkbench/) ESP-NOW benchmark +
`tools/serial_bw.py`; values below captured from a 2-board run):

| Link | Direction | Measured | Note |
|---|---|---|---|
| ESP-NOW node↔node (2.4 GHz, 1 Mbps PHY) | board→board | **~60 KB/s** mean (36–38 @64 B, 60–63 @128 B, 79–80 @240 B) | measured on-device, 2 boards, 300 fwd/size, saturated send, app-ACK + server ground-truth: 82–84% delivered; median RTT ~145–170 ms |
| USB-CDC host↔board | host→board | ~200 KB/s | reliable ceiling 200–290 KB/s (faster drops, no RX timeout) |
| USB-CDC host↔board | board→host | ~286 KB/s | identical on both boards |
| Driver pacing (device_test) | host→board | ~50 KB/s | 1 KB/20 ms, 7× safety margin |

Validation vs scoring (this repo, measured `node_bw=61470 B/s`, `node_traf=131072 B/fwd`):
`t_transfer = 2.13 s` = **5.1%** of `t_measured = 42.13 s` → link scale 1.000 →
**ExScore unchanged (26.7%)**. The node link has **~20× headroom**: it would only
become binding if per-forward node traffic grew to ≈2.4 MB (≈20× current 128 KB)
or the forward time shrank below ≈2.1 s. The USB-CDC host link (≈0.23 s host
share) also does not bind. Conclusion: for the current 2-board footprint the
**compute (42 s forward) dominates; node-to-node bandwidth is not the scoring
bottleneck**, but the model above reports the instant it becomes one.

Run: `python3 tools/score.py --node-bw <B/s> --node-traf <bytes/forward>`
(e.g. `--node-bw 200000 --node-traf 131072` = 64 KB in + 64 KB out per forward;
synthetic example — replace with the real split). Published reference
(primary Espressif docs, verified): ESP-NOW default link rate 1 Mbps, max
250 B/frame; classic-ESP32 measured payload ~27–100 KB/s at 1 Mbps; ESP32-C3
Wi-Fi PHY 1x1 b/g/n up to 150 Mbps, iperf C3 over-the-air TCP 20/35,
UDP 30/50 Mbit/s (so a TCP/UDP link would be ~100× faster than ESP-NOW if a
router is available).
