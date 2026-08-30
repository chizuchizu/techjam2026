# ESP32-C3 case-09 Transformer

Single-board and opt-in WiFi implementation of the case-9 transformer
(B=64 streamed inputs, S=128, D=128, H=1, HD=128, F=128, L=4, causal,
~399K params) tuned for the
**Seeed XIAO ESP32C3** (400 KB SRAM, 4 MB flash, 160 MHz RV32IMC, **no FPU**).

The model is the reference `BaselineTransformer` from
[`../../../../torch_transformer_benchmark.py`](../../../../torch_transformer_benchmark.py)
(the official competition benchmark, with fp32 weights initialised from seed
1234 and an fp32 reference forward).

## Two numeric modes

* **EXACT** (`TM_MODE_EXACT`): fp32 GEMMs and reference softmax arithmetic via
  libgcc soft-float. It retains shared quantized attention staging, so it is a
  slower reference-quality mode rather than a bit-exact all-fp32 path.
* **FAST** (`TM_MODE_FAST`, default): Q15 activations and Q12 weights feed
  fixed-point GEMMs; attention uses integer QK/PV paths and an exp lookup
  table; GELU, LayerNorm and quantisation are fused where possible; weights
  and biases are pre-quantized offline.

FAST is validated against the real benchmark gate (|a-b| <= 0.002 OR
|a-b| <= 0.02*|b|). On-device (XIAO ESP32-C3, board A, FAST mode,
TM_PROFILE build): **5/5 seeds PASS, max_abs = 1.14e-3, 2.165 s/forward**
(the SRAM fit that makes this possible is documented in
`optimisations/24_sram_fit_sdk_patches.md`; a TM_PROFILE-off build is
faster still).

Scores: not computed for this case (no full on-board scoring run); see baseline/README.md.

## Repository layout

    platformio.ini        default USB targets plus opt-in tiled WiFi target
    src/tm_config.h       model geometry, numeric modes, weight layout
    src/kernels.h/.c      fp32 GEMM, Q15xQ12 GEMM, LayerNorm, GELU, fast exp
    src/model.h/.c        forward pass + streaming causal attention
    src/main.cpp          Arduino firmware (serial protocol, timing)
    src/model_tiled.c     16-row reduced-memory FAST forward
    src/main_wifi.cpp     persistent WiFi/TCP command endpoint
    tools/export_case2.py torch artifact exporter (system python3)
    tools/host_test.c     host validation vs torch references (25 seeds)
    tools/compare.py      verify a raw device output dump vs torch refs
    tools/score.py        MFU + bandwidth-aware execution-score calculator
    weights.bin           flat fp32 weights (1.59 MB, embedded)
    weights_q12.bin       Q12 weights + scales (0.79 MB, embedded)
    testdata/             per-seed input_<s>.bin / ref_<s>.bin

## Weight layout

Flat fp32 (`weights.bin`, 398,592 floats); per layer the 16 blocks
norm1(w,b), q/k/v/o(w,b), norm2(w,b), f1(w,b), f2(w,b), then final_norm(w,b).
See `TM_W_BLK_*` / `woff()` in src. `weights_q12.bin` is 24 matrices
(layers x q,k,v,o,f1,f2), each `{u32 count}{f32 w_scale}{i16 data}`.

## Build & validate on the host

    python3 tools/export_case2.py --outdir . --seeds 25 \
      --B 64 --S 128 --D 128 --H 1 --F 128 --L 4
    make -C tools host_test && ./tools/host_test all --both --reps 5
    make -C tools host_test_tiled && ./tools/host_test_tiled all --fast

## Build & run on the ESP32-C3

    pio run -e esp32-baseline -t upload        # weights embedded in flash
    pio device monitor -b 115200               # then, from the monitor or a
                                               # script: send 'M' (mode),
                                               # 'R' + 65536 input bytes,
                                               # 'T' + count (timing)

For the memory-reduced WiFi worker, copy `secrets.example.h` to the ignored
`secrets.h`, fill in the benchmark-LAN credentials, then run:

    pio run -e esp32-wifi-tiled -t upload --upload-port /dev/ttyACM0

Serial protocol (main.cpp): `M` prints mode; `R` reads 16384 floats,
runs one forward, streams 16384 output floats then `END`; `T <n>` does
n timed forwards and prints `TM <mode> <us>...`.

## Numbers

Param count 398,592 = 1.59 MB fp32. The optimized USB build links at
273,180 / 327,680 B static RAM. The 16-row WiFi build links at
**224,244 / 327,680 B** static RAM and 3,124,210 / 3,670,016 B flash, leaving
enough runtime heap for WiFi/lwIP on the physical boards.

## Current measured execution (on-device, FAST mode)

The optimized USB device gate passes 25/25 seeds at 2.157 s/forward. The tiled
host gate passes 25/25 seeds (worst `max_abs=1.1038e-3`). Two physical tiled
WiFi workers also completed a seed-0 TCP forward in 3.5633 / 3.5638 s with
zero failing elements and `max_abs=1.2649e-3`. A full distributed batch has
not yet been claimed. Fine-grained per-phase and per-version profiles are in
[`optimisations/README.md`](optimisations/README.md).

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

### Scoring snapshot

Scores: not computed for this case (no full on-board scoring run); see baseline/README.md.

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

Scores: not computed for this case (no full on-board scoring run); see baseline/README.md.

Run: `python3 tools/score.py --node-bw <B/s> --node-traf <bytes/forward>`
(e.g. `--node-bw 200000 --node-traf 131072` = 64 KB in + 64 KB out per forward;
synthetic example — replace with the real split). Published reference
(primary Espressif docs, verified): ESP-NOW default link rate 1 Mbps, max
250 B/frame; classic-ESP32 measured payload ~27–100 KB/s at 1 Mbps; ESP32-C3
Wi-Fi PHY 1x1 b/g/n up to 150 Mbps, iperf C3 over-the-air TCP 20/35,
UDP 30/50 Mbit/s (so a TCP/UDP link would be ~100× faster than ESP-NOW if a
router is available).
