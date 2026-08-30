# ESP32-C3 case-06 streamed Transformer baseline

Port of the case-02 ESP32 baseline to case 06
(`B=10000, S=128, D=128, H=4, HD=32, F=128, L=4`, causal). The per-input
geometry and all weights are **identical to case 02**, so the weights and the
per-frame torch testdata are reused from the case-02 port (same model weights,
same seed-1234 initialisation, same per-frame input/ref bins). The target is a
**Seeed XIAO ESP32C3** (320 KB SRAM, 4 MB flash, 160 MHz RV32IMC, no FPU).

`TM_B` is set to `10000` in `src/tm_config.h` for documentation only. The
firmware still runs **one forward per input frame**, so a full case-06 batch is
a *streamed* execution of 10,000 frames — it is never resident in SRAM at once.

## What exists

    platformio.ini        two espressif32 envs (XIAO C3 + generic devkit)
    src/tm_config.h       geometry (B=10000, S/D/H/F/L = 128/128/4/128/4, causal)
    src/kernels.h/.c      fp32 GEMM, Q15xQ12 GEMM, LayerNorm, GELU, fast exp
    src/model.h/.c        forward pass + streaming causal attention
    src/main.cpp          Arduino firmware (serial protocol, timing)
    tools/export_case2.py torch artifact exporter (system python3; kept for geometry reference)
    tools/torch_ref.py    vendored torch reference used by export_case2.py
    tools/host_test.c     host validation vs torch references (25 seeds)
    tools/device_test.py  per-frame serial driver + torch gate on captured output
    tools/batch_stream.py host-side batch stream driver (dry-run + live serial modes)
    weights.bin           flat fp32 weights (1.59 MB, embedded; identical to case 02)
    weights_q12.bin       Q12 weights + scales (0.79 MB, embedded; identical to case 02)
    manifest.json         seed/layout/hash manifest (identical to case 02)
    testdata/             per-seed input_<s>.bin / ref_<s>.bin (reused from case 02)

## Host validation (measured, this repo)

Built and ran the C forward in both numeric modes against the 25 torch
references:

    cd benchmarks/case-06/optimisation/esp32-baseline/tools
    make && ./host_test all --both --reps 5

Result: **ALL PASS** (50/50 seed-runs = 25 EXACT + 25 FAST).

- FAST worst `max_abs = 1.0320e-3` (gate is `|a-b| <= 0.002 OR <= 0.02*|b|`)
- EXACT worst `max_abs = 7.8201e-5`

Full captured output: `benchmarks/case-06/HOST_TEST.log`.

## Device build status

    cd benchmarks/case-06/optimisation/esp32-baseline
    timeout 600 pio run -e esp32-baseline

Result: **links successfully** (PlatformIO SUCCESS, no errors).

- Platform: Espressif 32 (51.3.6) / arduino-esp32 3.0.7
- RAM:   272,700 / 327,680 bytes (83.2%)
- Flash: 2,671,272 / 3,145,728 bytes (84.9%)

Full captured output: `benchmarks/case-06/BUILD.log`.
No `upload` / no physical run was performed for case 06, so **no case-06
physical timing is claimed**.

## Full B=10000 batch (not measured)

A full case-06 batch means 10,000 board forwards plus 2 x 10,000 x 64 KiB of
input/output transfer. This is a long-running execution on a single C3 and is
**not run or claimed as measured here**. Any per-forward time below is a
case-02 reference only.

`tools/batch_stream.py` is the host side of the stream. It walks the per-frame
bins (or synthesizes deterministic frames), speaks the same serial protocol as
`tools/device_test.py`, optionally gates each captured frame against its torch
reference, and reports the transport/estimation structure without inventing
physical timing:

    python3 tools/batch_stream.py --help
    python3 tools/batch_stream.py --dry-run --frames 10000
    python3 tools/batch_stream.py /dev/cu.usbmodem2101 --frames 10 --check

For any live run, all timing it prints comes from the firmware's own
`esp_timer_get_time()` delta (per-'R' `us=<n>`) and the host wall clock for that
specific run; it never fabricates a per-forward number.

## Numeric modes (unchanged from the case-02 port)

- **EXACT** (`TM_MODE_EXACT`): fp32 GEMMs and reference softmax via libgcc
  soft-float, with shared quantized attention staging (reference quality, not
  bit-exact).
- **FAST** (`TM_MODE_FAST`, default): Q15 activations x Q12 weights feed
  fixed-point GEMMs; integer attention/exp path; fused LayerNorm/GELU/quantize.
  The case-02 optimisation snapshot measured ~1.996 s/forward on device; that
  number is a *case-02 reference*, not a case-06 measurement.

## Serial protocol (src/main.cpp)

`M` prints mode/geometry; `S` re-inits; `R` + 65536 input bytes runs one
forward and streams 65536 output bytes then `END`; `T <n>` runs n timed
forwards; `P`/`K`/`Q` dump profile/microbench counters.
