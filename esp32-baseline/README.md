# ESP32-C3 case-2 transformer baseline (Tech Jam 2026)

Single-board C implementation of the case-2 transformer
(B=1, S=128, D=128, H=4, HD=32, F=128, L=4, causal, ~399K params) tuned for the
**Seeed XIAO ESP32C3** (400 KB SRAM, 4 MB flash, 160 MHz RV32IMC, **no FPU**).

The model is the reference `BaselineTransformer` from
`../torch_transformer_benchmark.py` (fp32 weights init with seed 1234, fp32
reference forward).

## Two numeric modes

* **EXACT** (`TM_MODE_EXACT`): every GEMM and op in IEEE fp32 via libgcc
  soft-float. Bit-for-bit torch-like; ~50-90 s/forward on the C3 (fine for a
  reference check, not for throughput).
* **FAST** (`TM_MODE_FAST`, default): the six per-layer projection GEMMs
  become **Q15 x Q12 fixed-point** (int16 x int16 -> int32 saturating),
  activations quantized per-tensor, weights pre-quantized offline. Attention
  QK/PV, LayerNorm and the deg-11-poly GELU stay in fp32. Target ~2-4 s/forward.

FAST is validated against the real benchmark gate (|a-b| <= 0.002 OR
|a-b| <= 0.02*|b|): **0 failures over 25 random seeds** (seeds 1234..1258),
worst max_abs error 9.6e-4.

## Repository layout

    platformio.ini        two espressif32 envs (XIAO C3 + generic devkit)
    src/tm_config.h       model geometry, numeric modes, weight layout
    src/kernels.h/.c      fp32 GEMM, Q15xQ12 GEMM, LayerNorm, GELU, fast exp
    src/model.h/.c        forward pass + streaming causal attention
    src/main.cpp          Arduino firmware (serial protocol, timing)
    tools/export_case2.py torch artifact exporter (system python3)
    tools/host_test.c     host validation vs torch references (25 seeds)
    tools/compare.py      verify a raw device output dump vs torch refs
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
