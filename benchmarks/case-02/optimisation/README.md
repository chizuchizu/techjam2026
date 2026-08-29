# Case-02 single-board optimisation

[`esp32-baseline/`](esp32-baseline/) is the complete four-layer case-02
Transformer body for one XIAO ESP32-C3. It includes firmware, generated-artifact
export, host validation, physical-device validation, scoring, and a measured
optimisation log.

## Result progression

All rows use the same physical board, case shape, complete-forward scope, and
correctness gate.

| Build | Time/forward | Speedup | Gate |
|---|---:|---:|---|
| baseline fp32 | 42.15 s | 1.00x | 25/25 host, 5/5 device |
| integer attention (opt 1) | 15.21 s | 2.77x | 25/25 + 25/25 EXACT |
| + exp LUT (opt 2) | 13.70 s | 3.08x | 25/25 + 25/25 EXACT |
| + GEMM tiling / int GELU / quant+attn micro-opts (opts 3-5) | 6.91 s | 6.10x | host 50/50, worst 8e-4 |
| + fused QKV int16 quantization (opt 6) | 6.56 s | 6.43x | host 50/50, device 5/5, worst 9.5e-4 |
| + integer exp index, core4 GEMM (opts 7-8) | 5.27 s | 8.0x | host 50/50, device 5/5, worst 9.4e-4 |
| + j-outer head GEMM, integer amax (opt 8b) | 4.862 s | 8.67x | host 50/50, worst 9.4e-4 |
| + fused LN->Q15 amax-bound (opt 9) | 4.784 s | 8.82x | host 50/50 (FAST+EXACT), worst 1.14e-3 |
| + integer LN pass, direct Q15 emit (opt 9b) | 4.160 s | 10.13x | host 50/50, worst 1.19e-3; device 5/5 |
| + oproj Q15-ctx fusion (opt 10) | 3.969 s | 10.62x | host 50/50 FAST+EXACT; device 5/5, worst 1.14e-3 |
| + integer-only attention PV + int ctx epilogue (opt 11) | 3.688 s | ~10.7x | host 50/50, worst 1.20e-3; device 5/5, worst 1.11e-3 |
| + core4_v2 j-tile-2 + K-pair prefetch (opt 12) | 3.664 s | ~11.5x | host 50/50, worst 1.20e-3; device 5/5 |
| + integer LN pass, int amax+quant (opts 13-14) | 3.205 s | ~13.1x | host 50/50, worst 1.27e-3; device 5/5, worst 1.088e-3 |
| + core5 GEMM j-tile-2 x IBLK=4 (opt 15) | 2.982 s | 14.1x | host 50/50, worst 1.277e-3; device 5/5, worst 1.088e-3 |
| + head_q15 8-MAC hand-asm (opt 16) | 2.838 s | 14.8x | host 50/50; device 5/5, worst 1.088e-3 (bit-exact) |
| + FFN1 fixed-point Q15 epilogue (opt 17) | 2.701 s | 15.6x | host 50/50 (9.05e-4..1.12e-3); device 5/5 |
| + KB1 asm requant, core5 bias-fold, int32-limb QK (opt 18) | 2.447 s | 17.2x | host 50/50 (FAST <=9.8e-4); device 5/5 (<=1.29e-3) |
| + QK j-unroll-4, PV 8-accumulator (opt 19) | 2.386 s | 17.6x | host 50/50 (<=9.99e-4); device 5/5; bit-exact attention |
| + integer-residual FAST path (opt 21) | 2.122 s | 19.8x | host 50/50 (worst 1.03e-3); device 25/25 PASS; ExScore 5.30 |
| + KB0 head-GEMM asm on R1 (opt 22) | 2.056 s | 20.5x | host 50/50 (worst 1.03e-3); device 25/25 PASS; ExScore 5.48 |
| **+ core5 4x2 asm fix (opt 23, col1 product-reuse)** | **1.996 s** | **21.1x** | host 54/54 (worst 1.03e-3); device 25/25 PASS (worst 1.24e-3); probe bad=0; ExScore 5.65 |

The detailed kernel timings, numerical decisions, and rejected variants are in
[`esp32-baseline/optimisations/`](esp32-baseline/optimisations/).

## Reproduce

From the repository root:

The exporter requires a Python environment with PyTorch and NumPy.

```bash
cd benchmarks/case-02/optimisation/esp32-baseline
python3 tools/export_case2.py --outdir . --seeds 25
make -C tools test
pio run -e esp32-baseline
```

Generated weights and test vectors are intentionally ignored. Regenerate them
inside `esp32-baseline/`; do not commit them.
