# Case-02 single-board optimisation

[`esp32-baseline/`](esp32-baseline/) is the complete four-layer case-02
Transformer body for one XIAO ESP32-C3. It includes firmware, generated-artifact
export, host validation, physical-device validation, scoring, and a measured
optimisation log.

## Result progression

All rows use the same physical board, case shape, complete-forward scope, and
correctness gate.

| Build | Time/forward | Speedup | Validation |
|---|---:|---:|---|
| Initial hybrid baseline | 42.15 s | 1.00x | Pass |
| Integer attention | 15.21 s | 2.77x | Pass |
| Integer attention + exp LUT | 13.70 s | 3.08x | Pass |
| GEMM/GELU/quantisation micro-optimisations | 6.91 s | 6.10x | Pass |
| Fused QKV quantisation | 6.56 s | 6.43x | Pass |
| Integer exp index + core4 GEMM | **5.27 s** | **8.0x** | Pass, 50/50 host checks and 5/5 device seeds |

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
