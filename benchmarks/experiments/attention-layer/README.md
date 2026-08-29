# Attention-layer experiment

This directory contains the Arduino firmware, independent host validator, and
all raw results for the small single-board attention benchmark.

| Path | Purpose |
|---|---|
| [`esp32_attention_benchmark/`](esp32_attention_benchmark/) | Firmware and C++ kernels |
| [`tools/`](tools/) | Independent end-to-end validator |
| [`results/`](results/) | Raw captures and interpreted results |

The best complete attention-layer configuration is 3.05x faster for padding
and 3.87x faster for padding plus causal masking while passing the numerical
gate. See [`results/RESULTS.md`](results/RESULTS.md).
