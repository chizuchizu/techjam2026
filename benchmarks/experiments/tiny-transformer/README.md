# Tiny trained Transformer experiment

This experiment proves a complete trained inference pipeline on one XIAO
ESP32-C3. It is not one of the 14 official benchmark shapes.

| Path | Purpose |
|---|---|
| [`esp32_tiny_transformer/`](esp32_tiny_transformer/) | Arduino firmware and model kernels |
| [`tools/`](tools/) | Training/export and independent validation |
| [`results/`](results/) | Physical capture and result report |

The measured median is 106.614 ms per complete forward and 9.38 generated
tokens/s. See [`results/TINY_TRANSFORMER_RESULTS.md`](results/TINY_TRANSFORMER_RESULTS.md).
