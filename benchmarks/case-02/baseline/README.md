# Case-02 baseline

The shared PyTorch definition is
[`../../../torch_transformer_benchmark.py`](../../../torch_transformer_benchmark.py).
This directory owns the physical starting point used to measure case-02
single-board speedup.

## Physical starting result

| Implementation | Board | Complete forward | Accuracy |
|---|---|---:|---|
| Initial hybrid C implementation | XIAO ESP32-C3, 160 MHz | 42.152 s | Pass, max absolute error 0.0008124 |

The result covers the complete case-02 Transformer body, not only attention.
Its independent review and raw physical capture are in [`results/`](results/).

The source evolved into the maintained implementation under
[`../optimisation/esp32-baseline/`](../optimisation/esp32-baseline/). The
baseline profile is retained in that implementation's
[`optimisation log`](../optimisation/esp32-baseline/optimisations/00_baseline_profile.md)
so the measured lineage remains explicit without duplicating firmware.
