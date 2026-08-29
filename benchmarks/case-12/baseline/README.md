# Case-12 baseline

The complete four-layer Transformer body for this shape is implemented in
`../optimisation/esp32-baseline/` with the case-12 geometry set in
`src/tm_config.h`. The torch reference is vendored at
[`../optimisation/esp32-baseline/tools/torch_ref.py`](../optimisation/esp32-baseline/tools/torch_ref.py)
(a self-contained copy of the official benchmark reference: same weight-init
seed 1234, same random-input generator, same fp32 reference forward).

This directory owns the physical starting point used to measure case-12
single-board speedup.

## Physical starting result

| Implementation | Board | Per-input forward (measured on-board) | Accuracy |
|---|---:|---:|---|
| Current implementation (first physical capture) | XIAO ESP32-C3, 160 MHz | 0.493 s | Pass, 5/5 device seeds (worst device max_abs 1.0332e-03) |

The result covers the complete case-12 Transformer body, not only attention.
Its independent review and raw physical capture are in [`results/`](results/).

The source is the maintained implementation under
[`../optimisation/esp32-baseline/`](../optimisation/esp32-baseline/); the
per-case geometry is set only in `src/tm_config.h`. Its optimisation log is
retained in that implementation's
[`optimisation log`](../optimisation/esp32-baseline/optimisations/00_baseline_profile.md)
so the measured lineage remains explicit without duplicating firmware.
