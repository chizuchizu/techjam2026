# Case 11 baseline

The shared PyTorch definition is
[`../../../torch_transformer_benchmark.py`](../../../torch_transformer_benchmark.py).
The torch reference is vendored inside the implementation at
[`../optimisation/esp32-baseline/tools/torch_ref.py`](../optimisation/esp32-baseline/tools/torch_ref.py)
(a self-contained copy of the official benchmark reference; same weight
seed 1234, same random-input generator, same fp32 forward). This
directory owns the physical starting point used to measure the
Case 11 (H=16) speedup.

## Physical starting result

Complete forward = all 64 batch inputs on one board (each input is one
`S*D` frame streamed through the firmware).

| Implementation | Board | Complete forward (single-input) | Accuracy |
|---|---:|---|
| Current hybrid C implementation | XIAO ESP32-C3, 160 MHz | 2.460 s (157.44 s batch-64) | Pass, worst max absolute error 1.0893e-03 (FAST, physical seed 3) |

The result covers the complete Case 11 Transformer body, not only
attention. Its raw physical capture and independent review are in
[`results/`](results/). The source lives under
[`../optimisation/esp32-baseline/`](../optimisation/esp32-baseline/);
the optimisation log retained there mirrors the case-02 baseline
profile so the measured lineage stays explicit without duplicating
firmware.

Timing note: single-input forward 2.460 s is a fresh physical capture
on PORT B (`/dev/cu.usbmodem1101`, heads group). Complete batch-64
forward = 64 x 2.460 s = 157.44 s (firmware streams one frame per
forward).
