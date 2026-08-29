# Case 4 baseline

The shared PyTorch definition is
[`../../../torch_transformer_benchmark.py`](../../../torch_transformer_benchmark.py).
The torch reference is vendored inside the implementation at
[`../optimisation/esp32-baseline/tools/torch_ref.py`](../optimisation/esp32-baseline/tools/torch_ref.py)
(a self-contained copy of the official benchmark reference; same weight
seed 1234, same random-input generator, same fp32 forward). This
directory owns the physical starting point used to measure the
Case 4 single-board batch speedup.

## Physical starting result

The batch of 16 inputs streams as 16 single-input forwards; only
the per-input forward is measured on-board (a complete-batch total
would be a derived projection and is not reported).

| Implementation | Board | Per-input forward (measured on-board) | Accuracy |
|---|---:|---|---|
| Current hybrid C implementation | XIAO ESP32-C3, 160 MHz | 1.990 s | Pass, worst max absolute error 1.0320e-03 (FAST) |

The result covers the complete Case 4 Transformer body, not only
attention. The raw capture and independent review are in
[`results/`](results/). The source lives under
[`../optimisation/esp32-baseline/`](../optimisation/esp32-baseline/);
the optimisation log retained there mirrors the case-02 baseline
profile so the measured lineage stays explicit without duplicating
firmware.

Timing note: per-input forward measured on-board (board A) = 1.990 s
(single-input, identical S/D/H/F/L geometry, 5/5 device seeds). Only
real on-board measurements are reported; a complete-batch total
would be a derived projection and is therefore omitted.
