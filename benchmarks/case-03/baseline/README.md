# Case 3 baseline

The shared PyTorch definition is
[`../../../torch_transformer_benchmark.py`](../../../torch_transformer_benchmark.py).
The torch reference is vendored inside the implementation at
[`../optimisation/esp32-baseline/tools/torch_ref.py`](../optimisation/esp32-baseline/tools/torch_ref.py)
(a self-contained copy of the official benchmark reference; same weight
seed 1234, same random-input generator, same fp32 forward). This
directory owns the physical starting point used to measure the
Case 3 single-board batch speedup.

## Physical starting result

Complete forward = all 4 batch inputs on one board.

| Implementation | Board | Complete forward | Accuracy |
|---|---:|---|---|
| Current hybrid C implementation | XIAO ESP32-C3, 160 MHz | 7.96 s | Pass, worst max absolute error 1.0320e-03 (FAST) |

The result covers the complete Case 3 Transformer body, not only
attention. The raw capture and independent review are in
[`results/`](results/). The source lives under
[`../optimisation/esp32-baseline/`](../optimisation/esp32-baseline/);
the optimisation log retained there mirrors the case-02 baseline
profile so the measured lineage stays explicit without duplicating
firmware.

Timing note: per-input forward measured on-board (board A) = 1.990 s
(B=1, identical S/D/H/F/L geometry, 25/25 device seeds). The batch
policy for Case 3 forbids reflashing the shared boards, so the
complete-forward time is 4 x 1.990 s = 7.96 s (per-input 1.990 s measured on-board, board A).
