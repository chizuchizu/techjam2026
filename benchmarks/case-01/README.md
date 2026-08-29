# Case 01

Configuration: `B=64, S=128, D=128, H=4, F=128, L=4`, causal.

Status: **not implemented on ESP32**. No physical timing or accuracy result is
claimed.

Likely focus: reuse case-2 kernels, tile the batch to fit SRAM, keep weights
resident, and compare single-board streaming with batch-parallel multiboard
dispatch. Add `baseline/`, `optimisation/`, and `multiboard/` here when work
starts, with results scoped only to case 1.
