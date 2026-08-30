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
so the measured lineage remains explicit.

## Restored firmware

[`esp32-baseline-v0/`](esp32-baseline-v0/) is the firmware that produced the
42.152 s figure, restored from commit `79f284a` so that the baseline can be
re-profiled with the same tool as the optimised build rather than compared
against transcribed numbers. It is that commit verbatim apart from opt-in
[`tinyprof`](../../../tinyprof/) zone brackets, which compile to nothing unless
`-DTINYPROF_LIB` is set — `pio run -e esp32-baseline` still builds the original
code with no instrumentation.

Two independent checks that it is the right revision:

- it builds to RAM 81.7% (267,804 B) and Flash 83.3% (2,621,584 B), matching
  [`FLASH_TEST.md`](../optimisation/esp32-baseline/FLASH_TEST.md);
- seed 0 gates at `max_abs = 8.1241e-04` with 0 failures, the same value as
  [`results/teammate_esp32_baseline_seed0_v1.log`](results/teammate_esp32_baseline_seed0_v1.log).

It passes 50/50 host seed-runs. `weights.bin`, `weights_q12.bin`, `manifest.json`
and `testdata/` are symlinks into `../optimisation/esp32-baseline/`, so both
firmwares are measured against byte-identical inputs.
