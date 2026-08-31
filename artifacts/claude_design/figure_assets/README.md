# Figure asset manifest

These are production assets for Claude Design, not screenshots of a proposed
deck. Prefer SVG for the editable presentation; PNG files are 240 dpi fallbacks.
All charts use measured repository values unless the convention column says
otherwise.

| Asset | Target slide | Data and convention | Primary source |
|---|---:|---|---|
| `fig01_fpu_cost_c3_vs_s3` | 5 | Cycles/op, logarithmic axis; C3 software FP versus S3 hardware FPU | `benchmarks/case-02/optimisation/esp32-baseline/optimisations/research.md` |
| `fig02_memory_scale` | 6, 19 | KB requirements and budgets, logarithmic axis; case-2 1.6 MB bar is **fp32** weights | `docs/report/index.html` |
| `fig03_final_sram_budget` | 6 | Linked ELF bytes; segment capacity 321,296 B; internal labels use KiB | `benchmarks/case-02/optimisation/esp32-baseline/optimisations/25_tinyprof.md` |
| `fig04_baseline_profile` | 7 | Normalized timed-operator shares; raw attention wall ratio is separately 30.09/42.15 = 71.4% | `benchmarks/case-02/optimisation/esp32-baseline/optimisations/00_baseline_profile.md` |
| `fig05_optimization_ladder` | 8 | Cumulative seconds/forward, logarithmic axis; 24 steps grouped into 21 recorded milestones | `docs/report/index.html` |
| `fig06_gemm_kernel` | 10 | Comparable core4_v2 to core5 microbenchmark, cycles/MAC | `benchmarks/case-02/optimisation/esp32-baseline/optimisations/15_gemm_core5_iblk4.md` |
| `fig07_fusion_deltas` | 11 | Three isolated transitions at different points in the cumulative sequence; do not sum them | `benchmarks/case-02/optimisation/README.md` |
| `fig08_accuracy_gate` | 12 | Maximum absolute error against the absolute branch of the OR gate | `COMPETITION_RULES.MD`, `docs/report/index.html` |
| `fig09_wifi_memory_fit` | 14 | Untiled estimate versus tiled post-association measurement; static/runtime/free are stacked separately | `benchmarks/case-02/optimisation/esp32-baseline/optimisations/24_sequence_tiled_wifi.md` |
| `fig10_link_throughput` | 15 | Dedicated link benchmark; ESP-NOW uses 240 B payload and TCP peak uses 4096 B payload | `benchmarks/case-02/multiboard/esp32-linkbench/README.md` |
| `fig11_cluster_scaling` | 17 | Device compute wall; host transfer excluded; each series is normalized to its one-node baseline | `benchmarks/README.md`, `docs/report/index.html` |

Machine-readable data are supplied for the two densest visuals:

- `optimization_ladder.csv`
- `benchmark_cases.csv`

The Matplotlib source for every figure is
`artifacts/claude_design/build_instruction_pack.py`.
