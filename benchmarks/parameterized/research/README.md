# Research synthesis (4 parallel firecrawl subagents)

This directory holds the four raw reports commissioned before/while implementing
`transformer.c`:

- `res-arch-dynamic-mcu.md` — how dynamic-shape MCU runtimes (TFLM tensor arena,
  llama2.c config struct, cONNXr, tinyml-rs) keep geometry out of `#define` and
  lay out one arena/bump allocator. Wired into `tm_case` + `tm_workspace_*`.
- `res-arch-quant-extreme.md` — no-FPU quantized math limits. Key fact wired into
  the code: Q15xQ12 GEMM must use an int64 accumulator (int32 overflows after
  ~16 worst-case / 256 RMS terms) while keeping each product in int32; softmax
  logits need ~7 fractional bits to pass the 0.002/0.02 gate.
- `res-arch-longseq.md` — feasibility envelope on a 400 KB / 4 MB C3:
  S=128/D=128 is practical; larger D (1024) exceeds flash (weights) and
  S=100000 is infeasible (attention H*S^2 alone is ~GB-scale).
- `res-arch-profiling.md` — `esp_timer_get_time()` (µs) for per-kernel timing and
  `esp_cpu_get_cycle_count()` (`rdcycle`) for fine microbenchmarks; SystemView /
  FreeRTOS run-time stats for island-wide profiling; esp-nn's harness as the
  canonical pattern. Wired into the `tm_profile` hooks and the `esp32/` sketch.

Note: the self-hosted Firecrawl instance had no web-search or developer-index
backend during these runs, so the agents scraped primary docs + GitHub API
directly; every source URL is listed at the bottom of each report.
