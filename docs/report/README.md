# Project report

`index.html` is the single-page engineering report for the ESP32 Transformer work:
why a microcontroller, what an ESP32-C3 is, the three constraints (no FPU, 321 KB of
RAM, a slow link), the case-2 optimisation from 42.15 s to 1.996 s, how the work was
parallelised, how it generalised to the other official shapes, and the four cases that
cannot run.

Open it in any browser. It is self-contained apart from the Google Fonts stylesheet;
all nine charts are inline SVG with no library.

## Regenerating

```sh
python3 docs/report/build_report.py
```

`build_report.py` holds the CSS shell and builds the body section by section. Chart
geometry is computed from the data, so correcting a measurement means editing one
number in one list and re-running the script rather than adjusting SVG coordinates.
The output is deterministic, so the committed HTML diffs cleanly.

## Sourcing

Every figure is either a physical device measurement or is labelled as a projection or
a derived bound. The numbers come from:

| Section | Source |
|---|---|
| Optimisation ladder, per-step timings | [`benchmarks/case-02/optimisation/esp32-baseline/optimisations/`](../../benchmarks/case-02/optimisation/esp32-baseline/optimisations/) |
| Baseline and post-optimisation profiles | `optimisations/00_baseline_profile.md`, `20_profiling_memory_compute.md` |
| Memory budgets, section sizes | [`tinyprof`](../../tinyprof/) and the linked ELF |
| Two-board case-2 split | [`benchmarks/case-02/multiboard/results/`](../../benchmarks/case-02/multiboard/results/) |
| Batch data-parallel scaling | [`benchmarks/batch-dp/`](../../benchmarks/batch-dp/) |
| Link throughput and latency | [`esp32-linkbench/`](../../esp32-linkbench/) |
| Per-case status and gates | [`benchmarks/README.md`](../../benchmarks/README.md) |
| Blocked-case arithmetic | the case READMEs plus `docs/esp32_implementation_summary.md` |

`benchmarks/README.md` is the authoritative case index. If a figure here disagrees with
a case README, the case README wins and this report should be corrected.
