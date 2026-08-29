# Official benchmark cases

Each official test case has its own directory because its shape can require a
different memory layout, kernel schedule, or distribution strategy. Keep all
case-specific code and measurements inside that case.

An implemented case uses this structure:

```text
case-NN/
├── README.md       status, approach, and comparable result summary
├── baseline/       reference evidence and baseline captures
├── optimisation/   single-board implementation and results
└── multiboard/     distributed implementation, tools, and results
```

Do not create speedup comparisons across cases or across different scopes. A
case README must state whether a result covers a kernel, a partial layer, or the
complete Transformer body.

## Index

| Case | Batch | Sequence | Model dim | Heads | FFN dim | Layers | Status |
|---:|---:|---:|---:|---:|---:|---:|---|
| [01](case-01/) | 64 | 128 | 128 | 4 | 128 | 4 | Implemented, complete body: **1.990 s/forward** on-board; 5/5 device PASS |
| [02](case-02/) | 1 | 128 | 128 | 4 | 128 | 4 | Single-board verified, **1.996 s/fwd (21.1×)**; partial multiboard verified |
| [03](case-03/) | 4 | 128 | 128 | 4 | 128 | 4 | Implemented, complete body: **1.990 s/forward** on-board; 5/5 device PASS |
| [04](case-04/) | 16 | 128 | 128 | 4 | 128 | 4 | Implemented, complete body: **1.990 s/forward** on-board; 5/5 device PASS |
| [05](case-05/) | 128 | 128 | 128 | 4 | 128 | 4 | Implemented, complete body: **1.990 s/forward** on-board; 5/5 device PASS |
| [06](case-06/) | 10,000 | 128 | 128 | 4 | 128 | 4 | Not implemented |
| [07](case-07/) | 64 | 128 | 32 | 4 | 32 | 4 | Implemented, complete body: **0.491 s/forward** on-board; 5/5 device PASS |
| [08](case-08/) | 64 | 128 | 1,024 | 4 | 1,024 | 4 | Not implemented |
| [09](case-09/) | 64 | 128 | 128 | 1 | 128 | 4 | **Not measurable on one XIAO** — DRAM link overflows by 73,072 B (needs 394,424 B) |
| [10](case-10/) | 64 | 128 | 128 | 2 | 128 | 4 | **Not measurable on one XIAO** — DRAM link overflows by 23,920 B (needs 345,272 B) |
| [11](case-11/) | 64 | 128 | 128 | 16 | 128 | 4 | Implemented, complete body: **2.462 s/forward** on-board; 5/5 device PASS |
| [12](case-12/) | 64 | 32 | 128 | 4 | 128 | 4 | Implemented, complete body: **0.493 s/forward** on-board; 5/5 device PASS |
| [13](case-13/) | 64 | 1,024 | 128 | 4 | 128 | 4 | Not implemented |
| [14](case-14/) | 32 | 100,000 | 1,024 | 16 | 1,024 | 2 | Not implemented |

The shared official
[`problem statement`](../COMPETITION_RULES.MD) and
[`PyTorch reference`](../torch_transformer_benchmark.py) remain at the
repository root. Supporting microbenchmarks and small-model demonstrations are
in [`experiments/`](experiments/), outside the official case results.
