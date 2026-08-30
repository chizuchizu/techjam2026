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

Measured cross-case run of the case-2 optimised firmware on cases 1–5:
[`case2_code_on_cases_1_to_5.md`](case2_code_on_cases_1_to_5.md).

## Index

| Case | Shape `(B,S,D,H,F,L)` | Baseline, 1 board | Optimised, 1 board | 2 boards | vs baseline | Multiboard split | Gate |
|---:|---|---:|---:|---:|---:|---|---|
| [01](case-01/) | `(64,128,128,4,128,4)` | 2,697.6 s * | 127.36 s | **63.7 s** | **42.3x** | data parallel, 2.00x | 64/64 forwards PASS |
| [02](case-02/) | `(1,128,128,4,128,4)` | 42.15 s | 1.990 s | **1.276 s** | **33.0x** | token-row split, 1.56x | 25/25 host, 0 failing elements |
| [03](case-03/) | `(4,128,128,4,128,4)` | 168.6 s * | 7.96 s | **4.0 s** | **42.1x** | data parallel, 2.00x | 4/4 forwards PASS |
| [04](case-04/) | `(16,128,128,4,128,4)` | 674.4 s * | 31.84 s | **15.9 s** | **42.4x** | data parallel, 2.00x | 16/16 forwards PASS |
| [05](case-05/) | `(128,128,128,4,128,4)` | 5,395.2 s * | 254.72 s | **127.4 s** | **42.3x** | data parallel, 2.00x | 128/128 forwards PASS |
| [06](case-06/) | `(10000,128,128,4,128,4)` | - | - | - | - | Not implemented | - |
| [07](case-07/) | `(64,128,32,4,32,4)` | - | **0.475 s/fwd** | - | - | - | 25/25 PASS (worst 1.46e-03) |
| [08](case-08/) | `(64,128,1024,4,1024,4)` | - | - | - | - | Not implemented | - |
| [09](case-09/) | `(64,128,128,1,128,4)` | - | **2.157 s/fwd** | - | - | - | 25/25 PASS (worst 1.24e-03; SRAM fixed) |
| [10](case-10/) | `(64,128,128,2,128,4)` | - | **2.165 s/fwd** | - | - | - | 25/25 PASS (worst 1.24e-03) |
| [11](case-11/) | `(64,128,128,16,128,4)` | - | **2.166 s/fwd** | - | - | - | 25/25 PASS (worst 1.24e-03) |
| [12](case-12/) | `(64,32,128,4,128,4)` | - | **0.529 s/fwd** | - | - | - | 25/25 PASS (worst 1.14e-03) |
| [13](case-13/) | `(64,1024,128,4,128,4)` | - | - | - | - | Not implemented | - |
| [14](case-14/) | `(32,100000,1024,16,1024,2)` | - | - | - | - | Not implemented | - |

All times are for the **whole case**: one forward for case 2, the full batch of
B inputs for the others. Every figure is the device's own measurement of the
complete four-layer body; host serial transfer is excluded throughout, on all
three columns alike.

`*` Cases 1, 3, 4 and 5 were never run on the pre-optimisation firmware, so
their baseline is **estimated** as `B x 42.15 s` from case 2's measured
starting point. Only case 2's baseline is a measurement. The optimised and
two-board columns are measured everywhere.

Cases 7, 9, 10, 11 and 12 each stream `B=64` independent inputs, so the
"Optimised, 1 board" figure above is **one complete four-layer forward** as
measured by the device (0.475 / 2.157 / 2.165 / 2.166 / 0.529 s). The
whole-case totals are the derived `B x` projection — 30.4 / 138.1 / 138.6 /
138.6 / 33.9 s — and are intentionally not reported as measurements; see each
case README for the raw 25-seed captures. Cases 8 and 13 have no consumable
firmware in this workspace and remain not implemented.


**Multiboard split** names which decomposition the two-board column used. Cases
1, 3, 4 and 5 are batches of independent forwards, so the boards run
`B/2` inputs each and exchange nothing - exactly 2.00x. Case 2 is a single
forward (`B=1`) and has to be split *inside* the model, by token row, which
costs one K/V exchange per layer and does not halve the per-board weight
streaming - hence 1.56x rather than 2x. See
[`case-02/multiboard/esp32-cluster-full/`](case-02/multiboard/esp32-cluster-full/)
and [`batch-dp/`](batch-dp/).

The shared official
[`problem statement`](../COMPETITION_RULES.MD) and
[`PyTorch reference`](../torch_transformer_benchmark.py) remain at the
repository root. Supporting microbenchmarks and small-model demonstrations are
in [`experiments/`](experiments/), outside the official case results.
