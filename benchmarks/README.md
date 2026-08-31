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

| Case | Shape `(B,S,D,H,F,L)` | Baseline, 1 board | Optimised, 1 board | 2 boards | 4-node WiFi DP | 8-node WiFi DP | vs baseline | Multiboard split | Gate |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| [01](case-01/) | `(64,128,128,4,128,4)` | 2,697.6 s * | 127.36 s | **63.7 s** | **67.465 s** | **33.713 s** | **80.0x** | data parallel, 8.00x | 64/64 WiFi forwards PASS |
| [02](case-02/) | `(1,128,128,4,128,4)` | 42.15 s | 1.990 s | **1.276 s** | **4.2137 s †** | **4.220 s †** | **33.0x** | token-row split, 1.56x | Physical WiFi gate PASS |
| [03](case-03/) | `(4,128,128,4,128,4)` | 168.6 s * | 7.96 s | **4.0 s** | **4.215 s** | **4.218 s ‡** | **42.1x** | data parallel, 4.00x max | 4/4 WiFi forwards PASS |
| [04](case-04/) | `(16,128,128,4,128,4)` | 674.4 s * | 31.84 s | **15.9 s** | **16.853 s** | **8.438 s** | **79.9x** | data parallel, 8.00x | 16/16 WiFi forwards PASS |
| [05](case-05/) | `(128,128,128,4,128,4)` | 5,395.2 s * | 254.72 s | **127.4 s** | **134.887 s** | **67.451 s** | **80.0x** | data parallel, 8.00x | 128/128 WiFi forwards PASS |
| [06](case-06/) | `(10000,128,128,4,128,4)` | - | - | - | - | - | - | Not implemented | - |
| [07](case-07/) | `(64,128,32,4,32,4)` | 295.05 s § | **30.427 s** (B=64 full case) | **15.822 s** | **7.922 s** | **3.963 s** | **74.5x** § | data parallel, 7.99x | 64/64 WiFi forwards PASS (worst 1.50e-03) |
| [08](case-08/) | `(64,128,1024,4,1024,4)` | - | - | - | - | - | - | Not implemented | - |
| [09](case-09/) | `(64,128,128,1,128,4)` | 2,697.6 s § | **138.027 s** (B=64 full case) | - | **57.005 s** | **28.508 s** | **94.6x** § | data parallel, 8.00x | 64/64 WiFi forwards PASS (worst 1.26e-03) |
| [10](case-10/) | `(64,128,128,2,128,4)` | 2,697.6 s § | **138.536 s** (B=64 full case) | **119.101 s** | **59.563 s** | **29.793 s** | **90.5x** § | data parallel, 7.999x | 64/64 WiFi forwards PASS (worst 1.25e-03) |
| [11](case-11/) | `(64,128,128,16,128,4)` | 2,697.6 s § | **138.610 s** (B=64 full case) | **206.354 s** | **103.169 s** | **51.604 s** | **52.3x** § | data parallel, 7.999x | 64/64 WiFi forwards PASS (worst 1.31e-03) |
| [12](case-12/) | `(64,32,128,4,128,4)` | 547.95 s § | **33.879 s** (B=64 full case) | **17.091 s** | **8.554 s** | **4.282 s** | **128.0x** § | data parallel, 8.00x | 64/64 WiFi forwards PASS (worst 1.28e-03) |
| [13](case-13/) | `(64,1024,128,4,128,4)` | - | - | - | - | - | - | Not implemented | - |
| [14](case-14/) | `(32,100000,1024,16,1024,2)` | - | - | - | - | - | - | Not implemented | - |

All times are for the **whole case**: one forward for case 2, the full batch of
B inputs for the others. Every non-dash optimised or multiboard timing is the
device's own measurement of the complete four-layer body; host serial transfer
is excluded throughout. Baseline-column projections are marked below.

Only case 2's pre-optimisation baseline is a physical measurement. `*` Cases
1, 3, 4 and 5 change only `B`, so their baseline is the direct batch projection
`B x 42.15 s`. `§` Cases 7 and 9–12 change the model shape and use the
lower-confidence FLOP-normalised projection

```
t_baseline_est = 42.15 s x FLOP_case / FLOP_case_02
```

at case 2's measured baseline effective rate (3.184 MFLOP/s). The exact dense
FLOP ratios are 7x for case 7, 64x for cases 9–11, and 13x for case 12. The
corresponding `vs baseline` entries divide these estimates by the measured
eight-node compute wall. The model does not include shape-dependent baseline
throughput effects such as head-loop and softmax overhead, so these values must
remain marked `§` and must not be described as device measurements. In
particular, the canonical case-9 and case-10 baseline workspaces exceed SRAM
and never produced runnable firmware; their numbers are counterfactual
compute-only projections. Later per-case "first physical capture" records are
hybrid ports and are not the pre-optimisation baseline estimated here.

Four- and eight-node WiFi results cover cases 1–5, 7, and 9–12; case 9 has no
recorded two-node run.

**4-node WiFi DP** is four full-forward replicas running independent batch
inputs over persistent TCP; the column reports compute wall. `†` Case 2 has
B=1, so it uses one active node and receives no data-parallel speedup. Full
method, end-to-end times, and raw JSON:
[`batch-dp/RESULTS_FOUR_C3_WIFI.md`](batch-dp/RESULTS_FOUR_C3_WIFI.md).
**8-node WiFi DP** uses the same decomposition with eight replicas. `‡` Case 3
has only four inputs, so four of the eight available nodes are active. Full
eight-node evidence:
[`batch-dp/RESULTS_EIGHT_C3_WIFI.md`](batch-dp/RESULTS_EIGHT_C3_WIFI.md).

Cases 7, 9, 10, 11 and 12 each stream `B=64` independent inputs, so the
"Optimised, 1 board" figure above is the **measured full-case total**: the sum
of the device's own `us=` forward counters over all 64 streamed frames
(30.427 / 138.027 / 138.536 / 138.610 / 33.879 s). Wall time including host
USB pacing is 70.227 / 262.073 / 262.778 / 262.754 / 73.744 s respectively.
Each full-case run also passed the gate on all 64/64 frames; see each case
README and `optimisation/results/` for the raw captures. Cases 7, 10, 11, and
12 also have complete two-, four-, and eight-node WiFi accuracy runs; case 9
has complete four- and eight-node runs. See each `multiboard/` directory for
the raw JSON. Cases 8 and 13 have no consumable firmware in this workspace and
remain not implemented.

An independent single-board repeat audit for cases 9, 10, and 11 reproduced
their complete B=64 compute totals within **0.53%**, with every output passing.
Their similar ~39% MFU is expected because head count partitions the same
total attention work. Method and raw repeat JSON:
[`SINGLE_BOARD_REPEAT_CASES_09_11.md`](SINGLE_BOARD_REPEAT_CASES_09_11.md).


**Multiboard split** names which decomposition the two-board column used. Cases
1, 3, 4 and 5 are batches of independent forwards, so the boards run
`B/2` inputs each and exchange nothing - exactly 2.00x. Case 2 is a single
forward (`B=1`) and has to be split *inside* the model, by token row, which
costs one K/V exchange per layer and does not halve the per-board weight
streaming - hence 1.56x rather than 2x. See
[`case-02/multiboard/esp32-cluster-full/`](case-02/multiboard/esp32-cluster-full/)
and [`batch-dp/`](batch-dp/).

## Per-device MFU

Model FLOPs Utilisation is the fraction of a board's arithmetic peak that the
measured run actually converts into useful model work. It is the one metric
that compares cases of different shapes, and node counts, on equal terms.

**Model FLOPs** per case are counted analytically from the shape, with one
multiply-accumulate as 2 FLOP:

```
FLOP = 2 · B · L · S · (4·D² + 2·S·D + 2·D·F)
       └ QKV 3·S·D²   └ QKᵀ and PV 2·S²·D   └ out-proj S·D²   └ FFN 2·S·D·F
```

`H` does not appear: splitting `D` into heads repartitions the attention work
without changing its volume. Cases 9, 10, 11 and case 1 therefore have
identical FLOP counts and differ only in `H`.

**Peak** is `160 MFLOP/s` per board `[E]`: 160 MHz, single-issue in-order
RV32IMC at 1 IPC, and 2 instructions per int16×int16→int32 MAC (`mul` then
`add` — RV32IMC has no fused multiply-add and no SIMD). MFU is
`achieved ÷ 160 MFLOP/s`. This peak is a derived bound, not a datasheet
figure; if `mul` is multi-cycle on this core the true peak is lower and every
MFU below is correspondingly conservative.

| Case | Model FLOP | Config | Time | Boards | FLOP/s/device | **MFU/device** |
|---:|---:|---|---:|---:|---:|---:|
| 02 | 0.134 G | baseline, 1 board | 42.15 s | 1 | 3.18 M | 2.0% |
| 01–05 | — | **optimised (opt23), 1 board** | — | 1 | **67.45 M** | **42.2%** |
| 02 | 0.134 G | two-board token-row split | 1.276 s | 2 | 52.59 M | 32.9% |
| 01/03/04/05 | — | two-board data parallel | — | 2 | 67.4 M | 42.1% |
| 01–05 | — | **WiFi DP (opt24 tiled), 4 or 8 nodes** | — | 4–8 | **31.84 M** | **19.9%** |
| 07 | 0.940 G | optimised, 1 board | 30.427 s | 1 | 30.88 M | 19.3% |
| 07 | 0.940 G | two-board WiFi DP | 15.822 s | 2 | 29.71 M | 18.6% |
| 07 | 0.940 G | four-board WiFi DP | 7.922 s | 4 | 29.66 M | 18.5% |
| 07 | 0.940 G | eight-board WiFi DP | 3.963 s | 8 | 29.65 M | 18.5% |
| 09 | 8.590 G | optimised, 1 board | 138.027 s | 1 | 62.23 M | 38.9% |
| 09 | 8.590 G | four-board tiled WiFi DP | 57.005 s | 4 | 37.67 M | 23.5% |
| 09 | 8.590 G | eight-board tiled WiFi DP | 28.508 s | 8 | 37.66 M | 23.5% |
| 10 | 8.590 G | optimised, 1 board | 138.536 s | 1 | 62.01 M | 38.8% |
| 10 | 8.590 G | two-board tiled WiFi DP | 119.101 s | 2 | 36.06 M | 22.5% |
| 10 | 8.590 G | four-board tiled WiFi DP | 59.563 s | 4 | 36.05 M | 22.5% |
| 10 | 8.590 G | eight-board tiled WiFi DP | 29.793 s | 8 | 36.04 M | 22.5% |
| 11 | 8.590 G | optimised, 1 board | 138.610 s | 1 | 61.97 M | 38.7% |
| 11 | 8.590 G | two-board tiled WiFi DP | 206.354 s | 2 | 20.81 M | 13.0% |
| 11 | 8.590 G | four-board tiled WiFi DP | 103.169 s | 4 | 20.82 M | 13.0% |
| 11 | 8.590 G | eight-board tiled WiFi DP | 51.604 s | 8 | 20.81 M | 13.0% |
| 12 | 1.745 G | optimised, 1 board | 33.879 s | 1 | 51.50 M | 32.2% |
| 12 | 1.745 G | two-board WiFi DP | 17.091 s | 2 | 51.05 M | 31.9% |
| 12 | 1.745 G | four-board WiFi DP | 8.554 s | 4 | 51.00 M | 31.9% |
| 12 | 1.745 G | eight-board WiFi DP | 4.282 s | 8 | 50.94 M | 31.8% |

Per-case FLOP counts: case 01/09/10/11 8.590 G, case 02 0.134 G, case 03
0.537 G, case 04 2.147 G, case 05 17.180 G, case 07 0.940 G, case 12 1.745 G.
Counting only unmasked causal attention instead of the dense `S²` scales every
MFU above down by roughly 13% (case 02: 42.2% → 36.9%).

Four things this exposes that the wall-time columns do not:

1. **The opt23 firmware holds 42.2% MFU across cases 1–5 to within 0.1
   point.** Those cases share one shape and differ only in `B`, so a constant
   MFU is evidence that batching adds no per-input overhead — the 8.0x baseline
   ratio is real work, not amortised setup.
2. **Data-parallel scaling is free; the WiFi firmware is not.** For cases 1–5,
   the two-board DP rows hold 42.1% per device, so 2.00x is genuine. Their
   4- and 8-node WiFi rows run at **19.9% per device — less than half** —
   because they use the memory-first row-tiled `opt24` build (4.214 s/forward
   vs 1.996 s) needed to fit the WiFi stack in SRAM. The 8.00x node scaling is
   therefore bought at a 2.1x per-device efficiency loss; against the *best*
   single board the true eight-node gain is 3.78x, not 8.00x.
3. **Case 2's token-row split costs 9.3 MFU points** (42.2% → 32.9%), which is
   the per-layer K/V exchange and the unhalved weight streaming made
   quantitative — the same reason it reaches 1.56x rather than 2.00x.
4. **Shape penalties are visible directly.** Case 7 (`D=F=32`) drops to 19.3%,
   confirming the narrow-kernel overhead its README predicts; case 12 (`S=32`)
   to 32.2% for short-sequence overhead; cases 9/10/11 sit ~3.4 points below
   case 1 at identical FLOPs, so `H≠4` costs about 8% purely in head-loop
   scheduling. The tiled WiFi path makes head-count overhead clearer still:
   cases 9, 10, and 11 sustain 23.5%, 22.5%, and 13.0% MFU per device.

For the same metric on a laptop, a consumer GPU and an H200 — where an H200
running case 2 sits below 1% MFU — see
[`../docs/HARDWARE_COMPARISON.md`](../docs/HARDWARE_COMPARISON.md).

The shared official
[`problem statement`](../COMPETITION_RULES.MD) and
[`PyTorch reference`](../torch_transformer_benchmark.py) remain at the
repository root. Supporting microbenchmarks and small-model demonstrations are
in [`experiments/`](experiments/), outside the official case results.
