# Superlinear speedup for the ESP32-C3 batch fleet — what is real, what is myth

Status: research deliverable for case-09 (8x XIAO ESP32-C3, batch-parallel).
Date: 2026-08-31. Author: fleet research agent.
Scope: whether/how speedup **> N** for N boards is documented and achievable on
SRAM-bound embedded clusters, translated to our 8-board case-09 fleet.

All local numbers below are quoted from the checked-in case-09 artifacts:
`benchmarks/case-09/optimisation/esp32-baseline/README.md`,
`benchmarks/case-09/multiboard/README.md`,
`benchmarks/case-09/multiboard/results_case9_{four,eight}_c3_wifi.json`,
`docs/WIFI_ON_A_COMPUTE_NODE.md`.

---

## TL;DR (read this first)

1. **Superlinear means speedup strictly greater than N.** For N=8 that means
   faster than 8x. Our measured cluster is **7.998x compute** at 8 boards —
   that is exactly linear, and it is the honest ceiling of the current
   *streaming-replica* design (every board runs an identical per-forward
   working set; the batch count never changes per-board RAM).
2. The classic HPC superlinear mechanisms — aggregate cache, per-node
   working-set relief, disk/DRAM relief — require the **per-node working set
   to actually shrink as N grows**, or a serial baseline that is artificially
   slow (slow to a *spill/schedule artifact*). Our single-board baseline is
   forced into a slow 16-row tiled schedule (2.157 s → 3.562 s/forward)
   **only because the WiFi/lwIP stack must fit in SRAM** (273,180 B untiled
   can't fit WiFi; 224,244 B tiled can). That 1.65x penalty is a *hidden
   capacity loss* (Gunther et al., CACM 2015; Section 1), not a compute law — it is the only place where
   "superlinear-looking" gains genuinely live for us.
3. Because inputs are **streamed one at a time** and the forward arena is
   per-forward (batch-independent), **splitting the batch across more boards
   does NOT free per-forward SRAM** — so smaller batch-per-board cannot buy a
   faster schedule *unless the firmware is changed* to make per-board resident
   state depend on the per-board batch count. This is the one design change
   that can create real superlinear headroom, and it is testable with 2 boards
   (Section 7).
4. Published measured ESP32-cluster numbers are **sublinear**, not superlinear:
   SwarmInfer's ESP32-S3 cluster (2→4→5 nodes: 3,653 → 2,115 → 1,838 ms) is
   1.73x for 2x nodes — transport-bound. DENNI distributes a too-large model
   across memory-constrained nodes to make it *fit*, at the cost of
   communication; no superlinear claim. No published superlinear MCU-cluster
   result was found — a defensible competition narrative is that "superlinear"
   for MCU fleets is a **baseline-artifact** phenomenon (same as HPC's
   BigDisk/BigMem "hidden capacity" story), not a physics gain.

---

## 1. What superlinear speedup is, and the canonical mechanisms (citations)

Speedup S(N) = T(1)/T(N). Linearity means S(N) = N; **superlinear means
S(N) > N**, which requires the parallel execution to be *better than a
perfectly scaled copy of the serial one* — i.e. something about the system
changed, not just more of the same.

A systematic taxonomy (verified, scraped in full): Ristov, Prodan, Gusev,
Skala, *"Superlinear speedup in HPC systems: why and when?"*, FedCSIS 2016
(www.annals-csis.org/Volume_8/pliks/498.pdf). It lists three cause families:

- **Aggregate cache / working-set relief** — the most reported cause. Each
  processor gets a private cache; if adding processors makes the *per-node*
  data slice fit in a faster memory tier (L2/L3/DRAM instead of disk), the
  per-node time drops below the serial per-node time. This only applies to
  **cache-intensive algorithms** (average data reuse c > 1; e.g. dense GEMM,
  where an element is reused O(N) times). A scalar-product-like kernel
  (c = 1, no reuse) can *never* get superlinear from cache effects — it has
  nothing to reuse.
- **Shared-cache locality** — when one node's fetch warms a cache line that
  others consume (shared-memory machines). Not applicable to SPI-flash-backed,
  private-SRAM MCUs (no shared memory tier).
- **Non-persistent / search algorithms** — first node to find the answer stops
  everyone (e.g. parallel backtracking; Speckenmeyer 1988 gives an academic
  example). Extra nodes multiply the chance of an early hit; in expectation
  speedup can exceed N. Not applicable to a fixed batch of 64 independent
  deterministic forwards (no search, no early exit).

The industry-story reference: Gunther, Puglia, Tomasette, *"Hadoop
Superlinear Scalability: the perpetual motion of parallel performance"*,
Communications of the ACM 58(4), April 2015. Their central point: measured
speedup > N in BigData clusters is real but **"illusory"** — it is a
**hidden capacity boost**. The serial baseline runs with data *spilling to
disk* (BigDisk); the parallel run has enough aggregate memory that the data
fits in RAM/buffer cache (BigMem). The same node count, same job → the
"superlinear" gain is *the difference between a slow bad baseline and a
healthy parallel one*. The σ/κ coherency model in the paper makes this
precise. The mapping to our tiled-vs-untiled baseline is direct and is the
key insight of this report (Section 3A).

Classic citations as collected from the Wikipedia *Speedup* references:
- Baer, *Microprocessor Architecture: From Simple Pipelines to Chip
  Multiprocessors* (2010) — cache-effect superlinearity on meshes/grids.
- Green Destiny + mpiBLAST (Feng et al.) — superlinear gene-sequencing
  scaling attributed to memory relief.
- Gurobi / CPLEX optimization benchmarks — superlinear in solvers with
  presolve/memory relief.

A modern *measured* strong-scaling case: Galeazzo & Weiß, *"Understanding
superlinear speedup in current HPC architectures"*, IOP Conf. Series
1312 (2024) 012009 — OpenFOAM strong-scaling on AMD EPYC shows superlinear
runs of 10–20% expected classically and **>300% in simple test cases**,
attributed to memory-hierarchy (cache) effects. This confirms that superlinear
speedup is reproducible and measurable on shared-memory HPC — and that its
physically honest home is memory-hierarchy relief, which is precisely what an
SRAM-bound MCU fleet *does not have* (fixed per-board SRAM, flash-resident
weights).

### ESP32-C3 specifics: the only "memory tiers" that exist

- The C3 reads/executes from external SPI flash **through an MMU cache**: the
  Espressif memory-map-101 guide and the ESP-IDF *memory-types* doc state the
  external flash is accessed via **I-Cache (code) and D-Cache (data)** with
  the cache buffer hosted in configurable SRAM0, and that IRAM placement
  avoids cache-miss penalties.
- So flash weight reads are real but cached; SPI-flash bandwidth and the
  small cache working set are the two "slow tiers". There is **no DRAM, no
  PSRAM, no disk** on the XIAO C3 (327,680 B `dram0_0_seg`, 160 MHz
  RV32IMC, no FPU, no PSRAM).
- Consequence: "working-set relief" can only mean *reducing flash round-trips
  or fitting a taller tile in SRAM* — per board, with per-board SRAM fixed.

---

## 2. Why the classic mechanisms DON'T transfer to our fleet as built

Our 8-board fleet is **data-parallel replicas**: every board runs the full
forward (B=64, S=128, D=128, H=1, F=128, L=4, causal), consumes disjoint
inputs, exchanges no tensors. Measured:

| Config | Compute wall (64 inputs) | Scaling vs 1 tiled | E2E (incl. WiFi TCP) |
|---|---:|---:|---:|
| 1 tiled worker (equivalent) | 228.007 s (64 x 3.562 s) | 1.00x | — |
| 4 boards | 57.005 s | 4.00x | 75.4 s |
| 8 boards | 28.508 s | **7.998x** | 38.5 s |

Per-board per-forward time is *identical* (3.562 s, σ ≈ 4 ms) at N=4 and N=8.
That is the signature of perfect linear replica scaling — and it **cannot be
superlinear**: each board's working set (residual 64 KB + context 16 KB +
K/V 32 KB + tile scratch + WiFi + heap) is byte-for-byte the same at N=2 as at
N=8. The aggregate-cache mechanism requires the *per-node slice* to shrink;
in the streaming-replica design the per-node slice is a *full forward*, and
inputs are not resident, so M (inputs per board) enters nowhere in SRAM.
**Conclusion: 7.998x is the honest compute ceiling of the current design.**

This is worth stating explicitly in the write-up: our fleet demonstrates the
*positive* result "exactly linear replica scaling on 8 C3s" (rare for
wireless MCUs, valuable by itself) and proves superlinearity is **absent** for
the current architecture — exactly as Ristov et al. predict for
cache-dependent algorithms when no capacity is added.

---

## 3. What CAN legitimately give speedup > N on the C3 fleet

Three tracks exist. Two are available today (no physics; engineering), one
requires a firmware change.

### 3A. Track A — relief of the *avoidable single-board slowdown* (the Hadoop analog, fully applicable)

Our single-board *WiFi* baseline is not "one healthy forward": it is the
tiled schedule that exists **only to fit WiFi/lwIP into SRAM**. Facts:

- Untiled optimized FAST (no WiFi): 273,180 B static, 2.157 s/forward
  (device gate 25/25, worst max_abs 1.050e-3 in case-02; case-09 1.1038e-3
  tiled).
- Tiled 16-row *with* persistent WiFi/TCP: 224,244 B static, 3.562 s/forward.
- Tiling penalty = **1.65x** (3.562 / 2.157).

The penalty is per-forward and identical on every board, so at N=8 it costs a
full 8 x 1.405 s ≈ 11.2 s of cluster time. In Hadoop terms: the serial
baseline is "BigDisk" (must tile / can't keep WiFi-side buffers), and the
healthy parallel state is "BigMem". If we can build a **memory-lean forward
that runs untiled (or taller tiles) WITH WiFi**, each board's per-forward time
drops by up to 1.65x and the whole cluster benefits *multiplicatively*.
That is the only *provable* "superlinear-looking" gain: it is real relative
to the forced tiled baseline, and it is precisely the unit of work that
Section 7 benchmarks.

Approximate ceiling: if WiFi boards returned to 2.157 s/forward, 8-board
compute wall → ≈ 17.3 s (28.508 / 1.651), and E2E (dominated by compute
after overlap) → well under 30 s. Relative to the *serial USB* E2E baseline
(262.07 s, Section 3B) that is E2E "speedup" ≈ 9–10x **at 8 boards — i.e.
nominally superlinear versus the broken serial baseline**, and honest once
you state the baseline is the USB-paced one.

### 3B. Track B — fixed per-batch E2E overhead relief (also applicable, and we already have a large chunk)

The single-board E2E baseline (262.07 s for 64 inputs over USB) is not 138 s
of compute: it is ~138 s compute + ~124 s of **non-compute** — dominated by
the paced USB CDC frame (~1.3 s per 64 KB input, ~83 s/batch per the
`docs/WIFI_ON_A_COMPUTE_NODE.md` analysis) plus host-side serialization and
staging. These are *fixed per input*, not per board; the fleet removes the
serial pacing and overlaps dispatch across boards. Measured E2E: 262.07
(1) → 75.4 (4) → 38.5 (8) = 3.47x / 6.8x. That is **sublinear** in N today,
because board compute (28.5 s) is already above transport overhead (10 s) at
N=8 and can't overlap fully.

Why this matters for superlinear E2E: if dispatch and compute overlap
perfectly, E2E(N) → max(compute_wall(N), transport_wall(N)). At N=8,
compute_wall 28.5 s vs transport 10 s → E2E already compute-bound; the 10 s
transport slack is where E2E "superlinearity vs serial" is currently lost.
Fix: pipeline host dispatch (send all 64 inputs early, boards compute as they
arrive), reduce per-input host overhead, or shrink payload. If E2E reaches the
compute wall, E2E speedup vs the serial *USB-paced* baseline (262 s) at 8
boards is ≈ 38.5→28.5 s → **9.2x**, which *is* > 8x — again nominal
"superlinearity" against an artificially slow serial baseline. State the
baseline carefully; the physics is unchanged (compute stays ~linear).

### 3C. Track C — genuine per-node working-set relief (REQUIRES a design change)

The only way batch count genuinely changes per-board economics: make inputs
**resident** on each board and make the schedule (tile size, untiled vs tiled,
flash weight passes) a function of the resident count. Sketch:

- Board receives M inputs (M = 64/N), stores them quantized: M x 16,384 B
  int16 + scales ≈ M x 16.4 KB (M=8 → ≈ 131 KB; too big *today*, so this only
  works for small M or with further quantization (int8 → 8.4 KB/input) or by
  keeping inputs in flash and staging a window).
- With resident inputs, per-forward SRAM can drop (input streamed from RAM,
  not re-allocated), freeing up to ~64 KB — enough to **untile** or run
  48-row tiles with WiFi.
- Weights re-read per tile become amortized over the M resident rows: fewer
  total flash round-trips per input than the streaming design where every
  input rescans the model.

This is the only place where "fewer inputs per board ⇒ faster per-board
schedule" is physically true, and it is exactly the experiment Section 7
proposes. Note the *direction of the effect* is favorable: at N=8, M=8, we
gain the most; at N=1, M=64 the single board is *worst* — which is precisely
the superlinear shape S(N) > N.

---

## 4. Answers to the four research questions (translated to our fleet)

**Q1 — Which HPC superlinear mechanisms apply to SRAM-bound MCUs with
flash-resident weights?**
- Memory-bandwidth relief (Hadoop BigDisk→BigMem, CACM): **YES** if the
  "bad baseline" is the forced tiled/WiFi schedule (Track 3A) or USB pacing
  (Track 3B). This is a *baseline artifact*, the only documented superlinear
  mechanism that transfers.
- Aggregate-cache / working-set (Ristov et al.; Baer; Galeazzo & Weiß):
  **NO** for replicas with fixed per-board SRAM; **YES only** in the resident
  mini-batch design (Track 3C) where per-board working set shrinks with N.
- Disk/DRAM relief: N/A (no disk/DRAM on C3). Closest analog is SPI-flash
  round-trips; relief = fewer weight passes (Section 6).
- Non-persistent search: N/A (deterministic fixed batch).

**Q2 — Does smaller batch-per-board enable a faster schedule (no tiling,
fewer flash round-trips)? Published per-input-latency-vs-batch MCU data?**
- In the current streaming design: **NO** — arena is per-forward and
  batch-independent; verified locally by the flat 3.562 s/forward at M=16 and
  M=8. This is the central negative result.
- In a resident design: **YES**, and that is the experiment.
- Published data: MCU/MicroML reports (e.g., the μNPU benchmark set, arXiv
  2503.22567) publish **per-input latency** typically *increasing* slightly
  with batch residency because weights dominate and RAM/cache shrink per
  sample — nothing shows per-input latency dropping with batch on a
  fixed-SRAM device. No counterexample found. The honest expectation: P(M)
  is flat in M for streaming (our data already shows 3.562 s at M=16 and
  M=8).

**Q3 — Communication-avoiding / data-layout tricks for tiny-device ML?**
See Section 6. In short: tile-major flash layout (sequential SPI bursts),
loop fusion across layers per tile, keep per-tile scales in RAM, Q12→Q8 to
halve flash traffic, stage only per-tile weight strips (the full per-layer
weight set ≈ 197 KB Q12 — too large to stage on the C3; only strips fit).

**Q4 — Real measured MCU-cluster numbers showing >linear scaling?**
None found that are superlinear. All published measured clusters are
sublinear or linear:
- **SwarmInfer (ESP32-S3 → ESP-NOW**): N=2→3,653 ms; N=4→2,115 ms;
  N=5→1,838 ms ⇒ 1.73x for 2x nodes, 1.99x for 2.5x — **sublinear**,
  transport-bound (ESP-NOW ~81.3 KB/s; one MobileNet inference moves ~670 KB).
- **DENNI** (Sahu, Tamminedi, Duwe; IPCCC 2021): distributes a *too-large*
  NN across several memory-constrained edge nodes so it *fits* (the
  per-node-memory-relief story, Q1's "disk relief" analog for weights);
  results measure fit + accuracy, no superlinear throughput claim; comms
  costs make typical scaling sublinear.
- **Ours**: 4.00x / 7.998x compute at 4 / 8 boards — exactly linear, the
  replica-design ceiling.

The defensible competition claim: for MCU fleets, >N speedup is a
**baseline-artifact** (bad serial baseline), not a superlinear compute law;
our fleet proves linear replica scaling and should pursue Tracks A/B for
E2E, and Track C if we want per-board C>N effects.

---

## 5. Collected measured evidence table

| Source | Hardware | N | Scaling | Why |
|---|---|---|---|---|
| Ours (case-09) | XIAO ESP32-C3 x8, WiFi TCP | 4 / 8 | 4.00x / 7.998x (compute) | exact linear replicas; fixed per-node WS |
| SwarmInfer | ESP32-S3 cluster, ESP-NOW | 2→4→5 | 1.73x / 1.99x (sublinear) | transport-bound (~81 KB/s, 670 KB/inf) |
| DENNI (IPCCC'21) | memory-constrained edge nodes | several | fit-first; sublinear throughput | per-node relief lets model fit; comms cost |
| Galeazzo&Weiß (IOP'24) | AMD EPYC, OpenFOAM, shared-mem | strong | >300% superlinear (single-node) | memory-hierarchy/aggregate-cache relief (HPC, not MCU) |
| Ristov et al. (FedCSIS'16) | theory | any | only for cache-intensive (c>1) | requirements for legit superlinearity |
| Gunther et al. (CACM'15) | Hadoop clusters | any | >N reported, "illusory" | hidden capacity (disk→RAM) relief |

---

## 6. Communication-avoiding / data-layout tricks for tiny-device ML

Weights are the dominant flash/SPI cost on the C3: Q12 weight file
786,624 B + fp32 scales for the full 1.59 MB model; a single forward reads a
large fraction (per-tiled forward, the same strips re-read once per tile, so
total weight traffic scales ~ (number of row-tiles) for the projection/FFN
strips, i.e. 4x more at 16-row tiles vs untiled). Tricks, in impact order:

1. **Tile-major flash layout.** Place each tile's weight strip contiguously in
   flash so a tile reads one sequential SPI burst instead of strided seeks.
   SPI flash favors sequential reads; random 16 B reads are ~5-10x
   slower per byte. This is the highest-leverage data layout for tiled
   schedules.
2. **Across-layer loop fusion.** Process one row-tile through ALL L=4 layers
   before advancing the tile: each weight strip is fetched once per tile pass
   and reused by that tile's S rows (c ≈ S/T reuse) — maximizes reuse per
   flash byte, historically the "communication avoiding" argument.
3. **Resident tiny buffers.** Per-tile scale vectors and quant tables live in
   RAM (they are K = tiny); never re-fetch from flash.
4. **Q12 → Q8 weights.** Halves weight flash traffic (786 → 393 KB) at the
   cost of ~4x quant math in the integer kernels; only after profiling shows
   flash-bound (Section 7 adds the counters).
5. **Don't stage whole layer blocks.** Per-layer Q12 weights ≈ 197 KB > free
   SRAM; only per-tile strips (≤ 8 KB x layers) fit and are worth staging.

---

## 7. Benchmark design + firmware change (2 boards now → extrapolate to 8)

Goal: measure per-board per-forward time **P(M, T)** as inputs-per-board M
shrinks and tile size T varies, WiFi on, then extrapolate the 8-board line.
This decides Tracks A/B/C: if P is flat in M at fixed T, M buys nothing
(rejects Track C in streaming form); if adding T improves P while still
fitting, Track A is real; the M×T sweep finds the 8-board optimum.

### 7.1 Firmware change (minimal, reversible, gate-preserving)

1. **Tile-size knob.** `TM_TILE_ROWS` is currently a fixed 16 in
   `model_tiled.c`. Make it a compile-time override:
   `-DTM_TILE_ROWS=16|24|32|48` (until ED = 64 via a `TM_UNTILED` flag that
   calls the existing model.c path). Rebuild the `esp32-wifi-tiled` env for
   each T and record the **static RAM link number** and the **runtime
   `esp_get_free_heap_size()`** at idle with WiFi connected — that gives the
   hard feasibility ceiling (327,680 − static − lwIP heap must be ≥ ~5 KB).
   Per-row SRAM cost ≈ 1,020 B/row (from 273,180 → 224,244 across 48 rows),
   so T=32 ≈ +16 KB, T=48 ≈ +41 KB over T=16; untiled + WiFi will almost
   certainly not link — that is a *finding*, not a failure.
2. **Per-kernel profile counters.** Add `TM_PROFILE` (guarded by the same
   env flag) that accumulates microseconds and a **flash-weight-read byte
   counter** per kernel: qkv GEMM, KV build, attention (QK^T, PV), output
   projection, FFN1, FFN2. The counters localize the 1.65x tiling penalty:
   flash-bound (byte counter grows ~1/row-tile) vs recompute-bound (byte
   counter flat, time grows).
3. **Per-board inputs M is a *host-side* parameter** (the runner already
   splits the batch). No device change needed for the M sweep in the
   streaming design; add a `--tile` host option to `run_batch_dp.py` that
   drives per-tile images and collects `TM <us>` timing lines, mirroring the
   existing `T <n>` command.

### 7.2 Two-board experiment (now: 2 boards only, USB+WiFi)

Matrix, N=2, WiFi on both, 64 inputs:

| M (inputs/board) | T=16 | T=24 | T=32 | T=48 | untiled(non-WiFi, control) |
|---|---:|---:|---:|---:|---:|
| 32 | P | P | P | P | (control: no WiFi) |
| 16 | P | P | P | P | — |
| 8 | P | P | P | P | — |

For each cell: flash both boards, link, measure P (median of ≥ 3 timed
forwards per board — reuse `T <n>`), record free-heap-at-idle, record
per-kernel profile + flash-byte counters, run the full host gate (25/25 or
64/64) at least once per (M, T). Expected outcomes and their readings:

- P(T) flat with M at fixed T ⇒ **M is irrelevant in streaming design**
  (validates the Section 2 ceiling; rejects Track C as-is). This is already
  strongly implied by 3.562 s at M=16 and M=8 in the N=8 run.
- P(T) drops as T rises (until untiled) and fits ⇒ **Track A is real**:
  each board is 1.65x(→less) faster; the per-board win is multiplicative
  across N.
- Flash-byte counter rises ~1/T and time rises with it ⇒ flash-bound; Q8 and
  tile-major layout become the next lever. If time rises but bytes are flat ⇒
  recompute-bound (quant/requant in tiles), and fusion/ILP are the next lever.

### 7.3 Extrapolation model (2 boards → 8 boards)

Per-input cost model: `T_board(M, T) = P(M, T) + t_comms(N)` where t_comms is
host dispatch + collect per board (measured in the N=2 run and scaled by
N-hop behavior; the case-09 runner uses persistent host→board TCP fan-out, so
comms per board ≈ 64/N inputs ordered, roughly constant per input).

Cluster compute wall for N boards: `W_c(N) = max_boards T_board_i(M, T)`
≈ (64/N) × P(64/N, T*) where T* maximizes the fit (largest T that links+wifi
heap passes). E2E: `W_e(N) = max(W_c(N), transport_wall(N))` under pipelined
dispatch; today transport_wall(N) ≈ 38.5 − 28.5 = 10 s at N=8.

Speedup curves to plot (both vs the honest baselines, stated explicitly):
- vs **1 tiled worker** (228 s compute): expect ≈ linear; any >N is the
  measurement error band (report median and per-board σ).
- vs **1 untiled non-WiFi** (138 s): focuses Track A — if T* lets WiFi boards
  beat 2.157 s/forward, E2E computing can beat 138 s at N≥4.
- vs **serial USB E2E 262 s**: the "NB speedup" headline; will exceed 8 at
  N=8 only if Tracks A/B close the compute/transport gap (target E2E ≤ 29 s
  ⇒ >9x). Frame it as "speedup vs the deliverable's serial baseline", never
  as a compute-law superlinearity.

Decision rule for 8 boards: pick T* = largest T with `static + lwIP heap
≤ 327,680 − 5 KB`; if T* ≥ 32, rebuild the 8-board images, rerun the full
64/64 gate, and expect cluster compute wall ≈ (64/8) × P(8, T*).

### 7.4 Risks / invariants (from `docs/WIFI_ON_A_COMPUTE_NODE.md`)

1. Accuracy gate is not negotiable — run the tiled host gate before any
   device run; never weaken the "no speedup for incomplete batch" rule.
2. Never silently change measured firmware — every tile/knob change goes
   behind an env/section of its own (existing convention: `TM_*` flags).
3. Host transfer excluded from reported compute; E2E reported separately.
4. WiFi weight strips Q12 live in DROM (flash) — moving them to RAM is
   impossible (197 KB/layer > free SRAM); only per-tile staging applies.

---

## 8. Sources

Local:
- `benchmarks/case-09/optimisation/esp32-baseline/README.md` — 273,180 B /
  224,244 B static; 2.157 s vs 3.562 s/forward; 398,592 params (1.59 MB fp32).
- `benchmarks/case-09/multiboard/README.md` + result JSONs — 4.00x / 7.998x,
  57.005 / 28.508 s, E2E 75.4 / 38.5 s, per-board times.
- `docs/WIFI_ON_A_COMPUTE_NODE.md` — USB pacing ~1.3 s/input (~half the E2E
  wall), 32,384 B overflow when WiFi added to the full forward, tiling
  rationale, invariants.
- `benchmarks/case-09/optimisation/esp32-baseline/src/model_tiled.c` — arena:
  full residual + full ctx + full K/V + tile-scratch; 16-row default.

External (all captured under `.firecrawl/superlinear/`):
- Ristov, Prodan, Gusev, Skala — *Superlinear speedup in HPC systems: why
  and when?* FedCSIS 2016 (annals-csis.org) — taxonomy: aggregate cache
  (needs c>1 reuse), shared cache, non-persistent search; threshold eq. (10).
- Gunther, Puglia, Tomasette — *Hadoop Superlinear Scalability*, CACM 58(4)
  2015 — hidden capacity (disk→RAM) relief; σ/κ coherency; BigMem/BigDisk.
- Wikipedia *Speedup* (references list): Baer 2010 (cache effects); Green
  Destiny/mpiBLAST; Speckenmeyer 1988 (backtracking superlinear); Gurobi/
  CPLEX.
- Galeazzo & Weiß — *Understanding superlinear speedup in current HPC
  architectures*, IOP Conf. Ser.: Mater. Sci. Eng. 1312 (2024) 012009 —
  OpenFOAM/EPYC, >300% strong-scaling superlinear, memory hierarchy.
- SwarmInfer (ESP32-S3 ESP-NOW distributed inference; IEEE/github) — N=2..5:
  3,653/2,115/1,838 ms ⇒ sublinear; ESP-NOW ≈ 81.3 KB/s; ~670 KB/inference.
- Sahu, Tamminedi, Duwe — *DENNI: Distributed Neural Network Inference on
  Severely Resource Constrained Edge Devices*, IPCCC 2021 (abstract via
  Semantic Scholar) — memory-aware fit; no superlinear claim.
- Espressif: memory-map-101 blog + ESP-IDF *memory types* doc — flash via
  I/D-Cache in SRAM0; IRAM avoids cache misses (ESP32-C3).
- μNPU benchmark set (arXiv 2503.22567) — per-device MCU/NPU latency
  context for FlashAttention-scale KV ops (USB transit).
