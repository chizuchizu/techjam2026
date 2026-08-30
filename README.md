# TechJam 2026 — ESP32 Transformer

This repository explores a different ESP32 execution strategy for each of the
14 official Transformer benchmark cases. Every case owns its documentation,
baseline evidence, single-board optimisations, multiboard implementation, and
results under [`benchmarks/case-NN/`](benchmarks/).

## Case layout

[`benchmarks/README.md`](benchmarks/README.md) is the full case index. Each
implemented case follows the same layout; case-02 is shown as the example:

```text
benchmarks/case-02/
├── README.md
├── baseline/
│   └── results/
├── optimisation/
│   └── esp32-baseline/
└── multiboard/
    ├── esp32_cluster_transport/
    ├── esp32-linkbench/
    ├── tools/
    └── results/
```

The competition problem statement and official executable reference remain at
the repository root: [`COMPETITION_RULES.MD`](COMPETITION_RULES.MD) and
[`torch_transformer_benchmark.py`](torch_transformer_benchmark.py). Smaller
experiments that are not official cases are isolated in
[`benchmarks/experiments/`](benchmarks/experiments/).

## Official case status

| Case | Shape `(B,S,D,H,F,L)` | Baseline, 1 board | Optimised, 1 board | 2 boards | 4-node WiFi DP | 8-node WiFi DP | vs baseline | Status |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| [1](benchmarks/case-01/) | `(64,128,128,4,128,4)` | 2,697.6 s * | 127.36 s | **63.7 s** | **67.465 s** | **33.713 s** | **80.0x** | Eight-node WiFi DP verified, 64/64 PASS |
| [2](benchmarks/case-02/) | `(1,128,128,4,128,4)` | 42.15 s | 1.990 s | **1.276 s** | **4.2137 s †** | **4.220 s †** | **33.0x** | WiFi full-forward verified; B=1 activates one DP node |
| [3](benchmarks/case-03/) | `(4,128,128,4,128,4)` | 168.6 s * | 7.96 s | **4.0 s** | **4.215 s** | **4.218 s ‡** | **42.1x** | Eight available, four active; 4/4 PASS |
| [4](benchmarks/case-04/) | `(16,128,128,4,128,4)` | 674.4 s * | 31.84 s | **15.9 s** | **16.853 s** | **8.438 s** | **79.9x** | Eight-node WiFi DP verified, 16/16 PASS |
| [5](benchmarks/case-05/) | `(128,128,128,4,128,4)` | 5,395.2 s * | 254.72 s | **127.4 s** | **134.887 s** | **67.451 s** | **80.0x** | Eight-node WiFi DP verified, 128/128 PASS |
| [6](benchmarks/case-06/) | `(10000,128,128,4,128,4)` | - | - | - | - | - | - | Not implemented - streaming batch execution |
| [7](benchmarks/case-07/) | `(64,128,32,4,32,4)` | - | **30.427 s** | **15.822 s** | - | - | - | Two-node WiFi DP verified, 64/64 PASS |
| [8](benchmarks/case-08/) | `(64,128,1024,4,1024,4)` | - | - | - | - | - | - | Not implemented - weight and feature sharding |
| [9](benchmarks/case-09/) | `(64,128,128,1,128,4)` | - | - | - | - | - | - | Not implemented - sequence/model sharding |
| [10](benchmarks/case-10/) | `(64,128,128,2,128,4)` | - | - | - | - | - | - | Not implemented - two head shards plus batch parallelism |
| [11](benchmarks/case-11/) | `(64,128,128,16,128,4)` | - | - | - | - | - | - | Not implemented - fine-grained head parallelism |
| [12](benchmarks/case-12/) | `(64,32,128,4,128,4)` | - | **33.879 s** | **17.091 s** | - | - | - | Two-node WiFi DP verified, 64/64 PASS |
| [13](benchmarks/case-13/) | `(64,1024,128,4,128,4)` | - | - | - | - | - | - | Not implemented - online attention and KV sharding |
| [14](benchmarks/case-14/) | `(32,100000,1024,16,1024,2)` | - | - | - | - | - | - | Not implemented - extreme sequence streaming |

All times cover the **whole case**: one forward for case 2, the full batch of B
inputs for the others. Every figure is the device's own measurement of the
complete four-layer body, with host serial transfer excluded from all timing
columns alike.

`*` Cases 1, 3, 4 and 5 were never run on the pre-optimisation firmware, so
their baseline is **estimated** as `B x 42.15 s` from case 2's measured
starting point. Only case 2's baseline is a measurement. The optimised,
two-board, four-node WiFi, and eight-node WiFi columns are measured for cases
1–5.

**4-node WiFi DP** means four full-forward replicas receiving independent
batch inputs over persistent TCP. The column reports compute wall, excluding
transport like the other timing columns; measured WiFi-inclusive wall times
are in [`benchmarks/batch-dp/RESULTS_FOUR_C3_WIFI.md`](benchmarks/batch-dp/RESULTS_FOUR_C3_WIFI.md).
`†` Case 2 has B=1, so only one of the four available data-parallel nodes can
run; 4.2137 s is the tiled WiFi single-forward time, not a four-board speedup.
The same limit applies with eight available nodes. `‡` Case 3 has B=4, so it
uses four active nodes and cannot obtain an eight-board data-parallel speedup.
Complete eight-node measurements and WiFi-inclusive wall times are in
[`benchmarks/batch-dp/RESULTS_EIGHT_C3_WIFI.md`](benchmarks/batch-dp/RESULTS_EIGHT_C3_WIFI.md).

Cases 1, 3, 4 and 5 are batches of independent forwards, so two boards run B/2
inputs each and exchange nothing - exactly 2.00x. Case 2 is a single forward
and has to be split *inside* the model, by token row, which costs one K/V
exchange per layer and does not halve the per-board weight streaming - hence
1.56x.

The approach notes for unimplemented cases are design hypotheses, not measured
claims. Each case README records what must be validated before its status can
change.

## Hackathon submission TODO

### Benchmark evidence

#### Completed foundation

- [x] Add the official case-2 baseline implementation.
- [x] Validate the case-2 baseline against the required accuracy gate.
- [x] Record the physical single-board baseline timing.
- [x] Add the optimised case-2 single-board implementation.
- [x] Validate the optimised implementation on host and physical hardware.
- [x] Record the baseline-to-optimised single-board speedup.
- [x] Record a two-board partial-layer result.
- [x] Record a four-board attention-only result.
- [x] Label both multiboard results as partial, not end-to-end inference.

#### Complete two-board end-to-end case 2

- [x] Run all four layers and the final LayerNorm across two boards.
- [x] Add the missing projections, residuals, LayerNorm, and FFN path.
- [x] Keep weights on the workers; each board holds the full blobs in flash and
      returns only its own output rows.
- [x] Validate the two-board output with five seeds.
- [x] Measure full wall time and split compute from communication.
- [x] Compare it with one board and save the raw results.

Result: **1.276 s across two C3s against 1.990 s on one (1.56x)** on the opt23
kernels, 25/25 host seeds passing the accuracy gate with zero failing elements.
Time blocked on the board-to-board link is 5-88 ms per forward. The measured
window excludes host serial transfer, exactly as the single-board number does.
See
[`benchmarks/case-02/multiboard/results/CASE2_FULL_E2E_RESULTS.md`](benchmarks/case-02/multiboard/results/CASE2_FULL_E2E_RESULTS.md).

#### Multiboard for the batch cases

- [x] Data-parallel dispatch for cases 1, 3, 4 and 5 across N boards.
- [x] Validate every output of every batch against the torch reference.
- [x] Compare with one board and save the raw results.

Result: **2.00x on two boards, 4.00x on four WiFi nodes, and 8.00x on eight
WiFi nodes when B >= 8**. Case 3 saturates at 4.00x because B=4. The
eight-board cases 1–5 sweep gated all 213 forwards individually with zero
missing inputs and zero failing elements. These cases are independent forwards
over shared weights, so the boards exchange nothing. See
[`benchmarks/batch-dp/`](benchmarks/batch-dp/).

The same shape-aware data-parallel coordinator is physically verified for
case 7 (`D=F=32`) and case 12 (`S=32`). Two direct-WiFi replicas completed all
64/64 forwards in 15.822 s and 17.091 s compute wall respectively, with 2.00x
scaling versus one WiFi worker. See
[`benchmarks/case-07/multiboard/`](benchmarks/case-07/multiboard/) and
[`benchmarks/case-12/multiboard/`](benchmarks/case-12/multiboard/).

#### Benchmark four boards, then scale to eight

- [x] Run and validate the same end-to-end path on four boards.
- [x] Save the four-board speedup, efficiency, median, and raw results.
- [ ] Record per-forward samples and p90 for the four-board run.
- [x] Choose an eight-board split beyond the four available attention heads.
- [ ] Add stable board IDs, discovery, timeouts, retries, and failure handling.
- [x] Validate the eight-board output against the official reference.
- [ ] Benchmark one, two, four, and eight boards under the same conditions.
- [ ] Measure speedup, efficiency, communication, retries, and slowest-worker time.
- [x] Save raw results and explain where scaling improves or stops.

#### Support additional official benchmark cases

- [ ] Move case shapes out of case-2-specific code and into configuration.
- [ ] Add case selection, memory checks, and shared validated kernels.
- [ ] Support cases 3, 7, 9, 11, 12, and 13 first.
- [ ] Validate every case against its official reference.
- [ ] Keep each case's code, raw results, and report in its own directory.
- [ ] Add other cases after checking memory and runtime needs.

#### Final evidence pack

- [ ] Run the final case-2 commands from a fresh checkout.
- [ ] Record hardware, clocks, software versions, and timing rules.
- [ ] Check that every result passes accuracy and uses the same scope.
- [ ] Publish raw captures with median, p90, warm-ups, and run counts.
- [ ] Summarise what scaled, what did not, and why.

### Repository and README

- [x] Organise official work into one directory per benchmark case.
- [x] Keep case-2 baseline, optimisation, multiboard code, and results together.
- [x] Keep the competition problem statement at the repository root.
- [x] Document setup and reproduction commands.
- [ ] Add a short project overview for non-technical judges.
- [ ] Add a limitations and future-improvements section.
- [ ] Add a team-contributions section with one line per person.
- [ ] Document the AI tools, libraries, frameworks, and development tools used.
- [ ] Explain that the official benchmark uses seeded random weights and no dataset.
- [ ] Check the public repository for credentials, private addresses, and build files.
- [ ] Merge the final pull request and verify all README links on GitHub.

### Demo video

- [ ] Write a short problem → approach → result demo script.
- [ ] Record the ESP32 setup and identify the boards on camera.
- [ ] Record one reproducible inference or benchmark run.
- [ ] Show the accuracy result before showing the speedup.
- [ ] Explain the boundary between the full single-board result and partial multiboard results.
- [ ] Add captions or readable terminal zoom for timings and validation output.
- [ ] Upload the video publicly to YouTube.
- [ ] Add the public video link to the Devpost submission.

### Devpost submission

- [ ] Write the project description and problem statement.
- [ ] Describe the single-board and multiboard approaches.
- [ ] List development tools, APIs, libraries, frameworks, and assets.
- [ ] Add the GitHub repository link.
- [ ] Add the demo video link.
- [ ] Add limitations, future work, and practical impact.
- [ ] Add all team members and their contributions.
- [ ] Preview the complete submission while logged out.
- [ ] Submit before the deadline and save the confirmation.

### Optional technical stretch goals

- [ ] Complete output projection, residuals, second LayerNorm, and FFN for one distributed layer.
- [ ] Run all four case-2 layers across multiple boards.
- [ ] Add worker heartbeats, retry handling, and cached performance profiles.
- [ ] Implement and validate another official benchmark case in its own directory.

## Quick setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
make check
```

Build the current case-2 single-board implementation:

The exporter also requires PyTorch in the selected Python environment.

```bash
cd benchmarks/case-02/optimisation/esp32-baseline
python3 tools/export_case2.py --outdir . --seeds 25
pio run -e esp32-baseline
```

Configure a local, Git-ignored Wi-Fi secrets file before compiling the case-2
cluster worker:

```bash
cp benchmarks/case-02/multiboard/esp32_cluster_transport/secrets.example.h \
   benchmarks/case-02/multiboard/esp32_cluster_transport/secrets.h
arduino-cli compile --fqbn esp32:esp32:XIAO_ESP32C3 \
  benchmarks/case-02/multiboard/esp32_cluster_transport
```

Never commit credentials, private addresses, generated model artifacts, or
build directories.

## Project documentation

- [`TODO.md`](TODO.md) — shared priorities.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — validation and result conventions.
- [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) — milestones and acceptance gates.
- [`docs/MULTI_ESP32_DESIGN.md`](docs/MULTI_ESP32_DESIGN.md) — cluster decomposition and protocol design.
- [`docs/WIFI_ON_A_COMPUTE_NODE.md`](docs/WIFI_ON_A_COMPUTE_NODE.md) — SRAM challenge, tiled TCP solution, physical results, and sidecar alternative.
- [`esp32-linkbench/docs/PC_MASTER_WIFI_BRIDGE.md`](esp32-linkbench/docs/PC_MASTER_WIFI_BRIDGE.md) — what the ESP-NOW relay does and the full-protocol WiFi–UART sidecar.
- [`docs/PRIOR_ART.md`](docs/PRIOR_ART.md) — prior-art review and positioning.
