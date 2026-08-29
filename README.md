# TechJam 2026 — ESP32 Transformer

This repository explores a different ESP32 execution strategy for each of the
14 official Transformer benchmark cases. Every case owns its documentation,
baseline evidence, single-board optimisations, multiboard implementation, and
results under [`benchmarks/case-NN/`](benchmarks/).

## Case layout

[`benchmarks/README.md`](benchmarks/README.md) is the full case index. The
current implementation is concentrated in
[`case-02/`](benchmarks/case-02/):

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

| Case | Shape `(B,S,D,H,F,L)` | Status | Case notes |
|---:|---|---|---|
| [1](benchmarks/case-01/) | `(64,128,128,4,128,4)` | Not implemented | Batch-parallel candidate |
| [2](benchmarks/case-02/) | `(1,128,128,4,128,4)` | **Single-board verified** | Full body at 5.27 s; partial multiboard paths verified |
| [3](benchmarks/case-03/) | `(4,128,128,4,128,4)` | Not implemented | Small-batch scheduling |
| [4](benchmarks/case-04/) | `(16,128,128,4,128,4)` | Not implemented | Batch tiling and dispatch |
| [5](benchmarks/case-05/) | `(128,128,128,4,128,4)` | Not implemented | Throughput-oriented batch sharding |
| [6](benchmarks/case-06/) | `(10000,128,128,4,128,4)` | Not implemented | Streaming batch execution |
| [7](benchmarks/case-07/) | `(64,128,32,4,32,4)` | Not implemented | Narrow-kernel overhead and fusion |
| [8](benchmarks/case-08/) | `(64,128,1024,4,1024,4)` | Not implemented | Weight and feature sharding |
| [9](benchmarks/case-09/) | `(64,128,128,1,128,4)` | Not implemented | Sequence/model sharding; no head parallelism |
| [10](benchmarks/case-10/) | `(64,128,128,2,128,4)` | Not implemented | Two head shards plus batch parallelism |
| [11](benchmarks/case-11/) | `(64,128,128,16,128,4)` | Not implemented | Fine-grained head parallelism |
| [12](benchmarks/case-12/) | `(64,32,128,4,128,4)` | Not implemented | Short-sequence launch overhead |
| [13](benchmarks/case-13/) | `(64,1024,128,4,128,4)` | Not implemented | Online attention and KV sharding |
| [14](benchmarks/case-14/) | `(32,100000,1024,16,1024,2)` | Not implemented | Extreme sequence streaming |

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

- [ ] Run all four layers and the final LayerNorm across two boards.
- [ ] Add the missing projections, residuals, LayerNorm, and FFN path.
- [ ] Keep weights on the workers and return each complete layer output.
- [ ] Validate the two-board output with five seeds.
- [ ] Measure full wall time and split compute from communication.
- [ ] Compare it with one board and save the raw results.

#### Benchmark four boards, then scale to eight

- [ ] Run and validate the same end-to-end path on four boards.
- [ ] Save the four-board speedup, efficiency, median, and p90.
- [ ] Choose an eight-board split beyond the four available attention heads.
- [ ] Add stable board IDs, discovery, timeouts, retries, and failure handling.
- [ ] Validate the eight-board output against the official reference.
- [ ] Benchmark one, two, four, and eight boards under the same conditions.
- [ ] Measure speedup, efficiency, communication, retries, and slowest-worker time.
- [ ] Save raw results and explain where scaling improves or stops.

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
- [`docs/PRIOR_ART.md`](docs/PRIOR_ART.md) — prior-art review and positioning.
