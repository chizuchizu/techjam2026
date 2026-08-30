# Case 11 — WiFi data parallelism

Case 11 has 64 independent inputs. Complete ESP32-C3 forward replicas divide
the batch evenly and exchange no intermediate tensors. The host uses
persistent WiFi TCP to dispatch inputs and collect outputs.

## Evidence

| Stage | Result |
|---|---|
| Default FAST + EXACT host gate | 50/50 seed-runs PASS |
| Tiled FAST host gate | 25/25 PASS; worst `max_abs=1.1135e-3` |
| Two-worker physical smoke | Both seed-0 forwards PASS at 6.452 / 6.451 s |
| Two-worker complete batch | **206.354 s compute, 2.00x**, 64/64 PASS; worst `max_abs=1.3083e-3` |
| Four-worker complete batch | **103.169 s compute, 4.00x**, 64/64 PASS; worst `max_abs=1.3083e-3` |
| Eight-worker complete batch | **51.604 s compute, 8.00x**, 64/64 PASS; worst `max_abs=1.3083e-3` |

## Measurement

| Figure | Two boards | Four boards | Eight boards |
|---|---:|---:|---:|
| Inputs per board | `[32, 32]` | `[16, 16, 16, 16]` | `[8, 8, 8, 8, 8, 8, 8, 8]` |
| Per-board compute range | 206.353–206.354 s | 103.164–103.169 s | 51.594–51.604 s |
| Cluster compute wall | **206.354 s** | **103.169 s** | **51.604 s** |
| Equivalent one tiled worker | 412.707 s | 412.668 s | 412.794 s |
| Representative forward | 6.4477 s | 6.4479 s | 6.4499 s |
| Compute scaling vs one tiled worker | **2.00x** | **4.00x** | **8.00x** |
| Relative to best optimized single board | **0.67x** (1.49x slower) | **1.34x faster** | **2.69x faster** |
| End-to-end wall including WiFi TCP | **238.0 s** | **119.3 s** | **60.8 s** |
| Completed outputs | 64/64 | 64/64 | 64/64 |
| Missing inputs / failing elements | 0 / 0 | 0 / 0 | 0 / 0 |

The 16-row schedule reduces static RAM from 256,180 B to 158,964 B so the
complete forward and WiFi/lwIP can coexist. It also increases one forward from
about 2.166 s to 6.448 s. H=16 requires sixteen independent causal softmaxes
per layer, amplifying the tiled path's loop and staging overhead. Replica
scaling is exact. Four nodes are the measured crossover: 103.169 s is 1.34x
faster than the best optimized single-board compute result of 138.610 s.
Eight nodes reduce compute to 51.604 s, a 2.69x improvement over that result.

Raw results: [`results_case11_two_c3_wifi.json`](results_case11_two_c3_wifi.json),
[`results_case11_four_c3_wifi.json`](results_case11_four_c3_wifi.json), and
[`results_case11_eight_c3_wifi.json`](results_case11_eight_c3_wifi.json).

## Reproduce

Generate the official batch inputs and references:

```bash
.venv/bin/python benchmarks/batch-dp/tools/export_batch.py \
  --batch 64 --S 128 --D 128 --H 16 --F 128 --L 4 \
  --outdir benchmarks/case-11/multiboard
```

Copy `secrets.example.h` to the ignored `secrets.h`, fill in the benchmark-LAN
credentials, and flash the opt-in image to each worker:

```bash
cd benchmarks/case-11/optimisation/esp32-baseline
pio run -e esp32-wifi-tiled -t upload --upload-port /dev/ttyACM0
pio run -e esp32-wifi-tiled -t upload --upload-port /dev/ttyACM1
# Repeat for each additional worker before a multi-node run.
```

Run the full gate, replacing the example addresses with the workers:

```bash
.venv/bin/python benchmarks/batch-dp/tools/run_batch_dp.py \
  --batch 64 --seq-len 128 --d-model 128 \
  --root benchmarks/case-11/multiboard \
  --wifi 192.168.1.41 192.168.1.42 192.168.1.43 192.168.1.44 \
         192.168.1.45 192.168.1.46 192.168.1.47 192.168.1.48 \
  --json benchmarks/case-11/multiboard/results_case11_eight_c3_wifi.json
```

The runner checks every worker's model shape before dispatch and never reports
a speedup for an incomplete batch.
