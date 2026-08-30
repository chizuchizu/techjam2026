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

## Measurement

| Figure | Two boards |
|---|---:|
| Inputs per board | `[32, 32]` |
| Per-board compute | `[206.353, 206.354]` s |
| Cluster compute wall | **206.354 s** |
| Equivalent one tiled worker | 412.707 s |
| Representative forward | 6.4477 s |
| Compute scaling vs one tiled worker | **2.00x** |
| Relative to best optimized single board | **0.67x** (1.49x slower) |
| End-to-end wall including WiFi TCP | **238.0 s** |
| Completed outputs | 64/64 |
| Missing inputs / failing elements | 0 / 0 |

The 16-row schedule reduces static RAM from 256,180 B to 158,964 B so the
complete forward and WiFi/lwIP can coexist. It also increases one forward from
about 2.166 s to 6.448 s. H=16 requires sixteen independent causal softmaxes
per layer, amplifying the tiled path's loop and staging overhead. Replica
scaling is exact, but at least four nodes are needed to outperform the best
single-board compute result.

Raw result: [`results_case11_two_c3_wifi.json`](results_case11_two_c3_wifi.json).

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
```

Run the full gate, replacing the example addresses with the workers:

```bash
.venv/bin/python benchmarks/batch-dp/tools/run_batch_dp.py \
  --batch 64 --seq-len 128 --d-model 128 \
  --root benchmarks/case-11/multiboard \
  --wifi 192.168.1.41 192.168.1.42 \
  --json benchmarks/case-11/multiboard/results_case11_two_c3_wifi.json
```

The runner checks every worker's model shape before dispatch and never reports
a speedup for an incomplete batch.
