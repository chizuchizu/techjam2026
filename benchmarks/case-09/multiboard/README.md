# Case 09 — WiFi data parallelism

Case 09 has 64 independent inputs. Complete ESP32-C3 forward replicas divide
the batch evenly and exchange no intermediate tensors. The host uses
persistent WiFi TCP to dispatch inputs and collect outputs.

## Evidence

| Stage | Result |
|---|---|
| Default FAST + EXACT host gate | 50/50 seed-runs PASS |
| Tiled FAST host gate | 25/25 PASS; worst `max_abs=1.1038e-3` |
| Two-worker physical smoke | Both seed-0 forwards PASS at 3.5633 / 3.5638 s |
| Four-worker complete batch | **57.005 s compute, 4.00x**, 64/64 PASS; worst `max_abs=1.2649e-3` |

## Four-node measurement

| Figure | Result |
|---|---:|
| Inputs per board | `[16, 16, 16, 16]` |
| Per-board compute | `[57.004, 57.005, 56.997, 57.002]` s |
| Cluster compute wall | **57.005 s** |
| Equivalent one tiled worker | 228.007 s |
| Median forward | 3.562 s |
| Compute scaling vs one tiled worker | **4.00x** |
| Compute gain vs best optimized single board | **2.42x** |
| End-to-end wall including WiFi TCP | 75.4 s |
| Completed outputs | 64/64 |
| Missing inputs / failing elements | 0 / 0 |

The distinction between the two speedups matters. The 16-row schedule reduces
static RAM from 273,180 B to 224,244 B so WiFi/lwIP can run, but it increases
one forward from about 2.157 s to 3.562 s. Four replicas scale that tiled
forward almost perfectly, while the memory-saving schedule reduces the gain
against the fastest non-WiFi firmware.

Raw result:

- [`results_case9_four_c3_wifi.json`](results_case9_four_c3_wifi.json)

## Reproduce

Generate the official batch inputs and references:

```bash
.venv/bin/python benchmarks/batch-dp/tools/export_batch.py \
  --batch 64 --S 128 --D 128 --H 1 --F 128 --L 4 \
  --outdir benchmarks/case-09/multiboard
```

Copy `secrets.example.h` to the ignored `secrets.h`, fill in the benchmark-LAN
credentials, and flash the opt-in image to each worker:

```bash
cd benchmarks/case-09/optimisation/esp32-baseline
pio run -e esp32-wifi-tiled -t upload --upload-port /dev/ttyACM0
pio run -e esp32-wifi-tiled -t upload --upload-port /dev/ttyACM1
pio run -e esp32-wifi-tiled -t upload --upload-port /dev/ttyACM2
pio run -e esp32-wifi-tiled -t upload --upload-port /dev/ttyACM3
```

Run the full gate, replacing the example addresses with the workers:

```bash
.venv/bin/python benchmarks/batch-dp/tools/run_batch_dp.py \
  --batch 64 --seq-len 128 --d-model 128 \
  --root benchmarks/case-09/multiboard \
  --wifi 192.168.1.41 192.168.1.42 192.168.1.43 192.168.1.44 \
  --json benchmarks/case-09/multiboard/results_case9_four_c3_wifi.json
```

The runner checks every worker's model shape before dispatch and never reports
a speedup for an incomplete batch.
