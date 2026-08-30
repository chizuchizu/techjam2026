# Case 10 — WiFi data parallelism

Case 10 has 64 independent inputs. Complete ESP32-C3 forward replicas divide
the batch evenly and exchange no intermediate tensors. The host uses
persistent WiFi TCP to dispatch inputs and collect outputs.

## Evidence

| Stage | Result |
|---|---|
| Default FAST + EXACT host gate | 50/50 seed-runs PASS |
| Tiled FAST host gate | 25/25 PASS; worst `max_abs=1.1496e-3` |
| Two-worker physical smoke | Both seed-0 forwards PASS at 3.7218 / 3.7193 s |
| Two-worker complete batch | **119.101 s compute, 2.00x**, 64/64 PASS; worst `max_abs=1.2526e-3` |
| Four-worker complete batch | **59.563 s compute, 4.00x**, 64/64 PASS; worst `max_abs=1.2526e-3` |
| Eight-worker complete batch | **29.793 s compute, 7.999x**, 64/64 PASS; worst `max_abs=1.2526e-3` |

## Measurement

| Figure | Two boards | Four boards | Eight boards |
|---|---:|---:|---:|
| Inputs per board | `[32, 32]` | `[16, 16, 16, 16]` | `[8, 8, 8, 8, 8, 8, 8, 8]` |
| Per-board compute | `[119.094, 119.101]` s | `[59.560, 59.563, 59.555, 59.553]` s | `[29.787, 29.789, 29.793, 29.786, 29.782, 29.786, 29.786, 29.791]` s |
| Cluster compute wall | **119.101 s** | **59.563 s** | **29.793 s** |
| Equivalent one tiled worker | 238.195 s | 238.232 s | 238.299 s |
| Representative forward | 3.7216 s | 3.7215 s | 3.7227 s |
| Compute scaling vs one tiled worker | **2.00x** | **4.00x** | **7.999x** |
| Compute gain vs best optimized single board | **1.16x** | **2.33x** | **4.65x** |
| End-to-end wall including WiFi TCP | **146.7 s** | **74.2 s** | **39.2 s** |
| Completed outputs | 64/64 | 64/64 | 64/64 |
| Missing inputs / failing elements | 0 / 0 | 0 / 0 | 0 / 0 |

The distinction between the two speedups matters. The 16-row schedule reduces
static RAM from 265,324 B to 189,428 B so WiFi/lwIP can run, but it increases
one forward from about 2.165 s to 3.722 s. Two, four, and eight replicas scale
the tiled path almost perfectly, while the memory-saving schedule limits the
gain against the fastest non-WiFi firmware.

Raw results:

- [`results_case10_two_c3_wifi.json`](results_case10_two_c3_wifi.json)
- [`results_case10_four_c3_wifi.json`](results_case10_four_c3_wifi.json)
- [`results_case10_eight_c3_wifi.json`](results_case10_eight_c3_wifi.json)

## Reproduce

Generate the official batch inputs and references:

```bash
.venv/bin/python benchmarks/batch-dp/tools/export_batch.py \
  --batch 64 --S 128 --D 128 --H 2 --F 128 --L 4 \
  --outdir benchmarks/case-10/multiboard
```

Copy `secrets.example.h` to the ignored `secrets.h`, fill in the benchmark-LAN
credentials, and flash the opt-in image to each worker:

```bash
cd benchmarks/case-10/optimisation/esp32-baseline
pio run -e esp32-wifi-tiled -t upload --upload-port /dev/ttyACM0
pio run -e esp32-wifi-tiled -t upload --upload-port /dev/ttyACM1
pio run -e esp32-wifi-tiled -t upload --upload-port /dev/ttyACM2
pio run -e esp32-wifi-tiled -t upload --upload-port /dev/ttyACM3
pio run -e esp32-wifi-tiled -t upload --upload-port /dev/ttyACM4
pio run -e esp32-wifi-tiled -t upload --upload-port /dev/ttyACM5
pio run -e esp32-wifi-tiled -t upload --upload-port /dev/ttyACM6
pio run -e esp32-wifi-tiled -t upload --upload-port /dev/ttyACM7
```

Run the full gate, replacing the example addresses with the workers:

```bash
.venv/bin/python benchmarks/batch-dp/tools/run_batch_dp.py \
  --batch 64 --seq-len 128 --d-model 128 \
  --root benchmarks/case-10/multiboard \
  --wifi 192.168.1.41 192.168.1.42 192.168.1.43 192.168.1.44 \
         192.168.1.45 192.168.1.46 192.168.1.47 192.168.1.48 \
  --json benchmarks/case-10/multiboard/results_case10_eight_c3_wifi.json
```

The runner checks every worker's model shape before dispatch and never reports
a speedup for an incomplete batch.
