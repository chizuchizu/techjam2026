# Case 12 — WiFi data parallelism

Case 12 has 64 independent inputs with a short sequence (`S=32`). Complete
ESP32-C3 replicas divide the batch evenly and exchange no intermediate tensors.
The host uses persistent WiFi TCP to dispatch inputs and collect outputs.

## Evidence ladder

| Stage | Result |
|---|---|
| Official-shape host gate | FAST 25/25 PASS and EXACT 25/25 PASS; FAST worst `max_abs=1.0657e-3` |
| Fresh optimized one-board batch | 33.928 s compute, 64/64 PASS; worst `max_abs=1.2816e-3` |
| Two-board WiFi batch | **17.091 s compute, 2.00x**, 64/64 PASS; worst `max_abs=1.2816e-3` |
| Four-board WiFi batch | **8.554 s compute, 4.00x**, 64/64 PASS; worst `max_abs=1.2816e-3` |
| Eight-board WiFi batch | **4.282 s compute, 8.00x**, 64/64 PASS; worst `max_abs=1.2816e-3` |

The fresh one-board USB run had one successfully recovered short output frame.
All 64 inputs completed and passed, so its device-compute total is valid; the
retry inflated its 206.0 s transport-inclusive wall, which is not used for a
transport comparison. The earlier clean optimized capture remains 33.879 s
compute / 73.744 s USB-inclusive wall.

## WiFi measurements

| Figure | Two boards | Four boards | Eight boards |
|---|---:|---:|---:|
| Inputs per board | `[32, 32]` | `[16, 16, 16, 16]` | `[8, 8, 8, 8, 8, 8, 8, 8]` |
| Per-board compute | `[17.082, 17.091]` s | `[8.554, 8.554, 8.552, 8.549]` s | `[4.279, 4.282, 4.277, 4.280, 4.279, 4.280, 4.279, 4.279]` s |
| Cluster compute wall | **17.091 s** | **8.554 s** | **4.282 s** |
| Sum of all device compute | 34.172 s | 34.209 s | 34.234 s |
| Median forward | 0.5335 s | 0.5342 s | 0.5344 s |
| Compute scaling vs one WiFi worker | **1.999x** | **3.999x** | **7.996x** |
| End-to-end wall including WiFi TCP | 29.4 s | 15.7 s | 7.3 s |
| Completed outputs | 64/64 | 64/64 | 64/64 |
| Missing inputs / failing elements | 0 / 0 | 0 / 0 | 0 / 0 |

The direct-WiFi worker links at 104,956 / 327,680 bytes static RAM, so the
short sequence needs no tiling. Flash is the binding resource: the two weight
blobs plus WiFi image use 3,125,954 / 3,145,728 bytes (**99.4%**), leaving
19,774 bytes. This is valid for the measured image but leaves little room for
additional features.

The WiFi worker's 0.5335–0.5344 s median is within 1% of the fresh optimized
USB worker's 0.5301 s. Against the best earlier optimized single-board total
of 33.879 s, the measured two-, four-, and eight-node clusters are about
**1.98x**, **3.96x**, and **7.91x** faster respectively.

Raw results:

- [`results_case12_one_c3_optimized_usb.json`](results_case12_one_c3_optimized_usb.json)
- [`results_case12_two_c3_wifi.json`](results_case12_two_c3_wifi.json)
- [`results_case12_four_c3_wifi.json`](results_case12_four_c3_wifi.json)
- [`results_case12_eight_c3_wifi.json`](results_case12_eight_c3_wifi.json)

## Reproduce

```bash
.venv/bin/python benchmarks/batch-dp/tools/export_batch.py \
  --batch 64 --S 32 --D 128 --H 4 --F 128 --L 4 \
  --outdir benchmarks/case-12/multiboard

cd benchmarks/case-12/optimisation/esp32-baseline
cp secrets.example.h secrets.h
pio run -e esp32-wifi -t upload --upload-port /dev/ttyACM0
pio run -e esp32-wifi -t upload --upload-port /dev/ttyACM1
pio run -e esp32-wifi -t upload --upload-port /dev/ttyACM2
pio run -e esp32-wifi -t upload --upload-port /dev/ttyACM3
pio run -e esp32-wifi -t upload --upload-port /dev/ttyACM4
pio run -e esp32-wifi -t upload --upload-port /dev/ttyACM5
pio run -e esp32-wifi -t upload --upload-port /dev/ttyACM6
pio run -e esp32-wifi -t upload --upload-port /dev/ttyACM7
cd ../../../..

.venv/bin/python benchmarks/batch-dp/tools/run_batch_dp.py \
  --batch 64 --seq-len 32 --d-model 128 \
  --root benchmarks/case-12/multiboard \
  --wifi 192.168.1.41 192.168.1.42 192.168.1.43 192.168.1.44 \
         192.168.1.45 192.168.1.46 192.168.1.47 192.168.1.48
```

Replace the example addresses with the workers. The runner verifies the
reported model shape and never reports a speedup for an incomplete batch.
