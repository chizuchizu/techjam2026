# Case 12 — two-node WiFi data parallelism

Case 12 has 64 independent inputs with a short sequence (`S=32`). Two complete
ESP32-C3 replicas take 32 inputs each and exchange no intermediate tensors.
The host uses persistent WiFi TCP to dispatch inputs and collect outputs.

## Evidence ladder

| Stage | Result |
|---|---|
| Official-shape host gate | FAST 25/25 PASS and EXACT 25/25 PASS; FAST worst `max_abs=1.0657e-3` |
| Fresh optimized one-board batch | 33.928 s compute, 64/64 PASS; worst `max_abs=1.2816e-3` |
| Two-board WiFi batch | **17.091 s compute, 2.00x**, 64/64 PASS; worst `max_abs=1.2816e-3` |

The fresh one-board USB run had one successfully recovered short output frame.
All 64 inputs completed and passed, so its device-compute total is valid; the
retry inflated its 206.0 s transport-inclusive wall, which is not used for a
transport comparison. The earlier clean optimized capture remains 33.879 s
compute / 73.744 s USB-inclusive wall.

## Two-board measurement

| Figure | Value |
|---|---:|
| Inputs per board | `[32, 32]` |
| Per-board compute | `[17.082, 17.091]` s |
| Cluster compute wall | **17.091 s** |
| Sum of all device compute | 34.172 s |
| Median forward | 0.5335 s |
| Compute scaling vs one WiFi worker | **1.999x** |
| End-to-end wall including WiFi TCP | 29.4 s |
| Completed outputs | 64/64 |
| Missing inputs / failing elements | 0 / 0 |

The direct-WiFi worker links at 104,956 / 327,680 bytes static RAM, so the
short sequence needs no tiling. Flash is the binding resource: the two weight
blobs plus WiFi image use 3,125,954 / 3,145,728 bytes (**99.4%**), leaving
19,774 bytes. This is valid for the measured image but leaves little room for
additional features.

The WiFi worker's 0.5335 s median is within 1% of the fresh optimized USB
worker's 0.5301 s. Against the best earlier optimized single-board total of
33.879 s, the measured two-node cluster is about **1.98x faster**.

Raw results:

- [`results_case12_one_c3_optimized_usb.json`](results_case12_one_c3_optimized_usb.json)
- [`results_case12_two_c3_wifi.json`](results_case12_two_c3_wifi.json)

## Reproduce

```bash
.venv/bin/python benchmarks/batch-dp/tools/export_batch.py \
  --batch 64 --S 32 --D 128 --H 4 --F 128 --L 4 \
  --outdir benchmarks/case-12/multiboard

cd benchmarks/case-12/optimisation/esp32-baseline
cp secrets.example.h secrets.h
pio run -e esp32-wifi -t upload --upload-port /dev/ttyACM0
pio run -e esp32-wifi -t upload --upload-port /dev/ttyACM1
cd ../../../..

.venv/bin/python benchmarks/batch-dp/tools/run_batch_dp.py \
  --batch 64 --seq-len 32 --d-model 128 \
  --root benchmarks/case-12/multiboard \
  --wifi 192.168.1.41 192.168.1.42
```

Replace the example addresses with the two workers. The runner verifies the
reported model shape and never reports a speedup for an incomplete batch.
