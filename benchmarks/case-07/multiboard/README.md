# Case 07 — WiFi data parallelism

Case 07 has 64 independent inputs. Complete ESP32-C3 forward replicas divide
the batch evenly and exchange no intermediate tensors. The host uses persistent
WiFi TCP only to dispatch each input and collect its output.

## Evidence ladder

The multiboard result was recorded only after the earlier stages passed:

| Stage | Result |
|---|---|
| Official-shape host gate | FAST 25/25 PASS and EXACT 25/25 PASS; FAST worst `max_abs=1.4839e-3` |
| Fresh optimized one-board batch | 31.006 s compute, 64/64 PASS; worst `max_abs=1.4986e-3` |
| Two-board WiFi batch | **15.822 s compute, 2.00x**, 64/64 PASS; worst `max_abs=1.4986e-3` |
| Four-board WiFi batch | **7.922 s compute, 4.00x**, 64/64 PASS; worst `max_abs=1.4986e-3` |

The fresh one-board USB run had one successfully recovered short output frame.
All 64 assigned inputs still completed and passed, so its compute total is
valid; the retry inflated its 203.7 s transport-inclusive wall and that wall
figure is not used as a transport comparison. The earlier clean one-board
capture remains 30.427 s compute / 70.227 s USB-inclusive wall.

## WiFi measurements

| Figure | Two boards | Four boards |
|---|---:|---:|
| Inputs per board | `[32, 32]` | `[16, 16, 16, 16]` |
| Per-board compute | `[15.822, 15.818]` s | `[7.917, 7.922, 7.918, 7.921]` s |
| Cluster compute wall | **15.822 s** | **7.922 s** |
| Sum of all device compute | 31.639 s | 31.679 s |
| Median forward | 0.4940 s | 0.4948 s |
| Compute scaling vs one WiFi worker | **2.00x** | **3.999x** |
| End-to-end wall including WiFi TCP | 28.6 s | 17.9 s |
| Completed outputs | 64/64 | 64/64 |
| Missing inputs / failing elements | 0 / 0 | 0 / 0 |

The WiFi worker links at 108,300 / 327,680 bytes static RAM. Case 07's small
`D=F=32` workspace needs no sequence tiling; the WiFi image instead reads the
small weight blobs from flash rather than keeping the single-board firmware's
SRAM weight cache. That makes one WiFi forward about 4% slower than the
published 0.475 s optimized forward. Against the best measured optimized
single-board total, the two- and four-node clusters are about **1.92x** and
**3.84x** faster respectively.

Raw results:

- [`results_case7_one_c3_optimized_usb.json`](results_case7_one_c3_optimized_usb.json)
- [`results_case7_two_c3_wifi.json`](results_case7_two_c3_wifi.json)
- [`results_case7_four_c3_wifi.json`](results_case7_four_c3_wifi.json)

## Reproduce

Generate official batch inputs and references:

```bash
.venv/bin/python benchmarks/batch-dp/tools/export_batch.py \
  --batch 64 --S 128 --D 32 --H 4 --F 32 --L 4 \
  --outdir benchmarks/case-07/multiboard
```

Copy `secrets.example.h` to the ignored `secrets.h`, build, and flash the
opt-in image to each worker:

```bash
cd benchmarks/case-07/optimisation/esp32-baseline
pio run -e esp32-wifi -t upload --upload-port /dev/ttyACM0
pio run -e esp32-wifi -t upload --upload-port /dev/ttyACM1
pio run -e esp32-wifi -t upload --upload-port /dev/ttyACM2
pio run -e esp32-wifi -t upload --upload-port /dev/ttyACM3
```

Run the full gate, replacing the example addresses with the workers:

```bash
.venv/bin/python benchmarks/batch-dp/tools/run_batch_dp.py \
  --batch 64 --seq-len 128 --d-model 32 \
  --root benchmarks/case-07/multiboard \
  --wifi 192.168.1.41 192.168.1.42 192.168.1.43 192.168.1.44
```

The runner verifies the reported model shape before sending inputs and never
reports a speedup for an incomplete batch.
