# Case-02 multiboard approaches

This directory keeps the entire distributed case-02 path together: link tests,
worker firmware, host coordinators, scheduling policies, raw captures, and
interpreted results.

## Contents

| Directory | Purpose |
|---|---|
| [`esp32_cluster_transport/`](esp32_cluster_transport/) | UDP/TCP attention and official layer-0 worker |
| [`esp32-linkbench/`](esp32-linkbench/) | Two-board ESP-NOW bandwidth benchmark |
| [`tools/`](tools/) | Discovery, transport, head-parallel, KV-sharded, and layer-0 coordinators |
| [`results/`](results/) | Raw captures and result reports |

## Verified results

| Scope | Boards | Wall time | Speedup | Accuracy |
|---|---:|---:|---:|---|
| Layer-0 LayerNorm + Q/K/V + causal attention | 1 C3 | 9.693 s | 1.00x | Pass |
| Layer-0 LayerNorm + Q/K/V + causal attention | 2 C3s | **4.850 s** | **2.00x** | Pass, 5/5 seeds |
| Four causal `128 x 32` attention heads | 1 C3 | 3.000 s average | 1.00x | Pass |
| Four causal `128 x 32` attention heads | 4 C3s | **0.766 s** | **3.92x** | Pass, zero failed elements |

These are partial case-02 paths, not complete distributed Transformer
forwards. See [`../README.md`](../README.md) for the exact scope exclusions.

Detailed reports:

- [`results/CASE2_NORM_HEAD_RESULTS.md`](results/CASE2_NORM_HEAD_RESULTS.md)
- [`results/FOUR_C3_PARALLEL_RESULTS.md`](results/FOUR_C3_PARALLEL_RESULTS.md)
- [`results/TWO_C3_PARALLEL_RESULTS.md`](results/TWO_C3_PARALLEL_RESULTS.md)
- [`results/LARGE_HEAD_PARALLEL_RESULTS.md`](results/LARGE_HEAD_PARALLEL_RESULTS.md)
- [`results/TRANSPORT_RESULTS.md`](results/TRANSPORT_RESULTS.md)
- [`results/HEAD_PARALLEL_RESULTS.md`](results/HEAD_PARALLEL_RESULTS.md)
- [`results/KV_SHARDED_RESULTS.md`](results/KV_SHARDED_RESULTS.md)

## Reproduce

From the repository root:

```bash
cp benchmarks/case-02/multiboard/esp32_cluster_transport/secrets.example.h \
   benchmarks/case-02/multiboard/esp32_cluster_transport/secrets.h
arduino-cli compile --fqbn esp32:esp32:XIAO_ESP32C3 \
  benchmarks/case-02/multiboard/esp32_cluster_transport
.venv/bin/python benchmarks/case-02/multiboard/tools/run_large_head_parallel.py \
  --workers <C3-A-IP>,<C3-B-IP>,<C3-C-IP>,<C3-D-IP> \
  --scheduler round-robin
```

The worker is intended for a trusted benchmark LAN and has no authentication.
