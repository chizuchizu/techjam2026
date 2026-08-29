# Case-02 multiboard approaches

This directory keeps the entire distributed case-02 path together: link tests,
worker firmware, host coordinators, scheduling policies, raw captures, and
interpreted results.

## Contents

| Directory | Purpose |
|---|---|
| [`esp32-cluster-full/`](esp32-cluster-full/) | **Complete two-board distributed case-2 forward** |
| [`esp32_cluster_transport/`](esp32_cluster_transport/) | UDP/TCP attention and official layer-0 worker |
| [`esp32-linkbench/`](esp32-linkbench/) | Two-board ESP-NOW bandwidth benchmark |
| [`tools/`](tools/) | Discovery, transport, head-parallel, KV-sharded, and layer-0 coordinators |
| [`results/`](results/) | Raw captures and result reports |

## Complete distributed forward

| Scope | Boards | Wall time | Speedup | Accuracy |
|---|---:|---:|---:|---|
| All four layers, complete case-2 body | 1 C3 | 1.990 s | 1.00x | Pass, 5/5 seeds |
| All four layers, complete case-2 body | 2 C3s | **1.276 s** | **1.56x** | Pass, 0 failing elements |

See [`esp32-cluster-full/`](esp32-cluster-full/) and
[`results/CASE2_FULL_E2E_RESULTS.md`](results/CASE2_FULL_E2E_RESULTS.md).

## Earlier partial-scope results

| Scope | Boards | Wall time | Speedup | Accuracy |
|---|---:|---:|---:|---|
| Layer-0 LayerNorm + Q/K/V + causal attention | 1 C3 | 9.693 s | 1.00x | Pass |
| Layer-0 LayerNorm + Q/K/V + causal attention | 2 C3s | **4.850 s** | **2.00x** | Pass, 5/5 seeds |
| Four causal `128 x 32` attention heads | 1 C3 | 3.000 s average | 1.00x | Pass |
| Four causal `128 x 32` attention heads | 4 C3s | **0.766 s** | **3.92x** | Pass, zero failed elements |

These are partial case-02 paths. See [`../README.md`](../README.md) for the
exact scope exclusions.

Detailed reports:

- [`results/CASE2_FULL_E2E_RESULTS.md`](results/CASE2_FULL_E2E_RESULTS.md)
- [`results/CASE2_NORM_HEAD_RESULTS.md`](results/CASE2_NORM_HEAD_RESULTS.md)
- [`results/FOUR_C3_PARALLEL_RESULTS.md`](results/FOUR_C3_PARALLEL_RESULTS.md)
- [`results/TWO_C3_PARALLEL_RESULTS.md`](results/TWO_C3_PARALLEL_RESULTS.md)
- [`results/LARGE_HEAD_PARALLEL_RESULTS.md`](results/LARGE_HEAD_PARALLEL_RESULTS.md)
- [`results/TRANSPORT_RESULTS.md`](results/TRANSPORT_RESULTS.md)
- [`results/HEAD_PARALLEL_RESULTS.md`](results/HEAD_PARALLEL_RESULTS.md)
- [`results/KV_SHARDED_RESULTS.md`](results/KV_SHARDED_RESULTS.md)

## Reproduce

From the repository root:

The complete distributed forward (two boards, no router or credentials):

```bash
cd benchmarks/case-02/multiboard/esp32-cluster-full
./tools/flash_boards.sh
python3 tools/run_cluster_e2e.py --seeds 0 1 2 3 4   # accuracy
python3 tools/time_cluster.py --reps 6              # timing
```

The earlier partial-scope head-parallel path:

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
