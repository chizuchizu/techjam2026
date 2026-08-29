# Two matched ESP32-C3 head-parallel result

Two Seeed XIAO ESP32-C3 boards ran the same worker firmware on the same Wi-Fi
LAN. Both report one core at 160 MHz and the same `128 x 32` maximum head shape.
Private addresses and hardware identifiers are intentionally omitted.

The task is four independent attention heads at `N=128, d_head=32`. Q/K are
int8, V is int16, and each returned context is float32. Each head transfers a
16,420-byte TCP request and a 16,400-byte TCP response. A complete layer moves
131,280 payload bytes.

## Measured scaling

Each run uses one warm-up. Single-node results contain five measured
repetitions, round-robin contains ten, and calibrated contains five. The
single-node baseline for speedup is the arithmetic mean of the two isolated
board medians.

| Configuration | Assignment | Non-causal | Speedup | Efficiency | Causal | Speedup | Efficiency |
|---|---|---:|---:|---:|---:|---:|---:|
| C3 node A only | `0,0,0,0` | 5.767 s | — | — | 3.022 s | — | — |
| C3 node B only | `0,0,0,0` | 5.756 s | — | — | 3.021 s | — | — |
| Two C3, round-robin | `0,1,0,1` | 2.881 s | **2.00x** | **100.0%** | 1.523 s | **1.98x** | **99.2%** |
| Two C3, calibrated | balanced 2+2 | 2.889 s | **1.99x** | **99.7%** | 1.512 s | **2.00x** | **99.9%** |

The two isolated boards differ by about 0.2% non-causally and less than 0.02%
causally. Calibration therefore selects two heads per board: `0,1,0,1` for the
non-causal run and the equivalent `1,0,1,0` for the causal run.

Every returned element passes the combined accuracy rule. The worst absolute
error is 0.000003174, with zero failed elements. Calibration is recorded in the
CSV but excluded from the timed warm-up and repetitions; a fleet coordinator
should cache this profile.

## Raw captures

- [`xiao_c3_node_a_large_head_tcp_v1.csv`](xiao_c3_node_a_large_head_tcp_v1.csv)
- [`xiao_c3_node_b_large_head_tcp_v1.csv`](xiao_c3_node_b_large_head_tcp_v1.csv)
- [`two_xiao_c3_large_head_round_robin_v1.csv`](two_xiao_c3_large_head_round_robin_v1.csv)
- [`two_xiao_c3_large_head_calibrated_v1.csv`](two_xiao_c3_large_head_calibrated_v1.csv)

Reproduce with explicit worker addresses:

```bash
.venv/bin/python benchmarks/case-02/multiboard/tools/run_large_head_parallel.py \
  --workers <C3-A-IP>,<C3-B-IP> \
  --scheduler round-robin --warmups 1 --repetitions 10 \
  --output benchmarks/case-02/multiboard/results/two_xiao_c3_large_head_round_robin_v1.csv
```

This demonstrates attention-head parallelism, not a complete distributed
Transformer. The next integration step is keeping layer weights resident and
including Q/K/V plus output projections in the distributed layer timing.
