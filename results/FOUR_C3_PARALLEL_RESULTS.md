# Four ESP32-C3 head-parallel result

Four Seeed XIAO ESP32-C3 boards ran the same worker firmware on one Wi-Fi LAN.
Every worker reported one core at 160 MHz and support for the `128 x 32` head
shape. Private addresses and hardware identifiers are intentionally omitted.

This is a synthetic attention microbenchmark, not complete Transformer
inference. The host constructs four deterministic heads at `N=128,
d_head=32`; Q/K are int8, V is int16, and each returned context is float32.
There are no embeddings, learned weights, Q/K/V projections, output projection,
feed-forward network, or token generation in this measurement. Each head moves
a 16,420-byte TCP request and a 16,400-byte TCP response.

## Measured scaling

Each run uses one warm-up. Nodes A and B have five isolated repetitions, nodes
C and D have three, the four-node round-robin run has ten, and calibrated-all
has five. The single-node baseline is the arithmetic mean of the four isolated
board medians. Speedup includes host scheduling and Wi-Fi TCP communication.

| Configuration | Assignment | Non-causal | Speedup | Efficiency | Causal | Speedup | Efficiency |
|---|---|---:|---:|---:|---:|---:|---:|
| C3 node A only | `0,0,0,0` | 5.767 s | — | — | 3.022 s | — | — |
| C3 node B only | `0,0,0,0` | 5.756 s | — | — | 3.021 s | — | — |
| C3 node C only | `0,0,0,0` | 5.845 s | — | — | 2.979 s | — | — |
| C3 node D only | `0,0,0,0` | 5.743 s | — | — | 2.980 s | — | — |
| Four C3, round-robin | `0,1,2,3` | **1.453 s** | **3.98x** | **99.4%** | **0.766 s** | **3.92x** | **98.0%** |
| Four C3, calibrated-all | `0,1,2,3` | **1.457 s** | **3.97x** | **99.2%** | **0.766 s** | **3.92x** | **97.9%** |

The coordinator validates every measured response against its host float
reference. All 16,384 output elements per mode pass in every repetition. The
worst absolute error is 0.000003174, below the 0.002 absolute-error gate, and
there are zero failed elements. The raw CSVs retain more timing precision than
the table, but Wi-Fi variation does not justify reporting the speedup to four
decimal places.

## Raw captures

- [`xiao_c3_node_a_large_head_tcp_v1.csv`](xiao_c3_node_a_large_head_tcp_v1.csv)
- [`xiao_c3_node_b_large_head_tcp_v1.csv`](xiao_c3_node_b_large_head_tcp_v1.csv)
- [`xiao_c3_node_c_large_head_tcp_v1.csv`](xiao_c3_node_c_large_head_tcp_v1.csv)
- [`xiao_c3_node_d_large_head_tcp_v1.csv`](xiao_c3_node_d_large_head_tcp_v1.csv)
- [`four_xiao_c3_large_head_round_robin_v1.csv`](four_xiao_c3_large_head_round_robin_v1.csv)
- [`four_xiao_c3_large_head_calibrated_all_v1.csv`](four_xiao_c3_large_head_calibrated_all_v1.csv)

Reproduce with explicit worker addresses:

```bash
.venv/bin/python tools/run_large_head_parallel.py \
  --workers <C3-A-IP>,<C3-B-IP>,<C3-C-IP>,<C3-D-IP> \
  --scheduler round-robin --warmups 1 --repetitions 10 \
  --output results/four_xiao_c3_large_head_round_robin_v1.csv
```

Four boards are useful here because the fixture has exactly four independent
heads. The next meaningful step is to integrate this transport with one
complete official-size layer and include its projections and residual path in
the end-to-end timing.
