# Case-2 LayerNorm-to-attention cluster result

The cluster worker now starts from the official case-2 activation `X`, rather
than accepting precomputed Q/K/V. On every assigned head it performs:

1. layer-0 LayerNorm across all 128 model features using the official epsilon,
   gamma, and beta;
2. the assigned rows of the official layer-0 Q/K/V projections using the
   baseline's Q15 x Q12 numeric path; and
3. one causal attention head at `S=128, d_head=32`.

The worker returns both its 32-feature LayerNorm slice and its 32-feature
attention context. Across four heads, the coordinator therefore validates all
16,384 normalized values and all 16,384 context values directly.

## Physical timing

Two Seeed XIAO ESP32-C3 boards ran two heads each over persistent Wi-Fi TCP.
The baseline uses the same firmware on one board and executes all four heads
sequentially. Seed-0 timings are medians after one warm-up and three measured
repetitions.

| Workers | Assignment | Wall time | Speedup | Efficiency |
|---:|---|---:|---:|---:|
| 1 C3 | `0,0,0,0` | 9.693 s | 1.00x | — |
| 2 C3 | `0,1,0,1` | **4.850 s** | **2.00x** | **99.9%** |
| 4 C3 | `0,1,2,3` | Pending firmware upload to two LAN-only boards | — | — |

The median critical-head timing on the two-board run is approximately 0.360 s
for LayerNorm and activation quantization, 0.265 s for all three projections,
and 1.697 s for causal attention. Each request carries 32,784 bytes and each
response carries 32,796 bytes; communication and host scheduling are included
in wall time.

## Accuracy

Physical runs cover the five exported device seeds, 0 through 4. Every value
passes the competition rule `abs_error <= 0.002 OR rel_error <= 0.02`.

| Comparison | Worst max-absolute error | Failed elements |
|---|---:|---:|
| ESP32 LayerNorm vs host emulation of the quantized kernel | 0.000001431 | 0 |
| ESP32 LayerNorm vs official fp32 LayerNorm | 0.000101448 | 0 |
| ESP32 context vs host emulation of the quantized kernel | 0.000064850 | 0 |
| ESP32 context vs official fp32 attention context | 0.000447452 | 0 |

The firmware embeds only the official layer-0 Norm/Q/K/V weights needed for
this split. They are generated reproducibly from the baseline artifacts by
`benchmarks/case-02/multiboard/tools/export_case2_cluster_layer0.py`; no
training is performed.

## Scope boundary

This result adds the previously missing first LayerNorm and Q/K/V projections,
but it is not yet a complete official layer or case-2 inference. Output
projection, attention residual, second LayerNorm, FFN, FFN residual, the other
three layers, and final LayerNorm remain outside this distributed path.

Raw captures:

- [`xiao_c3_case2_layer0_norm_head_single_v1.csv`](xiao_c3_case2_layer0_norm_head_single_v1.csv)
- [`two_xiao_c3_case2_layer0_norm_head_v1.csv`](two_xiao_c3_case2_layer0_norm_head_v1.csv)
- [`two_xiao_c3_case2_layer0_norm_head_seeds_0_4_v1.csv`](two_xiao_c3_case2_layer0_norm_head_seeds_0_4_v1.csv)
