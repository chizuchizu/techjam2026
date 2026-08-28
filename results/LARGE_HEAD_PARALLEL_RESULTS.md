# Official-size head-parallel transport results

The cluster worker accepts the teammate benchmark's attention-head shape:
`N=128`, four heads, and `d_head=32`. Q/K are int8, V is int16, and the returned
context is float32. Each task uses a persistent TCP connection and carries a
16,420-byte request plus a 16,400-byte response.

## Physical hardware

Two different ESP32 boards were flashed with the same worker firmware and
joined to the same Wi-Fi LAN:

| Worker | Chip | Cores | Clock | Maximum head shape |
|---|---|---:|---:|---|
| 0 | Seeed XIAO ESP32-C3 | 1 | 160 MHz | `128 x 32` |
| 1 | ESP32-D0WDQ6 via CP210x | 2 | 240 MHz | `128 x 32` |

The coordinator queried these properties from the devices rather than assuming
identical hardware. IP addresses and Wi-Fi credentials are deliberately absent
from tracked result files.

## End-to-end measurements

Times are medians after one warm-up. Every row transfers four requests and four
responses (131,280 payload bytes total) and reconstructs all 16,384 output
elements. Speedup uses the one-XIAO result as the baseline.

| Workers and policy | Head assignment | Non-causal | Speedup | Causal | Speedup | Gate |
|---|---|---:|---:|---:|---:|---|
| XIAO only | `0,0,0,0` | 5.731 s | 1.00x | 3.064 s | 1.00x | pass |
| Dual-core ESP32 only | `0,0,0,0` | 0.622 s | 9.21x | 0.434 s | 7.06x | pass |
| Both, round-robin | `0,1,0,1` | 2.949 s | 1.94x | 1.573 s | 1.95x | pass |
| Both, calibrated-all | `0,1,1,1` | 1.449 s | 3.96x | 0.779 s | 3.93x | pass |
| Both, calibrated | `1,1,1,1` | 0.616 s | 9.31x | 0.460 s | 6.65x | pass |

The maximum absolute error across the two-board runs is 0.000003174, below the
project's combined absolute/relative tolerance, with zero failed elements.

The raw captures are:

- [`esp32c3_large_head_tcp_v1.csv`](esp32c3_large_head_tcp_v1.csv): XIAO-only
  baseline, three measured repetitions.
- [`esp32_dualcore_large_head_tcp_v1.csv`](esp32_dualcore_large_head_tcp_v1.csv):
  dual-core-only baseline, three measured repetitions.
- [`two_esp32_large_head_round_robin_v1.csv`](two_esp32_large_head_round_robin_v1.csv):
  equal static split, three measured repetitions.
- [`two_esp32_large_head_calibrated_all_v1.csv`](two_esp32_large_head_calibrated_all_v1.csv):
  profiled split that uses both nodes, five measured repetitions.
- [`two_esp32_large_head_calibrated_v1.csv`](two_esp32_large_head_calibrated_v1.csv):
  unconstrained profiled split, five measured repetitions.

## Interpretation

This is a real concurrent two-board result, but it is not a homogeneous
two-node scaling result. One head takes about 1.38 s of compute on the C3 and
0.096 s on the dual-core ESP32. Equal round-robin assignment serializes two
heads on the slow node, so its critical path dominates. The `calibrated-all`
policy gives the C3 one head and the faster board three, improving the XIAO
baseline by about 3.9x while still using both devices.

The latency-minimizing `calibrated` policy correctly assigns all four heads to
the faster board. Its result is essentially the dual-core-only baseline. This
is important negative evidence: adding a much slower node cannot improve this
four-task batch. Homogeneous scaling still requires two comparable boards.

Calibration sends one head to each worker before the timed warm-up and records
the elapsed values in the CSV. Its cost is excluded from the table, so a
deployment should cache each board's performance profile; recalibrating for
every one-shot request would erase the benefit.

The small `N=16, d_head=8` regression also passes over both UDP and TCP. Its raw
captures are [`esp32c3_head_parallel_udp_v2.csv`](esp32c3_head_parallel_udp_v2.csv)
and [`esp32c3_head_parallel_tcp_v1.csv`](esp32c3_head_parallel_tcp_v1.csv).
