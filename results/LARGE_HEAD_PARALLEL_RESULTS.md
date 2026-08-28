# Official-size head-parallel transport baseline

The cluster worker now accepts the teammate benchmark's attention-head shape:
`N=128`, four heads, and `d_head=32`. Q/K are int8, V is int16, and the returned
context is float32. Each task uses a persistent TCP connection because its
payload is too large for a safe UDP datagram.

The authoritative raw capture is
[`esp32c3_large_head_tcp_v1.csv`](esp32c3_large_head_tcp_v1.csv). It contains
three measured repetitions after one warm-up on the physical XIAO ESP32-C3.

| Mode | Four heads, one worker | Worker compute total | Per-head request | Per-head response | Max abs | Gate |
|---|---:|---:|---:|---:|---:|---|
| Non-causal | 5.731 s | 5.529 s | 16,420 B | 16,400 B | 0.000000952 | pass |
| Causal | 3.064 s | 2.790 s | 16,420 B | 16,400 B | 0.000003174 | pass |

All 16,384 returned elements per mode pass the project accuracy rule. The
small `N=16, d_head=8` regression also passes over both UDP and TCP; its raw
captures are [`esp32c3_head_parallel_udp_v2.csv`](esp32c3_head_parallel_udp_v2.csv)
and [`esp32c3_head_parallel_tcp_v1.csv`](esp32c3_head_parallel_tcp_v1.csv).

This is a one-worker baseline. With two workers, round-robin scheduling assigns
two heads to each and runs the workers concurrently; with four, it assigns one
head per worker. The one-worker medians suggest compute-only lower bounds near
2.76 s / 1.40 s for two workers and 1.38 s / 0.70 s for four workers, but those
are projections, not results. Concurrent Wi-Fi transfers, unequal ESP32 models,
and stragglers must be measured on the actual boards.

The firmware exposes a capability query containing chip model, core count,
clock, free heap, maximum shape, and supported transports. This lets the future
coordinator replace round-robin assignment with measured heterogeneous load
balancing.
