# Case 04

Configuration: `B=16, S=128, D=128, H=4, F=128, L=4`, causal.

Status: **not implemented on ESP32**. No physical timing or accuracy result is
claimed.

Likely focus: batch tiling with resident weights on a single board, then
throughput-oriented batch sharding across boards. Record both per-batch latency
and steady-state throughput inside this directory.
