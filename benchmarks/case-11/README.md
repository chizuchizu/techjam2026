# Case 11

Configuration: `B=64, S=128, D=128, H=16, F=128, L=4`, causal.

Status: **not implemented on ESP32**. No physical timing or accuracy result is
claimed.

Likely focus: fine-grained head parallelism and head grouping per worker. The
small head width increases protocol overhead, so measure grouping, stragglers,
and batch parallel alternatives rather than extrapolating case-2 efficiency.
