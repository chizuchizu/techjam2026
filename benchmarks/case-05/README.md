# Case 05

Configuration: `B=128, S=128, D=128, H=4, F=128, L=4`, causal.

Status: **not implemented on ESP32**. No physical timing or accuracy result is
claimed.

Likely focus: stream batch tiles through resident model weights and distribute
independent samples across boards. This case should optimise aggregate
throughput without representing pipeline fill time as single-sample latency.
