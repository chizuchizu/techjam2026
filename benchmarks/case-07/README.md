# Case 07

Configuration: `B=64, S=128, D=32, H=4, F=32, L=4`, causal.

Status: **not implemented on ESP32**. No physical timing or accuracy result is
claimed.

Likely focus: fuse narrow projections and normalization because dispatch and
loop overhead may dominate the small matrices. Compare that single-board path
with batch parallelism; do not assume head distribution will amortise network
cost.
