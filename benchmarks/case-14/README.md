# Case 14

Configuration: `B=32, S=100000, D=1024, H=16, F=1024, L=2`, causal.

Status: **not implemented on ESP32**. No physical timing or accuracy result is
claimed.

Likely focus: external/streamed state, block-online causal attention, and
combined sequence, feature, and batch partitioning. Any proposal must first
show that storage, transfer volume, and runtime are feasible; case-2 timings
cannot be extrapolated to this shape.
