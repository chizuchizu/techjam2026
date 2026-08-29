# Case 12

Configuration: `B=64, S=32, D=128, H=4, F=128, L=4`, causal.

Status: **not implemented on ESP32**. No physical timing or accuracy result is
claimed.

Likely focus: kernel fusion and batch parallelism. Short sequences reduce
attention work, making setup, transport, and projection costs relatively more
important than in case 2.
