# Case 09

Configuration: `B=64, S=128, D=128, H=1, F=128, L=4`, causal.

Status: **not implemented on ESP32**. No physical timing or accuracy result is
claimed.

Likely focus: batch, feature, or key/value sequence sharding. With one head,
the case-2 whole-head strategy offers no parallelism, so a different
decomposition and its reduction traffic must be measured here.
