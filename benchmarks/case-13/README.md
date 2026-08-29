# Case 13

Configuration: `B=64, S=1024, D=128, H=4, F=128, L=4`, causal.

Status: **not implemented on ESP32**. No physical timing or accuracy result is
claimed.

Likely focus: block-online attention to avoid materialising `S x S` scores,
plus exact key/value sequence sharding when a whole head is too large. Measure
the larger softmax-statistic traffic and merge cost.
