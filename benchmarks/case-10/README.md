# Case 10

Configuration: `B=64, S=128, D=128, H=2, F=128, L=4`, causal.

Status: **not implemented on ESP32**. No physical timing or accuracy result is
claimed.

Likely focus: at most two whole-head shards per sample, combined with batch
parallelism across additional boards. Compare this hybrid assignment with pure
batch sharding under identical communication accounting.
