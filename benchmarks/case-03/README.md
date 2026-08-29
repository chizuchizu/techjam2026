# Case 03

Configuration: `B=4, S=128, D=128, H=4, F=128, L=4`, causal.

Status: **not implemented on ESP32**. No physical timing or accuracy result is
claimed.

Likely focus: evaluate whether four independent batch items should be streamed
on one board or assigned one per board. Reuse of case-2 weights and kernels must
be measured rather than inferred. Add all case-3 code and results under this
directory.
