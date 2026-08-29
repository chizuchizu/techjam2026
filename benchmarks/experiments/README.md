# Supporting experiments

These benchmarks informed the official case implementations but do not match
an official test case end to end. Their results must not be presented as
official case timings.

| Experiment | Purpose | Result |
|---|---|---|
| [`attention-layer/`](attention-layer/) | Single-board attention kernels and complete small attention layer | Up to 3.87x end-to-end speedup |
| [`tiny-transformer/`](tiny-transformer/) | Complete trained two-block character model | 106.614 ms/forward, 9.38 token/s |
| [`random_weight_transformer/`](random_weight_transformer/) | Generated-weight memory experiment | Experimental; no canonical result |
