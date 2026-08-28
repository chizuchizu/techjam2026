We are targetin the 1st place in this competition.
problemstatement is defined STATEMENT.md and benchmark is placed in torch_transformer_benchmark.py
Also we try to understand the kernel things and want to apply other competition.

1. Define a baseline
2. keep do this
use nsys and ncu to profild and analyze this application for 1 x h200, identify bottlenecks, and launch parallel research agents to find how existing open source or literature or competition (like gpu-mode) have tried optimising it, and whether it already exist in the current code. If not, list them out as todo, document findings in markdown. Also include in implementation difficulty and expected gain and identify if any of them may compound or conflict, or are prerequisit of others. Then it will produce the markdown Then you review and see which make most sense to you, and proriotize yourself.
Then once you know which to do first, '/goal' make a new branch in a new git worktree and implement XXX. Make changes an opt-in via build or runtime flag/env then do A/B testing.
Also every experiments should be simple so that i can trace later.
You should push the performance as much as possible but the same time keep track why it worked
Also calculate or estimate the theoretical performance and compare the performance with it and make a document.

## Current prioritized work (2026-08-28)

Detailed evidence, estimates, dependencies, and rejected experiments are in
`OPTIMIZATION_REPORT.md`.

1. Recover the missing official Feishu test-shape/dtype/mask table.
2. Enable Nsight Compute performance counters and collect per-kernel rooflines.
3. Keep and expand the opt-in packed-QKV + SDPA implementation with per-shape
   accuracy dispatch (four layers currently pass the default FP16 case).
4. Prototype fused residual + LayerNorm + linear, then fused exact-GELU FFN.
5. Add whole-block valid-token packing for padded cases.
6. Extend the verified FP32 `torch.compile`/CUDA graph path across the official
   shape table; treat Transformer Engine and FP8 as separate accuracy-gated
   experiments.

you can use nvhpc 26.2 from here
source /export/home/alien/software/nvhpc/setup.sh
