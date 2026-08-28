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
   accuracy dispatch (known FP16 shape: four unmasked, three padded, two causal).
   Prefer the implemented score-rounded Triton attention for the known default
   FP16 shape: all six layers are bit-exact in the stronger 100-trial audit for
   unmasked, padded, causal, and causal+padded modes.
4. Extend raw CUDA graph replay (now verified for FP16/BF16/FP32) across the
   official static shape and mask regimes.
5. Keep the implemented static-mask hoisting/final-only invalid-row zeroing and
   verify it across the official padding-mask cases.
6. Extend the rejected Triton residual-add + LayerNorm prototype toward a
   numerically exact linear fusion; then test fused exact-GELU FFN.
7. Add whole-block valid-token packing for padded cases.
8. Extend the verified FP32 `torch.compile`/CUDA graph path across the official
   shape table; the fixed unmasked TensorRT FP32 graph path is now implemented
   and faster, while TensorRT FP16 fails accuracy. Treat Transformer Engine and
   FP8 as separate accuracy-gated experiments.
9. Later, investigate novel sensitivity-guided approximations: identify the
   measured bottleneck, estimate layerwise error amplification, then apply
   cheaper math/precision only where the final elementwise tolerance has margin.

you can use nvhpc 26.2 from here
source /export/home/alien/software/nvhpc/setup.sh
