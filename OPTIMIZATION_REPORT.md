# Optimization Report

## Scope and current limitation

The checked-in problem statement promises a fixed test-shape appendix, but the
export contains only “This content is only supported in a Feishu Docs”. The
default benchmark shape is therefore the only authoritative local target:

`B=8, S=128, D=512, H=8, F=2048, L=6`, noncausal, no padding.

All shape-specialized work remains provisional until the official table and
tested dtypes/mask modes are recovered.

## Baseline and current result

Hardware/software used on 2026-08-28:

- one NVIDIA H200 NVL, SM90, 143,771 MiB reported memory;
- driver 610.43.02;
- PyTorch 2.13.0+cu130 and Triton 3.7.1;
- NVHPC setup reports 26.5 (the TODO's stated 26.2 path currently loads 26.5).

Default-shape FP16 results use alternating baseline/optimized rounds and CUDA
events. Absolute clocks varied between runs, so speedup from the interleaved A/B
is the primary metric.

| Experiment | Accuracy | Median baseline | Median candidate | Speedup | Decision |
|---|---:|---:|---:|---:|---|
| No-op baseline | pass, exact | 2.534 ms | 2.524 ms | 1.004x | Measurement sanity check |
| SDPA, all 6 layers | fail, 4/1,572,864 | — | — | — | Reject |
| SDPA, trailing 4 layers | pass, 0/13,107,200 | 2.561 ms | 2.160 ms | 1.186x | Keep |
| Packed QKV + SDPA, trailing 4, no redundant mask | pass, 0/13,107,200 | 2.425 ms | 1.829 ms | **1.325x** | Best eager FP16 |
| Compile packed-QKV/SDPA (`reduce-overhead`) | fail, 46/2,621,440 | — | — | — | Reject for FP16 |
| Compile with eager precision casts + rounded division + CUDA libdevice | fail, 45/13,107,200 | — | — | — | Reject: same FP16 error pattern |
| Previous FP16 path + packed prefix QKV + raw CUDA graph | pass, 0/13,107,200 | 2.389 ms | 0.391 ms | **6.116x** | Current FP16 best |
| Padded FP16, three SDPA layers + mask cleanup + raw graph | pass, 0/13,107,200 | 2.612 ms | 0.469 ms | **5.575x** | Current padded best |
| Causal FP16, two SDPA layers + raw graph | pass, 0/13,107,200 | 2.295 ms | 0.477 ms | **4.809x** | Keep |
| Causal+padded FP16, two SDPA layers + raw graph | pass, 0/13,107,200 | 2.879 ms | 0.528 ms | **5.451x** | Current causal+padded best |
| BF16 packed QKV + reference attention + raw graph | exact, 0/13,107,200 | 2.396 ms | 0.492 ms | **4.871x** | Current BF16 best |
| Causal+padded BF16, same path | exact, 0/5,242,880 | 3.790 ms | 0.603 ms | **6.286x** | Keep |
| FP32 packed QKV + all-layer SDPA | pass, 0/5,242,880 | 2.296 ms | 1.185 ms | 1.938x | Keep |
| FP32 previous row + raw CUDA graph | pass, 0/5,242,880 | 2.389 ms | 0.562 ms | 4.248x | Safer/no compiler startup |
| FP32 previous row + `reduce-overhead` compile | pass, 0/13,107,200 | 2.302 ms | 0.549 ms | **4.197x** | Current default-dtype best |
| FP32 previous row + `max-autotune` | pass, 0/5,242,880 | 1.921 ms | 0.525 ms | 3.659x | Slightly faster candidate, less accuracy headroom |
| FP32 with FP16 attention core | pass | 2.305 ms | 0.549 ms compiled | 4.200x | Reject: no compiled gain, slower eager |
| FP16 tanh-approximate GELU | fail, 74/13,107,200 | — | — | — | Reject: exceeds elementwise tolerance |

The packed weights are built once after weight copying and device/dtype transfer,
outside accuracy and timing regions. Original parameter names remain intact.

The conservative one-layer SDPA setting also passed 25 FP16 trials for each of
noncausal, causal, padded, and causal+padded default-shape cases. BF16 SDPA did
not satisfy the fixed 0.002 absolute tolerance, so BF16 dispatches to the exact
baseline. Current PyTorch accepts `attn_mask` together with `is_causal`; this is
version-sensitive and contradicts older/documented pseudocode, so it remains an
explicit test dimension.

Automatic dispatch now uses all SDPA layers for FP32, four only for the known
default noncausal FP16 shape, three for its padded noncausal regime, two for its
causal regimes, one for undisclosed FP16 shapes, and zero for BF16.
The packed implementation still combines Q/K/V in BF16 while retaining the
reference attention algorithm; tested BF16 outputs are bit-exact. CPU uses the
baseline. Explicit `--sdpa-layers` values override FP32/FP16 selection.

Compiled FP32 `reduce-overhead` also passed tested small, default,
causal+padded, and long-sequence shapes. Observed speedups were respectively
6.36x, 4.20x, 6.99x, and 1.10x, confirming that CUDA graph/launch fusion is most
valuable for launch-bound cases and modest for large compute-bound work.

PyTorch 2.13's FP16 eager-numerics controls did not rescue whole-model
compilation. `TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1`, rounded division, and
the CUDA 13.2 `libdevice.10.bc` produced the same 45 failures over 13,107,200
outputs. Compiling the untouched baseline reproduced the same errors, proving
that packed QKV and SDPA were not responsible. Module-boundary isolation found
that compiling only the final LayerNorm passed five trials, while compiling all
LayerNorms or attention modules failed; individual compiled FFN islands passed
short screens but their accumulated six-layer result failed. Since raw CUDA
graph replay already removes island launch overhead without changing eager
arithmetic, the partial-compile route is not retained.

Raw CUDA graph capture preserves the selected eager kernels and their numerical
results. It passed 25 default FP16 trials, 25 causal+padded FP16 trials, 25
BF16 trials with exact output, and ten FP32 trials. Default FP16 reached 0.391
ms and 6.12x versus baseline. Padded and causal+padded FP16 reached 0.469 ms and
0.528 ms; BF16 reached 4.87x unmasked and 6.29x causal+padded, and FP32 4.25x.
Packing QKV in the two reference-attention prefix layers was
numerically identical to the prior candidate and removed four more graph nodes.

For padded cases, the candidate now negates the padding mask once, not once per
consumer. A causal triangle is created once during optimized-weight setup rather
than replayed in every reference-attention layer. Intermediate invalid query
rows are not zeroed: LayerNorm/FFN never mix tokens, every attention excludes
invalid keys, and one final mask restores the required zero output. Across the
tested cases this dead-work elimination leaves every valid output unchanged.

## Nsight Systems evidence

A CUDA-profiler capture range isolated one warmed-up forward after the all-true
mask cleanup. The baseline used 127 GPU kernel launches and 492.1 microseconds
of summed GPU kernel duration. Packed-QKV plus four SDPA layers used 83 launches
and 358.1 microseconds:

- **34.6% fewer launches**;
- **27.2% less summed GPU kernel time**;
- the optimized trace contains four cuDNN generated SM90 Flash SDPA kernels;
- explicit softmax, score-mask, score conversion, and attention layout kernels
  disappear from those four layers;
- packing replaces 12 Q/K/V projection GEMMs with four wider QKV GEMMs.

Before the all-true mask cleanup, redundant mask fills consumed 12% of traced
kernel time. In the final baseline trace the largest groups are projection/FFN
GEMMs, dtype/layout copies, LayerNorm, GELU, and softmax. This explains why
attention fusion helps despite attention being a small share of arithmetic: it
removes many launch- and bandwidth-bound kernels.

For default FP32, the eager baseline trace contained 115 launches and 687.6
microseconds of summed GPU kernel time. Compiled packed-QKV/SDPA contains 50
CUDA-graph nodes and 525.3 microseconds of GPU kernel time. The host submits the
compiled forward with one `cudaGraphLaunch`; the eager baseline makes 115 CUDA
kernel-launch API calls. Thus only about one quarter of the GPU work disappears,
while most of the 4.20x latency gain comes from eliminating serial host launch
gaps and graph-safe buffer reuse. In the compiled trace, GEMMs consume about 60%
of GPU time, attention 21%, fused GELU 6%, and fused residual/LayerNorm kernels
about 8%.

For default FP16, the final node-level raw-graph trace contains 79 computation
nodes and 342.5 microseconds of summed GPU kernel time. Host submission changes
from 79 individual launches to one
`cudaGraphLaunch`, plus one async input copy and one output clone. The measured
0.391 ms latency is therefore close to actual GPU work rather than host launch
gaps. The output clone deliberately preserves normal tensor ownership semantics;
without it, a later graph replay would overwrite a previous result.

The final two-SDPA-layer causal+padded FP16 trace has 113 nodes and 455.3
microseconds of GPU work. The one-layer path after dead-mask removal had 121
nodes and 483.9 microseconds; before that cleanup it had 132 nodes and 520.6
microseconds. Mask fills fell from 22 to 11 nodes, and hoisting causal-mask
construction removes additional fill/triangle work from every replay.

An exhaustive layer-placement screen confirmed the safe SDPA-count boundaries
on the known FP16 shape. No five-layer unmasked subset, no four-layer padded
subset, and no four-layer causal subset passed even five trials. Several
three-layer causal placements initially passed five trials, but alternative
`[2,3,5]` failed seven elements over 25 trials and trailing three failed one.
Three noncausal padded layers and two causal layers passed 25 trials, including
padding ratios 0.10, 0.25, 0.50, and 0.75. `--sdpa-layer-indices` preserves this
experimental control while automatic dispatch uses the proven trailing sets.

Nsight Compute is currently blocked by `ERR_NVGPUCTRPERM`. Hardware-counter
results require an administrator to enable non-admin profiling, commonly via
`NVreg_RestrictProfilingToAdminUsers=0`, followed by the required driver reload.
Until then, published roofs below are estimates and `nsys` supplies measured
kernel timing/launch evidence.

## Theoretical model

Let `T = B*S`. Counting a fused multiply-add as two FLOPs, dominant work per
Transformer block is:

```text
Q/K/V projections       6*T*D^2
attention output        2*T*D^2
QK^T and P*V            4*B*S^2*D
two FFN projections     4*T*D*F
total                   4*B*S*D*(2*D + F + S)
```

For the default shape this is 6.711 GFLOP/block and **40.265 GFLOP/call**:

| Component | Per block | Share |
|---|---:|---:|
| Q/K/V projections | 1.611 GFLOP | 24% |
| attention output projection | 0.537 GFLOP | 8% |
| QK and PV | 0.268 GFLOP | 4% |
| FFN | 4.295 GFLOP | 64% |

The model has 18,915,328 parameters (37.83 MB at FP16). NVIDIA publishes H200
NVL figures of 4.8 TB/s HBM bandwidth and 1,671 TFLOP/s FP16/BF16 Tensor Core
throughput with sparsity. Dense random weights cannot use 2:4 sparsity, so the
dense roof is approximately 835.5 TFLOP/s. The ideal compute-only floor is
48.2 microseconds. An eager logical-traffic estimate is about 570.5 MB/call,
whose 4.8 TB/s floor is 118.9 microseconds. Neither bound includes launch cost,
small-GEMM underutilization, reductions, exact GELU, or synchronization.

The measured baseline median of 2.4–3.3 ms is about 50–69 times the ideal dense
compute floor. This gap is expected for many medium GEMMs and 127 serial kernel
launches; it is an optimization opportunity, not evidence that the published
peak is attainable for this shape. Attention arithmetic becomes dominant over
projections plus FFN only around `S > 2D + F` (3072 for default D/F).

## Prioritized optimization backlog

| Priority | Work | Present? | Expected end-to-end gain | Difficulty | Dependencies / conflicts |
|---|---|---:|---:|---:|---|
| P0 | Recover official shapes and dtype/mask matrix | No | Enables reliable dispatch | Low/admin | Prerequisite for every specialization |
| P0 | Accuracy/backend gate SDPA per case | Partial | 1.05–2x, sequence-dependent | Low–medium | BF16 currently conflicts with tolerance |
| P1 | Packed QKV projection | **Implemented** | Observed incremental gain; usually 5–15% | Medium | Compounds with SDPA and BSHD layout |
| P1 | Compile/CUDA graph mode sweep | **Implemented for FP32; FP16 rejected** | Observed 1.10–6.99x | Low | Static shapes; accuracy gate every case |
| P1 | Raw CUDA graph replay | **Implemented** | Observed 4.25–6.12x | Low | Static shape/mask regime; preserves eager kernels and numerics |
| P1 | Eliminate all-true masks outside hot path | **Implemented** | Large observed absolute latency reduction | Low | Dispatch occurs in case generation; no `.all().item()` synchronization |
| P1 | Hoist static masks / remove invalid-query dead work | **Implemented** | Causal+padded candidate 0.675→0.568 ms | Low | Safe because masks exclude invalid keys and other ops are token-local |
| P2 | Fused residual + LayerNorm + linear | No | 5–15% | Medium | Preserve FP32 normalization; compounds with QKV/FFN |
| P2 | Fused LayerNorm + FFN and exact-GELU epilogue | No | 5–20% | Medium–high | FFN is 64% of FLOPs; tanh GELU may fail tolerance |
| P2 | Pack valid tokens / variable-length execution | No | 1.2–2x with substantial padding | Medium–high | Pack whole block, not only attention; prefix-valid mask is favorable |
| P2 | Per-shape autotuned dispatcher | Partial | 5–30% | Medium–high | Known shape has mask-aware layer counts; official matrix still required |
| P3 | FP8 Transformer Engine/torchao path | No | 1.1–1.7x on large aligned GEMMs | High | Accuracy/scaling overhead; separate opt-in experiment |
| P3 | Hopper TMA/WGMMA persistent kernels | No | 5–30% where libraries underfill | High | Only after `ncu` proves a specific library kernel inefficient |
| P4 | Whole-block persistent megakernel | No | Shape-dependent | Very high | Alternative to library/Inductor scheduling, not an early additive step |

Strongly compounding groups are packed QKV + BSHD + Flash attention; token
packing + variable-length attention + FFN skipping; and residual/LayerNorm +
linear epilogues. FlashAttention, cuDNN SDPA, and Transformer Engine attention
are competing backends and must be A/B tested rather than stacked. Approximate
GELU and FP8 conflict most directly with the strict numerical gate.

## Later approximation research

Approximation work should follow measured bottlenecks and use the final model
output—not isolated operator error—as its gate. Promising experiments include:

1. Layerwise sensitivity sweeps: approximate one layer at a time, measure error
   amplification, then combine only low-sensitivity layers.
2. Mixed-precision islands: retain normalization and sensitive late layers in
   higher precision while downcasting only attention or FFN work with margin.
3. Fast approximation plus correction: compute a cheap polynomial/quantized
   result and apply a small residual correction only in ranges that dominate
   the final error.
4. Long-sequence softmax sparsification: omit provably negligible probability
   mass by tile, with an explicit bound and dense fallback near the threshold.
5. Shape-specific low-rank or structured FFN experiments only if the disclosed
   fixed weights/shapes make setup cost amortizable and the output gate passes.

The first broad approximation test, tanh GELU in all six FP16 layers, failed 74
of 13,107,200 elements. This rules out indiscriminate replacement but motivates
a later per-layer sensitivity sweep rather than abandoning approximation.

## Next profiling and implementation steps

1. Recover the official test matrix and benchmark every case.
2. With counter permissions enabled, capture one representative D×D GEMM, D×F
   GEMM, LayerNorm, exact GELU, mask kernel, and fused SDPA kernel using `ncu`'s
   roofline set.
3. Prototype residual-add + LayerNorm fusion, because LayerNorm and add
   launches are now a larger fraction after attention/QKV fusion.
4. Evaluate variable-length whole-block packing when padded official cases exist.

## Primary sources

- [NVIDIA H200 specifications](https://www.nvidia.com/en-us/data-center/h200/)
- [NVIDIA Hopper architecture and structured sparsity](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [PyTorch scaled dot product attention](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [PyTorch compile modes](https://docs.pytorch.org/docs/stable/generated/torch.compile)
- [cuDNN attention layouts, masks, and sequence lengths](https://docs.nvidia.com/deeplearning/cudnn/latest/operations/Attention.html)
- [FlashAttention implementations and packed/varlen APIs](https://github.com/Dao-AILab/flash-attention)
- [FlashAttention-3 Hopper paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/7ede97c3e082c6df10a8d6103a2eebd2-Paper-Conference.pdf)
- [NVIDIA Transformer Engine fused modules](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/getting_started/index.html)
- [cuBLASLt epilogues](https://docs.nvidia.com/cuda/cublas/index.html)
- [Triton fused LayerNorm](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)
- [Triton fused attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
- [Triton Hopper persistent matmul](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html)
- [GPU MODE lectures](https://github.com/gpu-mode/lectures)
- [Nsight Systems guide](https://docs.nvidia.com/nsight-systems/UserGuide/)
- [Nsight Compute counter-permission remediation](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters)
