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
| Previous FP16 path + packed prefix QKV + raw CUDA graph | pass, 0/13,107,200 | 2.389 ms | 0.391 ms | **6.116x** | Previous FP16 best |
| Previous row + graph-owned static output | pass, 0/13,107,200 | 2.440 ms | 0.3900 ms | **6.258x** | Keep when output lifetime permits |
| Exact score-rounded Triton attention, all 6 + static graph output | exact, 0/209,715,200 across four mask regimes | 1.875 ms | **0.3237 ms** | **5.790x** | Previous robust FP16 best |
| Previous row + exact residual-add/LayerNorm at all 12 sites | exact, 0/209,715,200 across four mask regimes | 2.468 ms | **0.2923 ms** | **8.444x** | Previous exact FP16 best |
| Previous row + fused FFN input linear/exact-GELU, all 6 | pass, 0/209,715,200; unmasked exact | 2.401 ms | **0.2865 ms** | **8.383x** | Previous robust FP16 best |
| Previous row + pretransposed FFN-output weights | pass, 0/209,715,200; unmasked exact | 0.28675 ms prior graph | **0.28590 ms** | **1.003x** | Keep |
| Previous row + exact initial LayerNorm | pass, 0/209,715,200; unmasked exact | 0.28615 ms paired prior graph | **0.28530 ms** | **1.003x** | Keep |
| Previous row + integrated graph input-copy node | exact, 0/52,428,800 unmasked | 0.28455 ms paired prior graph | **0.28290 ms** | **1.006x** | Current robust FP16 best |
| Padded FP16, three SDPA layers + mask cleanup + raw graph | pass, 0/13,107,200 | 2.612 ms | 0.469 ms | **5.575x** | Previous padded best |
| Padded FP16, previous path + fused linear/exact-GELU | pass, 0/52,428,800 | 3.325 ms | **0.3040 ms** | **10.939x** | Current padded best |
| Padded FP16, integrated input+mask graph nodes | exact versus prior path, 0/5,242,880 | 0.30778 ms paired prior graph | **0.30658 ms** | **1.004x** | Keep |
| Causal FP16, two SDPA layers + raw graph | pass, 0/13,107,200 | 2.295 ms | 0.477 ms | **4.809x** | Keep |
| Causal FP16, previous path + fused linear/exact-GELU | pass, 0/52,428,800 | 2.968 ms | **0.2981 ms** | **9.957x** | Current causal best |
| Causal+padded FP16, two SDPA layers + raw graph | pass, 0/13,107,200 | 2.879 ms | 0.528 ms | **5.451x** | Previous causal+padded best |
| Causal+padded FP16, previous path + fused linear/exact-GELU | pass, 0/52,428,800 | 3.843 ms | **0.3443 ms** | **11.164x** | Current causal+padded best |
| BF16 packed QKV + reference attention + raw graph | exact, 0/13,107,200 | 2.396 ms | 0.492 ms | **4.871x** | Current BF16 best |
| Causal+padded BF16, same path | exact, 0/5,242,880 | 3.790 ms | 0.603 ms | **6.286x** | Keep |
| FP32 packed QKV + all-layer SDPA | pass, 0/5,242,880 | 2.296 ms | 1.185 ms | 1.938x | Keep |
| FP32 previous row + raw CUDA graph | pass, 0/5,242,880 | 2.389 ms | 0.562 ms | 4.248x | Safer/no compiler startup |
| FP32 previous row + `reduce-overhead` compile | pass, 0/13,107,200 | 2.302 ms | 0.549 ms | **4.197x** | Previous default-dtype best |
| TensorRT 11.2 FP32 engine + CUDA graph, safe I/O | pass, 0/13,107,200 | 2.282 ms | **0.5127 ms** | **4.451x** | Current default-dtype best |
| TensorRT 11.2 FP32 engine + graph, static output | pass, 0/2,621,440 | 2.312 ms | **0.5097 ms** | **4.537x** | Keep when output lifetime permits |
| TensorRT 11.2 FP16 engine | fail, 61,494/13,107,200 | — | — | — | Reject for FP16 |
| FP32 previous row + `max-autotune` | pass, 0/5,242,880 | 1.921 ms | 0.525 ms | 3.659x | Slightly faster candidate, less accuracy headroom |
| FP32 with FP16 attention core | pass | 2.305 ms | 0.549 ms compiled | 4.200x | Reject: no compiled gain, slower eager |
| FP16 tanh-approximate GELU | fail, 74/13,107,200 | — | — | — | Reject: exceeds elementwise tolerance |
| FP16 tanh GELU, one layer at a time | every layer fails | — | ~0.386 ms graphed | no measurable gain | Reject; retain opt-in sensitivity control |
| Exact Triton residual-add + LayerNorm, all sites | exact, 0/209,715,200 | 0.3233 ms prior path | **0.2933 ms** | **1.102x** | Keep and compound with exact attention |
| Force PyTorch FlashAttention instead of cuDNN SDPA | fail, 1/13,107,200 | — | — | — | Reject for default FP16 |

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

The new opt-in `--triton-rounded-attention` supersedes SDPA for the known
default FP16 target. Its specialized `S=128`, head-dimension-64 kernel tiles 16
queries against all keys for dense/causal inputs and 64 for padded inputs,
rounds the QK matmul result to FP16, and rounds the
scale multiply to FP16 again. Its FP32 softmax reproduces PyTorch
`PersistentSoftmax.cuh`: four lane-local values are summed sequentially, the 32
lane totals follow the XOR reduction tree, CUDA libdevice supplies `exp`, and
`tl.div_rn` supplies correctly rounded division. Probabilities are then rounded
to FP16 before the value dot product. These are the reference materialization
boundaries and arithmetic order, so all six layers are bit-exact. The final
kernel passed 100 trials in every mask regime (52,428,800 outputs per regime,
209,715,200 total), plus 25 trials at input scales 0.1 and 10. In contrast, a
stronger audit found that the previous four-layer cuDNN path fails seven
elements beyond its original 25-seed window. The shorthand is gated to the
fully tested model and FP16 dtype; explicit indices retain research control.

Compiled FP32 `reduce-overhead` also passed tested small, default,
causal+padded, and long-sequence shapes. Observed speedups were respectively
6.36x, 4.20x, 6.99x, and 1.10x, confirming that CUDA graph/launch fusion is most
valuable for launch-bound cases and modest for large compute-bound work.

TensorRT 11.2.1 was installed through its CUDA 13 Python wheel and tested
through the native builder/runtime rather than Torch-TensorRT. The exact seeded
baseline exports to a fixed-shape ONNX graph; TensorRT parses 417 layers and
builds a content-addressed engine. The FP32 engine is about 76.1 MB, passes 25
trials with maximum absolute error 0.000719, and takes 0.5127 ms when CUDA graph
replay includes the dynamic input copy and safe output clone. This is about
1.071x faster than the adjacent 0.5489 ms compiled PyTorch result. Static output
reduces it to 0.5097 ms. Native, uncaptured TensorRT measured about 0.595 ms,
showing that graph launch remains important even after TensorRT fusion.

The corresponding strongly typed FP16 engine is about 38.3 MB but fails 61,494
of 13,107,200 outputs, so it is rejected rather than benchmark-promoted. The
standalone `tensorrt_transformer_benchmark.py` currently covers only the known
unmasked FP32 shape; engine build artifacts are hardware/version-specific and
remain ignored under `.tensorrt-cache/`.

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

The safe graph wrapper clones its static output because a later replay reuses
and overwrites graph-owned storage. `--cuda-graph-static-output` makes that
lifetime contract explicit and returns the storage directly. It passed 25
default FP16 trials and ten causal+padded trials each for FP16 and bit-exact
BF16. In adjacent default-shape runs it reduced median latency from 0.3909 to
0.3900 ms (about 0.2%); the benefit is small but arithmetic is unchanged.

For padded cases, a causal triangle is created once during optimized-weight
setup rather than replayed in every reference-attention layer. Intermediate
invalid query rows are not zeroed: LayerNorm/FFN never mix tokens and every
attention excludes invalid keys. When the last residual/LayerNorm site is
fused, that kernel writes invalid rows as zero directly.
The best all-custom path also consumes the positive validity mask everywhere,
so it never computes the inverse mask. This removes both the final mask kernel
and the otherwise-dead bitwise-not kernel while leaving valid outputs unchanged.

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

The final exact rounded-Triton default trace has 61 graph nodes and 298.467
microseconds of summed GPU kernel time. Its six custom attention launches total
38.976 microseconds. Relative to the previous 79-node, 342.5-microsecond graph,
moving all six layers to the custom kernel removes 18 nodes and 44.0
microseconds (12.9%) of GPU work while becoming bit-exact. The measured
end-to-end latency is 0.3237 ms with graph-owned output and 0.3303 ms with a safe
output clone. Matching cuDNN's BSHD-backed output strides was essential: the
first prototype forced transpose copies and erased the kernel benefit. A sweep
over query tiles, warps, and pipeline stages selected 16 rows, four warps, and
three stages for dense/causal inputs. Padded inputs retain the audited 64-row,
four-warp, one-stage geometry because the faster tile changes eager equivalence.

The compounded exact add/LayerNorm trace has **49 graph nodes and 268.039
microseconds** of summed GPU kernel time. Twelve fused kernels take 38.593
microseconds, replacing twelve adds plus twelve PyTorch LayerNorms that took
about 67.7 microseconds. This removes 12 nodes and 30.428 microseconds (10.2%)
from the exact-attention trace. A shared-weight, same-input, interleaved graph
A/B measured 0.3233 ms before fusion and 0.2933 ms after fusion, also a
**1.102x incremental speedup**. The normal safe-output path measures 0.2942 ms;
graph-owned static output measures 0.2923 ms.

Fusing each FFN input projection with its exact-GELU consumer reduces the
unmasked graph again, from 49 to **43 nodes** and from 268.039 to 265.409
microseconds of summed kernel time. Six tuned 128x128x64, eight-warp,
four-stage kernels take 64.896 microseconds; the six prior library GEMMs plus
six standalone GELUs took 69.921 microseconds. A shared-weight, same-input
interleaved graph A/B measured 0.2929 ms before and 0.2869 ms after fusion
(**1.021x incremental**). Static output measures 0.2865 ms and safe output
0.2898 ms. The saved node launches explain why end-to-end improvement slightly
exceeds the 5.0-microsecond kernel-duration reduction.

Pretransposing only the six FFN-output weights once changes their NVJet tactic
from `TNT` to `NNT`, without changing the 43-node graph. QKV and
attention-output weights retain their original layouts: adjacent traces showed
that broad pretransposition was neutral or slightly negative at those sites.
That trace sums to **264.714 microseconds**, with 47.938 microseconds in the
new FFN-output tactic and 182.984 microseconds across every GEMM/GELU group.
Six order-balanced isolated-process A/B pairs all favored the packed FFN
layout, with aggregate medians of 0.28675 versus 0.28590 ms (**1.003x
incremental**). Prepacking is outside every accuracy and timing region. The
complete path passed 100 trials in each of the four mask regimes (209,715,200
outputs, zero failed elements), plus 25 trials each at input scales 0.1/10 and
padding ratios 0.1/0.75. Unmasked output is bit-exact; the masked audit retains
the previous 0.00390625 maximum absolute difference.

The remaining standalone initial LayerNorm uses the same width-512 Welford
tree as the exact fused add/LayerNorm kernel. Its isolated latency fell from
4.777 to 3.918 microseconds and was bit-exact. Six order-balanced process pairs
improved the complete graph from a 0.28615-ms aggregate median to 0.28530 ms
(**1.0030x incremental**), winning five of six pairs. The full path then passed
100 trials in all four mask regimes (209,715,200 outputs and zero tolerance
failures); the unmasked regime remained bit-exact.

The next exact scheduling change captures the unmasked input D2D copy as the
first CUDA graph node and uses `cudaGraphExecMemcpyNodeSetParams1D` to retarget
its source pointer on every call. This preserves changing input tensors while
replacing a separate copy submission plus replay with one 44-node graph replay
(one memcpy and 43 kernels). Six fresh-process, order-balanced pairs all won,
moving the aggregate median from 0.28455 to **0.28290 ms** (**1.0058x
incremental**). A 100-trial audit was bit-exact over 52,428,800 outputs. The
clean trace reports 261.859 microseconds of kernels and a 3.008-microsecond
input copy; the remaining end-to-end gap is about 18 microseconds of graph
scheduling and measurement overhead.

The generalized capture inspects each memcpy node's CUDA parameters and matches
it by captured destination pointer, allowing the validity-mask copy to be
retargeted without relying on node enumeration order. A shared-weight,
12-round padded A/B won every round and improved 0.30778 to 0.30658 ms
(**1.0039x incremental**). Both padded and causal+padded modes then passed 100
trials (104,857,600 outputs total and zero tolerance failures); their maximum
absolute difference from the baseline remained 0.00390625. The opt-in path now
supports both masked and unmasked contiguous inputs.

The corresponding padded graph also has **49 nodes** and 281.413 microseconds
of summed kernel time. Its extra attention cost comes from padding predicates,
but it contains neither mask negation nor final masked-fill. Folding final
zeroing into add/LayerNorm and deleting the dead negation improves the measured
padded candidate from 0.3145 to 0.3086 ms (1.9%) and causal+padded from 0.3556
to 0.3509 ms (1.3%).

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

At 0.2829 ms, the current Triton FP16 path sustains an effective 142.3 TFLOP/s
on the 40.265-GFLOP logical model, about 17.0% of the estimated 835.5 TFLOP/s
dense Tensor Core roof. It is 5.87x above the compute-only floor and 2.38x above
the logical-traffic floor. The remaining gap is consistent with the
trace: medium GEMMs, reductions, copies, GELU, and serial dependencies dominate
rather than the 39.0-microsecond total for six exact custom attention kernels.
TensorRT FP32 at 0.5127 ms
is 78.5 effective TFLOP/s; this is not directly comparable to the FP16 roof
because its engine uses FP32/TF32 tactics and doubles weight/activation bytes.

### Throughput and bottleneck comparison

The requested conventional-versus-theoretical comparison is:

| Execution point | Latency | Effective throughput | Dense compute roof achieved | Logical traffic rate | HBM roof fraction |
|---|---:|---:|---:|---:|---:|
| Conventional eager PyTorch FP16 | 2.401 ms | 16.8 TFLOP/s | 2.0% | 0.238 TB/s | 5.0% |
| Earlier library-kernel CUDA graph (legacy screen) | 0.3900 ms | 103.2 TFLOP/s | 12.4% | 1.463 TB/s | 30.5% |
| Current exact Triton/library CUDA graph | **0.2829 ms** | **142.3 TFLOP/s** | **17.0%** | **2.017 TB/s** | **42.0%** |
| Current trace, summed active GPU operations | 0.264867 ms | 152.0 TFLOP/s | 18.2% | 2.154 TB/s | 44.9% |
| H200 dense Tensor Core compute roof | 0.0482 ms | 835.5 TFLOP/s | 100% | — | — |

Thus the retained path delivers **8.49x conventional eager throughput**, a
**749% throughput increase** and **88.2% latency reduction**, while reaching
17.0% of the raw dense compute roof. If the 570.5-MB logical traffic estimate
were all served by HBM, the shape-aware bandwidth roof would instead be about
338.6 TFLOP/s (118.9 microseconds), of which the current path reaches 42.0%.
The traffic columns are an algorithmic proxy, not an HBM-counter measurement:
fusion reduces intermediate traffic and caches can serve weights or activations.
`ncu` counters are needed to replace them with achieved DRAM and Tensor Core
rates.

The current 44-node graph (43 kernels plus its input copy) identifies the
remaining serial critical path:

| Kernel group | Summed time | GPU-kernel share |
|---|---:|---:|
| FFN input projections + exact GELU | 65.410 us | 25.0% |
| FFN output projections | 47.169 us | 18.0% |
| Packed QKV projections | 41.536 us | 15.9% |
| Attention output projections | 27.136 us | 10.4% |
| Exact attention | 38.944 us | 14.9% |
| Residual + LayerNorm | 38.496 us | 14.7% |
| Initial LayerNorm | 3.168 us | 1.2% |
| Integrated input D2D copy | 3.008 us | separate graph memcpy node |

All GEMM-containing kernel groups total **181.251 microseconds (69.2%)**; the
two FFN projections alone total **112.579 microseconds (43.0%)**. The
bottleneck is therefore the sequence of medium, dependency-bound projection
GEMMs—not HBM bandwidth or attention alone. Each layer must finish
normalization before its next projection, limiting occupancy across the whole
call even though each individual GEMM uses Tensor Cores.

## Prioritized optimization backlog

| Priority | Work | Present? | Expected end-to-end gain | Difficulty | Dependencies / conflicts |
|---|---|---:|---:|---:|---|
| P0 | Recover official shapes and dtype/mask matrix | No | Enables reliable dispatch | Low/admin | Prerequisite for every specialization |
| P0 | Accuracy/backend gate attention per case | Exact rounded Triton implemented for all layers of known FP16 target | 1.05–2x, sequence-dependent | Low–medium | BF16 remains on reference math |
| P1 | Packed QKV projection | **Implemented** | Observed incremental gain; usually 5–15% | Medium | Compounds with SDPA and BSHD layout |
| P1 | Pretranspose FFN-output operands | **Implemented** | Observed 0.3%; exact tactic change | Low | Compounds with packed QKV and CUDA graph |
| P1 | Compile/CUDA graph/TensorRT sweep | **TensorRT FP32 + graph implemented; FP16 rejected** | Observed 1.10–6.99x | Low–medium | Static shapes; accuracy gate every engine |
| P1 | Raw CUDA graph replay | **Implemented** | Observed 4.25–6.12x | Low | Static shape/mask regime; preserves eager kernels and numerics |
| P1 | Eliminate all-true masks outside hot path | **Implemented** | Large observed absolute latency reduction | Low | Dispatch occurs in case generation; no `.all().item()` synchronization |
| P1 | Hoist static masks / remove invalid-query dead work | **Implemented** | Causal+padded candidate 0.675→0.568 ms | Low | Safe because masks exclude invalid keys and other ops are token-local |
| P2 | Fused residual + LayerNorm + linear | **Exact Triton add+norm implemented** | Observed 10.2%; linear fusion may add more | Medium | Width-512 specialization; linear fusion remains open |
| P2 | Fused LayerNorm + FFN and exact-GELU epilogue | Linear+exact-GELU implemented; output composite prototype rejected | Observed 2.1% | Medium–high | FFN is 64% of FLOPs; preserve FP16 boundaries |
| P2 | Pack valid tokens / variable-length execution | No | 1.2–2x with substantial padding | Medium–high | Pack whole block, not only attention; prefix-valid mask is favorable |
| P2 | Per-shape autotuned dispatcher | Partial | 5–30% | Medium–high | Known shape has mask-aware layer counts; official matrix still required |
| P3 | FP8 Transformer Engine/torchao path | No | 1.1–1.7x on large aligned GEMMs | High | Accuracy/scaling overhead; separate opt-in experiment |
| P3 | Hopper TMA/WGMMA persistent kernels | TMA input-FFN prototype rejected | 5–30% where libraries underfill | High | Revisit only after `ncu` proves a specific library kernel inefficient |
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
of 13,107,200 elements. The follow-up per-layer sweep also rejected every
layer: layers zero through five respectively failed 39, 13, 2, 2, 1, and 1
elements over 13,107,200 outputs. Sensitivity falls strongly toward the output,
but the strict gate allows no failures. Graph-replayed latency was about 0.386
ms for both one approximate layer and all six, so the cheaper formula produced
no measurable end-to-end gain on this shape. The opt-in
`--gelu-approx-layer-indices` control remains only to reproduce sensitivity
experiments on future shapes; automatic/default dispatch never enables it.

An FP8 follow-up quantized only one FFN input projection at a time with Hopper
E4M3 scaled GEMM. Even the least-sensitive final layer failed 1,028,312 of
5,242,880 outputs over ten trials with per-tensor scales. Row-wise activation
and output-channel weight scales still failed 1,023,841 elements and added
reduction/quantization work. This is far outside the strict accuracy envelope,
so FP8 remains rejected rather than hidden behind the production dispatcher.

A later finite-domain experiment observed that FP16 GELU has only 65,536 input
bit patterns. It generated the exact PyTorch output for every pattern once and
replaced runtime `erf` with a 128-KiB Triton lookup table. The lookup was
bit-exact exhaustively (including NaN bit patterns), but random cached gathers
still measured 24.4 microseconds per 2,097,152-element activation versus 19.3
microseconds for PyTorch's vectorized exact GELU, so the code was removed. A
direct Triton libdevice-`erf` kernel was slower again at 27.9 microseconds and
differed on 151 finite FP16 inputs. Exact finite-domain lookup is therefore a
useful novel technique only when the table access is cheaper than the native
operation; it is not a win for this bandwidth-efficient CUDA GELU kernel.

The retained alternative fuses GELU into the producer GEMM rather than
replacing its mathematics. A 20-configuration launch sweep selected
128x128x64, eight warps, four stages. The kernel rounds the biased GEMM output
to FP16 before applying the same libdevice `erf` formula, preserving the
reference materialization boundary. It was operator-bit-exact on a full
2,097,152-element random test. At model level, unmasked output stayed bit-exact
over 100 trials; padded, causal, and combined modes each passed 100 trials with
zero failed elements and 0.00390625 maximum absolute difference. Input scales
0.1 and 10 and padding ratios 0.1, 0.5, and 0.75 also passed 25 trials.

A follow-up Hopper scheduling experiment replaced pointer loads with Triton TMA
descriptors and swept two pipeline depths beyond the baseline, four/eight
warps, and warp specialization. The best TMA result was about 19.8 microseconds
versus 19.4 microseconds for the retained pointer kernel, so TMA was removed.
The default input projection has 128 output tiles, approximately one wave on
this H200, leaving no persistent scheduling opportunity to amortize descriptor
or work-queue overhead.

A CUTLASS 3.x SM90 TMA/WGMMA prototype then kept the biased GEMM's FP16
materialization boundary and evaluated exact erf-GELU in a visitor epilogue.
The cooperative 128x128 tile was best at 11.38 microseconds; four ping-pong
64/128 tile combinations took 14.51--16.54 microseconds. The retained Triton
kernel takes about 10.89 microseconds in the graph. CUTLASS differed from the
PyTorch operator by at most 3.81e-6 in the isolated screen, but it was still
slower, so the prototype remains outside the repository.

The next trace-driven prototype fused the 2048-to-512 FFN output projection's
residual addition, followed by a separate exact LayerNorm over the materialized
sum. Its isolated outputs were bit-exact, but the pair took 18.91 microseconds
versus 11.96 microseconds for the NVIDIA library GEMM plus retained exact
add/LayerNorm. Saving one activation transfer could not offset the custom
GEMM's lower efficiency, so this code was also removed. A future output
epilogue should retain the library-quality GEMM, for example through a CUTLASS
visitor epilogue, rather than replace it with a generic Triton matmul.

The first Triton residual-add/LayerNorm prototype used a generic Welford
reduction and failed 36 elements in 52,428,800 outputs. Inspecting PyTorch's
CUDA implementation exposed the missing detail: width 512 launches four warps,
each thread consumes four consecutive features online, then moments follow a
specific shuffle-down and inter-warp tree. The replacement reproduces that
tree, reciprocal-multiply order, reciprocal square root, and affine order.
Residual sums and normalized outputs are bit-exact in isolated random tests;
all twelve model sites are bit-exact over 100 trials in all four mask regimes.
`--triton-exact-add-norm` enables every site, while
`--triton-fused-add-norm-sites` retains explicit placement control.

## Next profiling and implementation steps

1. Recover the official test matrix and benchmark every case.
2. With counter permissions enabled, capture one representative D×D GEMM, D×F
   GEMM, LayerNorm, exact GELU, mask kernel, and fused SDPA kernel using `ncu`'s
   roofline set.
3. Verify integrated argument-copy node matching across every official static
   shape once the missing matrix is recovered.
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
- [PyTorch CUDA LayerNorm implementation](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/cuda/layer_norm_kernel.cu)
- [PyTorch CUDA GELU implementation](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/cuda/ActivationGeluKernel.cu)
- [Triton fused attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
- [Triton Hopper persistent matmul](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html)
- [TensorRT Python installation](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/install-pip.html)
- [TensorRT performance benchmarking](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/benchmarking.html)
- [GPU MODE lectures](https://github.com/gpu-mode/lectures)
- [Nsight Systems guide](https://docs.nvidia.com/nsight-systems/UserGuide/)
- [Nsight Compute counter-permission remediation](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters)
