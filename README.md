# H200 Transformer Kernel Optimization

This repository A/B tests an explicit PyTorch Transformer reference against
opt-in optimized implementations on one NVIDIA H200 NVL. The best pure-PyTorch
FP32/TF32 path packs Q/K/V, uses fused SDPA, and compiles into a CUDA graph; the
optional fixed-shape TensorRT graph is faster. The best FP16 path uses custom
exact Triton attention/normalization kernels, pretransposed FFN-output weights,
and a CUDA graph with its dynamic input copy captured as a retargetable node.

## Setup

The measured environment uses Python 3.12, PyTorch 2.13.0+cu130, CUDA 13.0,
and an NVIDIA H200 NVL.

```bash
python3 -m venv .venv
.venv/bin/pip install numpy
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu130
```

On Debian, install `python3.12-venv` if `venv` cannot bootstrap pip, and
`python3.12-dev` if `torch.compile` reports a missing `Python.h`.

The optional TensorRT experiment has a separate dependency set because its
CUDA 13 wheel is approximately 3.8 GB:

```bash
.venv/bin/pip install -r requirements-tensorrt.txt
```

## Reproduce the current result

Baseline/no-op A/B:

```bash
.venv/bin/python torch_transformer_benchmark.py \
  --device cuda:0 --dtype float16 \
  --accuracy-trials 25 --warmup 20 \
  --repeats 100 --benchmark-rounds 5
```

Current best verified experiment for the script's default FP32 shape:

```bash
.venv/bin/python torch_transformer_benchmark.py \
  --device cuda:0 --dtype float32 \
  --user-implementation sdpa-packed-qkv \
  --compile-user --compile-mode reduce-overhead \
  --accuracy-trials 25 --warmup 20 \
  --repeats 100 --benchmark-rounds 5
```

Current best verified FP16 experiment:

```bash
.venv/bin/python torch_transformer_benchmark.py \
  --device cuda:0 --dtype float16 \
  --user-implementation packed-qkv \
  --triton-rounded-attention \
  --triton-exact-add-norm \
  --triton-exact-initial-norm \
  --triton-linear-gelu \
  --triton-poly11-gelu-layer-indices 5 \
  --pretranspose-ffn-output-weights \
  --cuda-graph-user --cuda-graph-static-output \
  --cuda-graph-integrated-input-copy \
  --accuracy-trials 25 --warmup 20 \
  --repeats 100 --benchmark-rounds 5
```

If the caller consumes an output before the next invocation and does not retain
it, add `--cuda-graph-static-output` to return graph-owned storage directly.
This removes the final clone, but the next call overwrites the previous output.

`--user-implementation` defaults to `baseline`; all optimizations are opt-in.
`--sdpa-layers auto` selects all layers for FP32, four for the known default
noncausal FP16 shape, three for padded noncausal FP16, two for causal FP16, and
one for undisclosed FP16 shapes. BF16 uses packed QKV with the original attention
math and is bit-exact on the tested cases. Explicit values and `all` override
auto; `--sdpa-layer-indices 1,3,5` can override placement for experiments.
`--gelu-approx-layer-indices 1,3,5` similarly exposes the cheaper tanh GELU
for sensitivity experiments only; every individual layer failed the default
FP16 25-trial accuracy screen, so the verified best commands leave it empty.

Compilation is recommended only for accuracy-tested FP32 cases. It failed the
strict FP16 numerical gate.

`--triton-rounded-attention` selects a fixed-shape Triton kernel in all six FP16
layers of the known default model. It preserves the reference path's two FP16
score-rounding boundaries, matches PyTorch's lane-local/XOR warp softmax sum,
uses CUDA libdevice exponentiation and correctly rounded division, then rounds
the probabilities to FP16 before the value dot product. The six-layer setting
is bit-exact over 100 trials each for unmasked, padded, causal, and
causal+padded cases, and over 25-trial input-scale checks at 0.1 and 10. Use
`--triton-rounded-attention-layer-indices 2,3,4,5` only for explicit research;
the shorthand deliberately rejects other shapes and dtypes.

`--cuda-graph-user` captures the unchanged eager kernels, copies each new input
into graph-owned storage, submits the model with one graph launch, and clones
the output so later calls cannot mutate earlier results. It preserves FP16 and
BF16 numerics and is the recommended general launch-overhead optimization.
Shapes and the presence/absence of a padding mask are static for each capture.
It is mutually exclusive with `--compile-user`.

`--cuda-graph-integrated-input-copy` captures dynamic input and optional
validity-mask copies as graph nodes and retargets them to each caller tensor
before replay. Nodes are matched by their captured destination pointers, not
enumeration order. This removes separate GPU submissions without assuming
fixed caller pointers. Six fresh-process unmasked pairs improved the aggregate
median from 0.28455 to 0.28290 ms (1.0058x), and all six pairs won. A 12-round
padded A/B improved 0.30778 to 0.30658 ms (1.0039x). Inputs and masks must retain
the captured shape, dtype, device, and contiguous strides.

`--triton-exact-add-norm` fuses every residual addition with its following
LayerNorm. The kernel reproduces PyTorch CUDA's four-values-per-thread online
Welford calculation and exact four-warp reduction tree; both the residual sum
and normalized output are bit-exact. It compounds with rounded attention and
passed 100 trials in every mask regime. Use
`--triton-fused-add-norm-sites 8,9,10,11` only for explicit site experiments.

`--triton-exact-initial-norm` applies the same PyTorch-compatible Welford tree
to the one initial width-512 LayerNorm that cannot be folded into a preceding
residual addition. It is bit-exact in isolation and across the full unmasked
100-trial audit; the complete path also passed 100 trials in every mask regime.
Six order-balanced pairs improved the prior aggregate median from 0.28615 to
0.28530 ms (1.0030x) before the integrated-input graph optimization.

`--triton-linear-gelu` replaces each width-512-to-2048 FFN input projection and
following exact GELU with one tuned Triton kernel. It retains the intervening
FP16 materialization and CUDA libdevice `erf`. The all-layer path had zero
failed elements over 100 trials in each mask regime; unmasked output was
bit-exact, while masked/causal maximum absolute difference was 0.00390625 and
remained inside the combined tolerance. Use
`--triton-linear-gelu-layer-indices 4,5` for explicit placement or other-row-
count experiments; the shorthand is gated to the audited default model.

`--triton-poly11-gelu-layer-indices 5` replaces libdevice `erf` only in the
least-sensitive final FFN with CUTLASS's clipped degree-11 standard-normal-CDF
polynomial. A layerwise screen rejected earlier layers. The retained final-layer
setting passed 100 trials in every mask regime (209,715,200 outputs total), plus
25-trial checks at input scales 0.1 and 10 and padding ratios 0.1 and 0.75. A
14-round shared-weight alternating A/B improved 0.28169 to 0.27859 ms (1.0111x),
with every round favoring the polynomial. Approximating layers 4 and 5 was
faster but is deliberately rejected: it failed 79 of 13,107,200 outputs at
input scale 0.1.

`--pretranspose-ffn-output-weights` packs the six FFN-output weights once as
contiguous `[K, N]` operands and dispatches them through `torch.addmm`. On H200
this selects an NVJet `NNT` tactic instead of `TNT`. QKV and attention-output
weights deliberately retain their original layout because adjacent traces found
no benefit. Six order-balanced isolated process pairs improved the prior graph
from a 0.28675-ms aggregate median to 0.28590 ms (1.003x), with all six pairs
winning. The flag is bit-identical to the prior optimized graph over the tested
inputs and passed the full 100-trial audit in all four mask regimes. Packing
happens after device/dtype transfer and outside accuracy and timing regions.

For padded cases, the optimized path precomputes static causal masks, negates
the padding mask only when a reference-attention fallback needs it, and excludes
invalid keys in every attention. With exact add/LayerNorm enabled, its final
fusion writes invalid query rows directly as zero, avoiding a separate mask
kernel.

## TensorRT FP32 experiment

TensorRT FP16 fails the strict elementwise accuracy gate, but a fixed-shape
FP32/TF32 engine passes. The standalone script exports the exact seeded model
to ONNX, builds or reuses a content-addressed engine under `.tensorrt-cache/`,
captures TensorRT execution in a CUDA graph, and applies the same accuracy and
timing harness:

```bash
.venv/bin/python tensorrt_transformer_benchmark.py \
  --accuracy-trials 25 --warmup 20 \
  --repeats 200 --benchmark-rounds 7
```

Add `--rebuild` to discard the cached engine choice. `--static-output` removes
the safe output clone under the same overwrite-on-next-call contract as the
PyTorch graph option. The current TensorRT script is intentionally restricted
to the default unmasked FP32 shape.

## Clean profiling

The profiling mode warms up normally and exposes exactly one forward between
CUDA profiler start/stop calls:

```bash
source /export/home/alien/software/nvhpc/setup.sh
nsys profile --trace=cuda,nvtx,cublas \
  --cuda-graph-trace=node \
  --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output=profile_baseline \
  .venv/bin/python torch_transformer_benchmark.py \
    --device cuda:0 --dtype float16 \
    --profile-model baseline --warmup 20
```

See [OPTIMIZATION_REPORT.md](OPTIMIZATION_REPORT.md) for the roofline model,
profiles, experiment ledger, research sources, and prioritized next work.
