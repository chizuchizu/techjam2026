# H200 Transformer Kernel Optimization

This repository A/B tests an explicit PyTorch Transformer reference against
opt-in optimized implementations on one NVIDIA H200 NVL. The current best
default FP32/TF32 path packs Q/K/V, uses fused SDPA in all layers, and compiles
the model into a CUDA graph with fused pointwise/reduction kernels.

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
  --user-implementation sdpa-packed-qkv \
  --cuda-graph-user \
  --accuracy-trials 25 --warmup 20 \
  --repeats 100 --benchmark-rounds 5
```

`--user-implementation` defaults to `baseline`; all optimizations are opt-in.
`--sdpa-layers auto` selects all layers for FP32, four for the known default
noncausal FP16 shape, and one for undisclosed FP16 cases. BF16 uses packed QKV
with the original attention math and is bit-exact on the tested cases. Explicit
values and `all` override auto.

Compilation is recommended only for accuracy-tested FP32 cases. It failed the
strict FP16 numerical gate.

`--cuda-graph-user` captures the unchanged eager kernels, copies each new input
into graph-owned storage, submits the model with one graph launch, and clones
the output so later calls cannot mutate earlier results. It preserves FP16 and
BF16 numerics and is the recommended general launch-overhead optimization.
Shapes and the presence/absence of a padding mask are static for each capture.
It is mutually exclusive with `--compile-user`.

For padded cases, the optimized path precomputes static causal masks, negates
the padding mask once, excludes invalid keys in every attention, and zeroes
invalid query rows only once at final output.

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
