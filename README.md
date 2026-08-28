# H200 Transformer Kernel Optimization

This repository A/B tests an explicit PyTorch Transformer reference against
opt-in optimized implementations on one NVIDIA H200 NVL. The current best
verified default-shape experiment packs Q/K/V into one projection for four
trailing layers and dispatches their attention through fused SDPA.

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

Current best verified experiment for the script's default shape:

```bash
.venv/bin/python torch_transformer_benchmark.py \
  --device cuda:0 --dtype float16 \
  --user-implementation sdpa-packed-qkv --sdpa-layers 4 \
  --accuracy-trials 25 --warmup 20 \
  --repeats 100 --benchmark-rounds 5
```

`--user-implementation` defaults to `baseline`; all optimizations are opt-in.
The conservative SDPA default is one trailing layer. Four layers passed the
known default noncausal FP16 case, but must not be assumed correct for undisclosed
competition shapes. BF16 currently falls back to the bit-identical baseline.

## Clean profiling

The profiling mode warms up normally and exposes exactly one forward between
CUDA profiler start/stop calls:

```bash
source /export/home/alien/software/nvhpc/setup.sh
nsys profile --trace=cuda,nvtx,cublas \
  --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output=profile_baseline \
  .venv/bin/python torch_transformer_benchmark.py \
    --device cuda:0 --dtype float16 \
    --profile-model baseline --warmup 20
```

See [OPTIMIZATION_REPORT.md](OPTIMIZATION_REPORT.md) for the roofline model,
profiles, experiment ledger, research sources, and prioritized next work.
