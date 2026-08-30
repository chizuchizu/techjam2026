# Intel Iris Xe SYCL attention prototype

This is an exact, causal, online-softmax attention microbenchmark for the Case
14 head geometry (`head_dim=64`). It never creates an `S x S` score matrix.
It intentionally benchmarks attention only; projections, residuals, layer
normalization, FFNs, padding masks, weight copying, and the batch loop still
need to be integrated into the official PyTorch harness.

The kernel stores FP32 online maximum, denominator, and weighted-value state.
Its validation uses the official element rule:

```text
abs(user - reference) <= 0.001 OR
abs(user - reference) <= 0.01 * abs(reference)
```

## Build on Windows

Intel oneAPI Toolkit 2026 is installed on the tested laptop. Build with:

```powershell
& benchmarks/case-14/sycl_attention/build.ps1
```

List devices and run a bounded correctness/performance check:

```powershell
& benchmarks/case-14/sycl_attention/run.ps1 --list-devices `
  --sequence 2048 --heads 1 --head-dim 64 --repeats 3
```

Scale one head before attempting all 16:

```powershell
& benchmarks/case-14/sycl_attention/run.ps1 `
  --sequence 8192 --heads 1 --head-dim 64 --validate-sequence 256
& benchmarks/case-14/sycl_attention/run.ps1 `
  --sequence 16384 --heads 1 --head-dim 64 --validate-sequence 256
```

Only after those runs are stable should the full attention geometry be tried:

```powershell
& benchmarks/case-14/sycl_attention/run.ps1 `
  --sequence 100000 --heads 16 --head-dim 64 `
  --validate-sequence 256 --batches 32 --repeats 1
```

That command represents one batch item and one attention layer. Its Q, K, V,
and output tensors require about 1.53 GiB total. Case 14 must loop over the 32
batch items and two layers; it must not allocate all batches simultaneously.
The `--batches 32` option performs those attention passes sequentially with a
separate input fill for each batch. It does not implement projections or the
layer-to-layer state update.

## Measured on this laptop

Device: Intel Iris Xe, OpenCL driver `31.0.101.4146`, FP32 storage and FP32
online accumulation.

| Shape | Median kernel time | Throughput | Accuracy |
|---|---:|---:|---:|
| `H=1, S=2,048, Dh=64` | 5.494 ms | 97.8 GFLOP/s | validation passed |
| `H=1, S=8,192, Dh=64` | 90.670 ms | 94.8 GFLOP/s | validation passed |
| `H=1, S=16,384, Dh=64` | 359.583 ms | 95.6 GFLOP/s | validation passed |
| `H=1, S=100,000, Dh=64` | 13.851 s | 92.4 GFLOP/s | validation passed |
| `H=16, S=512, Dh=64` | 5.149 ms | 104.5 GFLOP/s | `0/524,288` failed |
| `H=16, S=100,000, Dh=64` | 220.945 s | 92.7 GFLOP/s | bounded validation passed |

The full 16-head geometry was physically measured and consumed 1,562.5 MiB for
Q, K, V, and output. Repeating its 220.945-second kernel for `B=32, L=2`
projects to 3 hours 56 minutes. That is an attention-only estimate, not an
official full-Transformer timing.

The current kernel maps one 16-lane subgroup to each query. Sixteen query
subgroups share a 32-key K/V tile in 16 KiB of work-group local memory. It
stores a 16-by-32 score tile and merges block softmax state online.
