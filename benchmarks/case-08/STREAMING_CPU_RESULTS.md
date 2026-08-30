# Case 8 CPU streaming-attention experiment

Configuration: `B=64, D=1024, H=4, S=128, HD=256`, causal. This experiment
isolates the attention kernel; projections, FFNs, residuals and LayerNorm are
not included.

Run on the Intel Core i5-1340P laptop with NumPy/OpenBLAS:

```powershell
python benchmarks/case-08/streaming_cpu_attention.py `
  --repeats 5 `
  --tiles 16,32,64,16x64,32x64,64x32,64x16
```

All streamed configurations passed `abs <= 0.001 OR relative <= 0.01`. The
largest observed absolute difference from dense fp32 attention was `2.67e-6`.

| Algorithm | Q tile | KV tile | Median | Score tile | Relative speed |
|---|---:|---:|---:|---:|---:|
| Dense | 128 | 128 | 146.553 ms | 16 MiB | 1.00x |
| Streaming | 16 | 16 | 335.599 ms | 0.25 MiB | 0.44x |
| Streaming | 64 | 32 | 236.759 ms | 2 MiB | 0.62x |
| Streaming | 64 | 16 | 341.047 ms | 1 MiB | 0.43x |

The other tested streaming shapes were slower. At `S=128`, the dense score
matrix is only 16 MiB, so blocking saves little total memory relative to the
Q/K/V and output tensors while introducing additional BLAS calls, masks and
online-softmax merges. Dense attention is therefore the appropriate case-8
CPU path; streaming should be selected only for the long-sequence cases.
