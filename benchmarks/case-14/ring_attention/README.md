# Two-ESP32 exact ring attention

This experiment implements the case-14 attention geometry (`head_dim=64`) as
streamed, exact causal attention over two ESP32 workers. Query tiles remain
stationary while disjoint KV tiles are scheduled round-robin. Workers return
unnormalized online-softmax state, which the coordinator merges before the one
final normalization. No `S x S` matrix is allocated.

The worker command is part of the shared cluster firmware in
`../../case-02/multiboard/esp32_cluster_transport`. Flash the same image onto
both boards using the `esp32-ring-serial` environment, then run:

```powershell
pio run --project-dir benchmarks/case-02/multiboard/esp32_cluster_transport `
  -e esp32-ring-serial -t upload --upload-port COM10
pio run --project-dir benchmarks/case-02/multiboard/esp32_cluster_transport `
  -e esp32-ring-serial -t upload --upload-port COM11
python benchmarks/case-14/ring_attention/run_two_esp32_ring.py `
  --serial-ports COM10,COM11 `
  --mode float32 --sequence 128 --head-dim 64 `
  --query-tile 32 --kv-tile 32
```

The quantized TCP transport remains available with `--workers IP1,IP2 --mode
quantized` when Wi-Fi is configured.

The protocol uses 32-bit global positions and can address `S=100000`, but a
complete case-14 dense forward remains quadratic and is not practically
finishable on two ESP32-C3 boards. The default run validates a bounded head;
its timings can be used to estimate the full dense workload honestly.

## Measured results

Both physical boards are Seeed XIAO ESP32-C3 devices at 160 MHz. Float32 mode
was checked with the stricter `atol=0.001 OR rtol=0.01` criterion.

| Shape | Boards | Wall time | Max abs error | Failed |
|---|---:|---:|---:|---:|
| `S=128, HD=64` | 1 | 3663.238 ms | 4.47e-7 | 0 / 8192 |
| `S=128, HD=64` | 2 | 2261.571 ms | 4.47e-7 | 0 / 8192 |
| `S=256, HD=64` | 2 | 7942.390 ms | 5.36e-7 | 0 / 16384 |

The measured two-board speedup at `S=128` is 1.62x. Extrapolating the `S=256`
quadratic measurement to all `B=32 x H=16 x L=2` case-14 head/layer tasks is
roughly 39 years on these two boards, before projections and FFNs. The streamed
protocol makes the memory footprint feasible; it does not remove dense
attention's quadratic arithmetic.
