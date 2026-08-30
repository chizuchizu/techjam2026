# Case 8: two-ESP32 Ring Attention Transformer

This project runs the official case-8 geometry (`B=64, S=128, D=1024, H=4,
F=1024, L=4`, causal) across two Seeed XIAO ESP32-C3 boards. The boards own
the two 64-token sequence shards. The PC stores tensors and relays KV blocks
between COM ports, while LayerNorm, projections, exact online-softmax
attention, GELU, residuals, and final normalization execute on the boards.

The implementation does not allocate an `S x S` score matrix. Each query tile
stays on its worker while 8-token KV transport blocks traverse it; the worker updates
the online `(maximum, denominator, numerator)` state and normalizes once.
This is dense-equivalent Ring Attention, not sparse LongNet.

## Layout

- `export_case8.py` exports weights from the official PyTorch model as
  per-output-row int16 matrices and generates reference inputs/outputs.
- `run_case8_ring.py` stages one matrix at a time, drives both boards in
  parallel, relays KV blocks, validates the result, and records timings.
- `protocol.py` defines the CRC-protected versioned serial framing.
- `firmware/` contains the identical serial-only worker image for both boards.

The 4 MB board cannot retain the complete model. A dedicated 2.44 MB scratch
partition holds one `1024 x 1024` int16 matrix; it is reused across all batch
items before the next matrix is staged. Host checkpoints retain the output of
each completed Transformer layer.

## Build and run

PyTorch is needed only on the PC to export the official seed and reference:

```powershell
python benchmarks/case-08/ring_attention/export_case8.py `
  --reference-batch 1 --reference-trials 1
```

Build and flash the same firmware to each board:

```powershell
$env:PLATFORMIO_CORE_DIR = Join-Path (Get-Location) '.pio-core'
pio run --project-dir benchmarks/case-08/ring_attention/firmware `
  -e case8-ring-worker -t upload --upload-port COM10
pio run --project-dir benchmarks/case-08/ring_attention/firmware `
  -e case8-ring-worker -t upload --upload-port COM11
```

Validate Ring Attention before attempting a complete forward:

```powershell
python benchmarks/case-08/ring_attention/run_case8_ring.py `
  --serial-ports COM10,COM11 --attention-only
```

Then run the four-layer smoke test:

```powershell
python benchmarks/case-08/ring_attention/run_case8_ring.py `
  --serial-ports COM10,COM11 --batch 1
```

For the official batch, export a B=64 reference and run the resumable path:

```powershell
python benchmarks/case-08/ring_attention/export_case8.py `
  --reference-batch 64 --reference-trials 1
python benchmarks/case-08/ring_attention/run_case8_ring.py `
  --serial-ports COM10,COM11 --batch 64
```

Results are written under `results/`. A B=1 result is only a smoke test. A
case-8 claim requires `B=64`, and the competition's full accuracy claim needs
five complete B=64 trials using seeds `1234..1238`.

## Numeric and reliability contract

- Matrices use symmetric int16 quantization with a float32 scale per output
  row; activations are quantized per four-token tile on-device.
- GEMMs use int64 accumulators. Attention and normalization use float32.
- Every frame and staged matrix is CRC32 checked and matched by request ID.
- The official gate is applied exactly: absolute error `<= 0.002` **or**
  relative error `<= 0.02`, with finite values required.
the online `(maximum, denominator, numerator)` state and normalizes once.
