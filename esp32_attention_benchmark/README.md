# ESP32-C3 attention benchmark

This Arduino sketch measures scaled dot-product attention on the Seeed Studio
XIAO ESP32-C3. The primitive suite compares several schedules and numeric
formats using deterministic float32 inputs, including:

- `materialized_ref`: stable softmax with a full `N x N` score matrix.
- `tiled_online_exact`: block-online softmax without the score matrix.
- `tiled_online_fast_exp`: the same tiled schedule with a range-reduced
  polynomial exponential approximation.

All candidates are checked against the hackathon's exact acceptance rule:
absolute error <= 0.002 **or** relative error <= 0.02 for every output.

## Build and upload

```sh
arduino-cli compile --fqbn esp32:esp32:XIAO_ESP32C3 esp32_attention_benchmark
arduino-cli upload --fqbn esp32:esp32:XIAO_ESP32C3 \
  --port /dev/ttyACM0 esp32_attention_benchmark
```

Open the USB serial port at 115200 baud. Send `r` to repeat the suite. Output is
CSV-like so it can be captured and compared across firmware revisions and
boards.

Send `e` to run the complete attention-layer benchmark: Q/K/V projections,
four heads, padding and causal masks, mixed-precision tiled attention, and the
output projection. It compares float projections against int16 activations with
per-output-channel int8 projection weights. Timing includes activation
conversion, and offline weight quantization is excluded because weights are
stored in inference format. Validate every emitted output against the
independent host implementation with:

```sh
python3 tools/validate_e2e.py --port /dev/ttyACM0 \
  --capture results/esp32c3_end_to_end_v2.log
```

## Scope

The primitive suite remains useful for isolating kernels. The end-to-end suite
now covers the complete attention layer at batch 1, but layer normalization,
residuals, FFN, energy, and network communication remain separate milestones.
