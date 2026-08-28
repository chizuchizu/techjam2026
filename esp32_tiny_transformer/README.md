# Complete tiny Transformer on XIAO ESP32-C3

This sketch runs a trained, two-block causal character Transformer end to end.
It includes embeddings, positional embeddings, RMSNorm, four-head causal
self-attention, residual connections, ReLU feed-forward networks, final norm,
and a 24-token language-model head.

The model uses per-output-channel int8 linear weights, dynamic int16 linear
activations, and the accuracy-qualified int8 Q/K plus int16 V attention path.
Its context is 16 tokens, model width is 32, and FFN width is 64. It is a small
memorization model used to prove the complete inference pipeline; it is not a
general-purpose language model.

Build and upload:

```sh
arduino-cli compile --fqbn esp32:esp32:XIAO_ESP32C3 esp32_tiny_transformer
arduino-cli upload --fqbn esp32:esp32:XIAO_ESP32C3 \
  --port /dev/ttyACM0 esp32_tiny_transformer
```

Validate every output logit and 48 generated tokens against the independent
NumPy implementation:

```sh
.venv/bin/python tools/tiny_transformer_reference.py \
  --port /dev/ttyACM0 --capture results/esp32c3_tiny_transformer_v1.log
```

The generated weights in `tiny_transformer_weights.h` come from:

```sh
.venv/bin/python tools/train_tiny_transformer.py
```
