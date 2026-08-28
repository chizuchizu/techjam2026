# Prior art and novelty boundary

The original idea asked whether attention on ESP32s or parallel attention on
microcontrollers had “never been” done. Current evidence does not support either
broad claim.

Relevant work includes:

- [Optimizing the Deployment of Tiny Transformers on Low-Power MCUs](https://arxiv.org/abs/2404.02945)
  introduces fused-weight self-attention and depth-first tiling that avoids
  materializing a complete attention map on ARM and RISC-V MCUs.
- [Distributed Inference with Minimal Off-Chip Traffic for Transformers on Low-Power MCUs](https://arxiv.org/abs/2412.04372)
  evaluates Transformer inference across up to eight low-power MCUs, including
  TinyLlama and MobileBERT.
- [Going Beyond the Edge: Distributed Inference of Transformer Models on Ultra-Low-Power Wireless Devices](https://arxiv.org/abs/2605.15694)
  presents CATS and evaluates distributed Transformer inference on up to sixteen
  wireless devices.
- [When the Edge Meets Transformers](https://iqua.ece.utoronto.ca/papers/chenghao-icdcs24.pdf)
  studies cross-device distributed Transformer inference and its communication
  constraints.
- The open-source [esp32-llm](https://github.com/jake-g/esp32-llm) project ports a
  Llama-style Transformer implementation to ESP32-S3.

Hardware facts used in this project come from the
[ESP32-C3 datasheet](https://documentation.espressif.com/esp32-c3_datasheet_en.html):
single-core 32-bit RISC-V up to 160 MHz and 400 KB SRAM, of which 16 KB is used
for cache. Espressif's
[performance guide](https://docs.espressif.com/projects/esp-idf/en/v5.1/esp32c3/api-guides/performance/speed.html)
also warns that ESP32-C3 floating-point calculations are software-emulated and
slow.

## Defensible positioning

Do not say “the first attention on an ESP32” or “the first parallel attention on
MCUs.” A narrower claim may become defensible after deeper review, for example:

> An open, reproducible study of exact online-softmax attention and its
> compute/communication crossover on a cluster of no-PSRAM XIAO ESP32-C3 boards.

Even that should be presented as the project's focus, not a world-first, until
the implementation and literature review are complete.

