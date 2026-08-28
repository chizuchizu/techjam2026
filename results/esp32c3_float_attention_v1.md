# ESP32-C3 float attention v1

- Board: Seeed Studio XIAO ESP32-C3, revision 0.4
- CPU: single-core RISC-V at 160 MHz
- Firmware: Arduino ESP32 core 3.3.11
- Arithmetic: float32, batch 1, one head, precomputed Q/K/V
- Accuracy gate: every element must satisfy absolute error <= 0.002 or relative
  error <= 0.02
- Timing: allocation and deterministic input generation excluded; one warm-up;
  median of 3, 5, or 7 samples based on workload

All optimized outputs passed. The exact tiled schedule reduced workspace from
65,536 bytes to 160 bytes for `N=128, d=32` (409.6x), and reduced the modeled
working set from 131,072 to 65,696 bytes (49.9%). It was 8.0% slower than the
materialized reference for that case (2.015 s versus 1.866 s). The polynomial
exponential approximation was also slower, so it is not retained as the
preferred kernel.

This result establishes that float tiling solves the quadratic activation-memory
problem but not the ESP32-C3 compute bottleneck. The next experiment quantizes
Q/K/V to int8 while preserving the float reference and the same accuracy gate.
