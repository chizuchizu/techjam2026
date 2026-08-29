# Measured results

The authoritative raw data is
[`esp32c3_attention_v3.csv`](esp32c3_attention_v3.csv). It was captured from a
physical XIAO ESP32-C3 at 160 MHz after compiling with Arduino-ESP32 3.3.11.

## Selected results

| Shape | Kernel | Median | Speedup | Workspace | Working set | Max abs | Gate |
|---|---|---:|---:|---:|---:|---:|---|
| `N=64,d=32` | float materialized | 466.684 ms | 1.00x | 16,384 B | 49,152 B | 0 | pass |
| `N=64,d=32` | mixed tiled | 368.958 ms | 1.27x | 160 B | 16,544 B | 0.000129 | pass |
| `N=64,d=32`, causal | all-int8 tiled | 188.708 ms | 1.26x | 160 B | 14,496 B | 0.003645 | **fail** |
| `N=64,d=32`, causal | mixed tiled | 187.603 ms | 1.26x | 160 B | 16,544 B | 0.000369 | pass |
| `N=128,d=32` | float materialized | 1.866 s | 1.00x | 65,536 B | 131,072 B | 0 | pass |
| `N=128,d=32` | mixed materialized | 1.336 s | 1.40x | 65,536 B | 98,304 B | 0.000098 | pass |
| `N=128,d=32` | mixed tiled | 1.483 s | 1.26x | 160 B | 32,928 B | 0.000098 | pass |

“Mixed” means int8 Q/K, int16 V, float stable softmax, and float output. Its
quantization error is included in the gate. Quantization itself is outside the
timed region because the intended end-to-end format supplies quantized Q/K/V;
the end-to-end benchmark below measures that conversion cost explicitly.

## End-to-end attention layer

The latest physical-board capture is
[`esp32c3_end_to_end_v2.log`](esp32c3_end_to_end_v2.log), with its summary in
[`esp32c3_end_to_end_v2.csv`](esp32c3_end_to_end_v2.csv). It uses 16 tokens,
model dimension 32, four heads, tile size 8, and two padded tokens. Both causal
and non-causal cases include Q/K/V projections, activation quantization, tiled
mixed attention, fused V dequantization, and output projection.

| Mask | Projection format | Median | Speedup | Workspace | Working set | Host max abs | Host gate |
|---|---|---:|---:|---:|---:|---:|---|
| Padding | Float weights and activations | 130.428 ms | 1.014x | 96 B | 31,328 B | 0.000180 | pass |
| Padding | Int8 weights, int16 activations | 43.362 ms | 3.051x | 96 B | 21,600 B | 0.000452 | pass |
| Padding + causal | Float weights and activations | 117.278 ms | 0.998x | 96 B | 31,328 B | 0.000232 | pass |
| Padding + causal | Int8 weights, int16 activations | 30.250 ms | 3.870x | 96 B | 21,600 B | 0.002192 | pass |

The independent Python reference rebuilds the fixture from documented integer
formulas and compares all 512 emitted elements. It shares no attention or
projection implementation with the firmware. Large maximum relative errors are
from expected values near zero; every such element passes the 0.002 absolute
error branch or the 2% relative-error branch of the required OR rule. The V1
capture is retained to preserve the first end-to-end iteration.

## Conclusions supported by the data

1. Online tiling is a memory optimization on float ESP32-C3, not automatically a
   speed optimization: exact float tiling is 7–8% slower for larger cases.
2. Replacing `expf` with the tested polynomial approximation is not beneficial.
3. Int8 Q/K produces a consistent speedup, but int8 V can violate the absolute
   tolerance on causal attention.
4. Int16 V fixes that failure across all nine tested cases while retaining most
   of the speed and memory benefit.
5. The best kernel depends on the constraint: mixed materialized is fastest;
   mixed tiled is the choice when RAM or larger sequence lengths matter.
6. Primitive speedup does not predict whole-layer speedup. Projection and
   conversion work consumes nearly all of the mixed kernel's advantage at this
   small end-to-end shape.
7. Quantizing all four projection matrices to int8 with per-output-channel
   scales, while retaining int16 projection activations, restores a 3.05–3.87x
   measured whole-layer speedup and reduces the modeled working set by 31.1%.
