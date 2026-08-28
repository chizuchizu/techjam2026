# Problems, evidence, and solutions

This is the pitch backbone: each solution is tied to an observed or testable
problem rather than presented as an unsupported optimization claim.

| Problem | Evidence / consequence | Implemented response | Next response |
|---|---|---|---|
| Attention scores use quadratic RAM. | At `N=128`, a float score matrix alone is 65,536 B. | Block-online softmax reduces attention workspace to 160 B at tile 8. | Use the same online statistics for distributed key/value sharding. |
| ESP32-C3 float arithmetic is slow. | Float reference takes 1.866 s at `N=128,d=32`; Espressif documents float as software-emulated on C3. | Int8 Q/K accelerates dot products. | Fixed-point/LUT softmax and ESP-IDF `-O3`. |
| Memory optimization can hurt speed. | Exact tiled float is about 8% slower than materialized float at the largest shape. | Keep both kernels and select by shape/memory budget. | Sweep tile sizes and fuse projections so tiling avoids more traffic. |
| A “fast” exponential may not be fast on this CPU. | Polynomial exp was consistently slower than `expf`. | It remains as measured negative evidence, not the default. | Profile ROM `expf`; only revisit with integer LUT softmax. |
| Naive int8 can violate accuracy. | Fully int8 Q/K/V failed six elements for causal `N=64,d=32`, max absolute error 0.003645. | Preserve V as int16; max absolute error becomes 0.000369 and all cases pass. | Calibrate per layer/head on real model activations. |
| Relative error explodes near zero. | Some passing cases show large max relative error even though absolute error is below 0.002. | Apply the exact specified OR rule element by element and report both metrics. | Add error histograms and adversarial input scales. |
| Kernel speedup can disappear end to end. | With float projections and activation conversion included, mixed precision is only 1.015x faster non-causally and 0.999x for causal attention. | Int16 activations plus per-output-channel int8 projection weights raise measured speedup to 3.05x and 3.87x while passing. | Validate on trained-model data and fuse intermediate conversion. |
| Model weights exceed on-chip flash/RAM. | A standard 6-layer `d=512,ffn=2048` float model needs tens of MB of weights. | Current milestone isolates attention and reports its true scope. | Quantize and partition stationary weights across nodes. |
| Wireless traffic can erase parallel speedup. | Measured 1 KiB RTT is 7.1 ms UDP and 8.9 ms TCP; TCP peaks at 5.79 Mbps for 4 KiB in the current LAN test. | Implemented a versioned binary protocol and measured zero loss/corruption across 330 one-worker trials. | Add head tasks, concurrent workers, batching, and crossover measurements. |
| Broadcast discovery may fail across WSL/AP networking. | UDP broadcast received no reply, but direct UDP/TCP to the board completed every trial. | Coordinator accepts explicit worker IPs and the device reports its address over USB serial. | Add static leases or provisioning and retry discovery on the native host. |
| A cluster protocol diagram is not an implementation. | Real head tasks need dtype/shape/mask metadata, byte order, validation, timing, retry behavior, and reassembly. | Added a 534-byte binary `HEAD_TASK`, 528-byte `HEAD_RESULT`, stateless retries, concurrent coordinator, and independent full-output validation. | Measure four simultaneous workers and add duplicate/straggler telemetry. |
| Distributed softmax needs global normalization. | Local softmax values cannot simply be concatenated. | Implemented mergeable `(max, denominator, numerator)` worker results; all four heads and both full outputs pass after host merging. | Validate concurrent behavior on two and four nodes. |
| Exact sharding can still be the wrong decomposition. | Four KV shards per head move 14,560 B and take 152 ms on one worker, versus 4,248 B and 58 ms for whole heads. | Measured both protocols at identical shape, dtype, and accuracy and selected head parallelism first. | Revisit KV sharding only when a whole head exceeds node memory or compute limits. |
| More boards introduce stragglers and failures. | End-to-end time becomes the slowest-node time plus communication. | Protocol design includes run IDs and explicit result ownership. | Add deadlines, retries, health telemetry, and profiled assignment. |
| “World first” is not defensible. | Published work already covers MCU attention and distributed Transformer inference. | Position the work as an open, measured ESP32-C3 cluster study. | Narrow any novelty claim after a formal literature review. |

## Results worth showing in the pitch

- 409.6x attention-workspace reduction at `N=128,d=32` (65,536 B to
  160 B).
- 1.40x fastest passing speedup with int8 Q/K + int16 V and materialized scores.
- 1.26x speedup with the memory-efficient mixed tiled kernel while cutting the
  full modeled working set by 74.9%.
- A failed all-int8 causal result followed by a measured mixed-precision fix.
- A complete four-head layer with padding and causal masks whose 512 outputs
  pass an independent host reference in both cases.
- An end-to-end result showing that projection and conversion overhead consumes
  the primitive's speedup, followed by a projection format that restores a
  measured 3.05–3.87x speedup.
- A real LAN head-task run where all individual heads and both reassembled layer
  outputs pass; the measured one-worker network remainder is about 31 ms for
  four sequential tasks.
- An exact distributed-softmax run that passes but moves 3.43x more data than
  whole-head tasks, giving measured evidence for the chosen decomposition.

The failed iteration is valuable: it shows why accuracy validation must guide
optimization rather than being added after performance work.
