# Ring + LongNet experiments

C++17 reference experiment for **case 8**. The leading `8` in the parameter
list is the case number:

```text
B=64  D=1024  S=128  H=4  L=4  causal=true  F=1024
```

`main.cpp` keeps the official pre-LN Transformer order and materializes two
separate worker shards:

- worker 0: heads 0–1 and FFN rows 0–511
- worker 1: heads 2–3 and FFN rows 512–1023
- the coordinator merges partial output projections and residuals

Modes:

```text
baseline  dense causal attention, head-sharded
ring      exact online-softmax attention over two KV blocks
longnet       LongNet-style geometric segment/dilation mixture
ring-longnet  LongNet patterns plus Ring Attention KV traversal
```

The `longnet` path follows the paper's gather/attention/scatter idea: it uses
segment/dilation patterns `(16,1)`, `(32,2)`, and `(128,8)`, mixes pattern
outputs using their softmax denominators, and never allocates a dense `S x S`
score matrix. `ring-longnet` adds two Ring Attention rotations for every
pattern: each board keeps its query shard fixed, streams one K/V tile from the
neighbor, and merges online softmax statistics. It is intentionally not
numerically identical to dense baseline.

`baseline` and `ring` implement the same attention graph; `longnet` is not
numerically equivalent to the dense baseline and is included for comparison.

## Build

```sh
cmake -S . -B build
cmake --build build --config Release
./build/ring_longnet_experiments --mode baseline --repeats 1
./build/ring_longnet_experiments --mode ring --repeats 1
./build/ring_longnet_experiments --mode longnet --batch 1
./build/ring_longnet_experiments --mode ring-longnet --batch 1
```

The implementation is intentionally portable C++17. The worker calls are
isolated so they can later be replaced by ESP-NOW/TCP RPCs. LongNet reduces
attention scratch/communication, but does **not** reduce the roughly 101 MB
fp32 model weights (about 25 MB if quantized to int8, or 12.6 MB per board for
two equal shards). The printed shard sizes therefore show why int8/4-bit
storage and external flash/PSRAM are still required. It is not directly
flashable to the XIAO ESP32-C3: a physical version also needs sequence-sharded
activations, streamed batch items, and transport worker firmware.
