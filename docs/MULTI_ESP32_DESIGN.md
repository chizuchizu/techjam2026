# Multi-ESP32 attention design

The cluster stage should begin with the decomposition that communicates the
least data, then add the more interesting distributed-softmax path.

## Option A: attention-head parallelism (implement first)

For `H` heads, assign one or more complete heads to each worker. The coordinator
sends the normalized input or precomputed Q/K/V slice; each worker returns its
`N x d_head` context. Heads require no cross-worker softmax reduction.

Advantages:

- Exact and simple.
- One synchronization point per attention layer.
- Natural fit for four boards and an eight-head test shape.
- Each board stores only its assigned projection weights.

Limitations:

- Speedup cannot exceed the number of useful head shards.
- The coordinator still performs concatenation and output projection unless
  that projection is also partitioned.
- Broadcasting float activations can dominate compute; quantized binary payloads
  are required.

## Option B: key/value sequence sharding

This is the stronger systems demonstration because it uses the mergeability of
online softmax. Partition key/value indices across workers. For query row `i`,
worker `r` computes:

```text
m[r,i] = max_j score(i,j)                         for j owned by r
l[r,i] = sum_j exp(score(i,j) - m[r,i])
u[r,i] = sum_j exp(score(i,j) - m[r,i]) * V[j]
```

The coordinator merges workers without approximation:

```text
m[i] = max_r m[r,i]
l[i] = sum_r exp(m[r,i] - m[i]) * l[r,i]
u[i] = sum_r exp(m[r,i] - m[i]) * u[r,i]
output[i] = u[i] / l[i]
```

This is the same stable statistic used by the single-board tiled kernel. It
avoids transferring an `N x N` score matrix, but every worker needs Q and the
coordinator receives one numerator vector per worker/query.

## Protocol

Use binary messages, never JSON, for benchmark traffic.

Common header:

```text
magic:u32 | version:u8 | type:u8 | node:u8 | flags:u8
run_id:u32 | layer:u16 | head:u16 | shard_begin:u16 | shard_end:u16
sequence:u16 | dimension:u16 | payload_bytes:u32 | checksum:u32
```

Message types: `HELLO`, `CONFIG`, `INPUT_SHARD`, `RUN`, `PARTIAL`, `RESULT`,
`ERROR`, and `HEARTBEAT`. Every result includes compute microseconds separately
from transport time.

## Network experiment order

1. Measure payload throughput and RTT for 256 B through 32 KB over TCP, UDP, and
   ESP-NOW.
2. Run one worker remotely with no partitioning to quantify pure overhead.
3. Run two head shards; compare compute-only, communication-only, and end-to-end
   latency.
4. Scale to four boards. Stop claiming speedup when the slowest worker or network
   dominates.
5. Only then try ten boards and dynamic assignment.

## Required cluster result table

For every node count report shape, dtype, accuracy, compute maximum across
workers, communication time, coordinator merge time, end-to-end median/p90,
speedup, payload bytes, packet loss/retries, and energy if instrumentation is
available.

No multi-node performance claim is currently made because only one physical
board has been measured.

