# Exact key/value-sharded result

The authoritative one-worker capture is
[`esp32c3_kv_sharded_v1.csv`](esp32c3_kv_sharded_v1.csv). Each of four attention
heads is split into four key/value ranges. The physical ESP32 processes a shard
and returns, for every query, its stable local maximum, exponential sum, and
unnormalized value numerator. The host merges those statistics with the exact
online-softmax equations before applying the output projection.

Each value below is the median of seven complete 16-shard dispatches after one
warm-up. One physical worker processed every shard sequentially.

| Mask | 16-shard wall time | Worker compute total | Decode total | Approx. protocol/network remainder | Full max abs | Gate |
|---|---:|---:|---:|---:|---:|---|
| Padding | 152.262 ms | 25.513 ms | 0.176 ms | 126.573 ms | 0.000180 | pass |
| Padding + causal | 142.536 ms | 14.074 ms | 0.176 ms | 128.286 ms | 0.000232 | pass |

All eight head merges and full layer outputs pass independent validation. Each
250-byte task fits comfortably in one UDP datagram. Each 660-byte reply contains
16 query records with `(max, sum, numerator[8])`.

## Comparison with head parallelism

For this four-head fixture, head parallelism is the better first cluster
strategy:

| One-worker protocol | Datagrams per layer | Total request + reply payload | Non-causal wall time | Causal wall time |
|---|---:|---:|---:|---:|
| Whole-head tasks | 4 | 4,248 B | 58.138 ms | 45.538 ms |
| Four KV shards per head | 16 | 14,560 B | 152.262 ms | 142.536 ms |

KV sharding performs similar total attention compute but sends 3.43x more data
and pays four times as many round trips. Its value is scalability when one head
or its key/value state no longer fits a node; it is not the right default for
this small shape. The coordinator can assign shards concurrently across
multiple worker IPs, but no multi-board timing claim is made yet.

