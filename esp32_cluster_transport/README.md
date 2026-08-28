# ESP32 cluster transport benchmark

This firmware measures the LAN before treating multiple ESP32 boards as a
compute cluster. It implements a versioned 12-byte binary protocol with UDP
discovery/echo on port 4210 and persistent TCP on port 4211. It accepts
`HEAD_TASK` messages containing one int8 Q/K + int16 V attention head and
returns the float context plus separate payload-decode and compute timings.
Small heads use UDP; official-size `N=128, d_head=32` tasks automatically use
TCP because each direction is about 16.4 KB.
It additionally accepts `KV_SHARD_TASK` messages and returns unnormalized local
softmax `(maximum, sum, numerator)` statistics for exact host-side merging.

Copy `secrets.example.h` to the ignored `secrets.h`, set the LAN credentials,
then compile and upload:

```sh
arduino-cli compile --fqbn esp32:esp32:XIAO_ESP32C3 esp32_cluster_transport
arduino-cli upload --fqbn esp32:esp32:XIAO_ESP32C3 \
  --port /dev/ttyACM0 esp32_cluster_transport
python3 tools/benchmark_transport.py --output results/esp32c3_transport_v1.csv
python3 tools/run_head_parallel.py --workers 192.168.0.X \
  --output results/esp32c3_head_parallel_v1.csv
python3 tools/run_large_head_parallel.py --workers 192.168.0.X \
  --scheduler round-robin \
  --output results/esp32c3_large_head_tcp_v1.csv
python3 tools/discover_esp32_workers.py --workers 192.168.0.X
python3 tools/run_kv_sharded.py --workers 192.168.0.X \
  --output results/esp32c3_kv_sharded_v1.csv
```

The host script first tries UDP broadcast discovery. WSL broadcast forwarding
can be blocked even when normal LAN traffic works; in that case read the
`TRANSPORT_READY` address from USB serial and pass it explicitly, for example
`--host 192.168.0.X`. Payloads and request IDs are verified; corrupt or missing
replies are counted rather than discarded.

`run_head_parallel.py` sends the four deterministic heads sequentially when one
worker is listed. With comma-separated worker IPs, it assigns heads round-robin,
serializes the heads assigned to the same worker, and runs different workers
concurrently. The small fixed task is 16 tokens by 8 features; each 534-byte
request and 528-byte response fits in one UDP datagram. `--transport tcp`
exercises the persistent TCP path explicitly.

`run_large_head_parallel.py` exercises four `128 x 32` heads using persistent
TCP connections. `--scheduler round-robin` assigns equal head counts;
`calibrated-all` measures the workers and uses every node; `calibrated` minimizes
predicted latency and may idle a much slower node. Capability queries report
chip model, core count, clock, free heap, supported transports, and maximum
head shape. Matched-C3 scaling is in
[`../results/TWO_C3_PARALLEL_RESULTS.md`](../results/TWO_C3_PARALLEL_RESULTS.md);
the heterogeneous-board experiment is in
[`../results/LARGE_HEAD_PARALLEL_RESULTS.md`](../results/LARGE_HEAD_PARALLEL_RESULTS.md).

`run_kv_sharded.py` splits each head's 16 keys into four shards. A 250-byte task
returns 660 bytes of stable online-softmax statistics. The coordinator merges
the four shards without normalizing locally, so the result is mathematically
equivalent to processing all visible keys together.

This protocol is intended for a trusted benchmark LAN. It has no authentication
and must not be exposed to the public internet.
