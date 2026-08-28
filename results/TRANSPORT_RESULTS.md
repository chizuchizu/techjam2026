# One-board LAN transport results

The authoritative data is
[`esp32c3_transport_v1.csv`](esp32c3_transport_v1.csv). It was measured between
WSL and the physical XIAO ESP32-C3 over a local 2.4 GHz LAN. The board reported
RSSI -26 dBm. The run sent 30 validated binary echo requests at every payload
size. Private addresses and the SSID are intentionally omitted.

| Protocol | Payload | Median RTT | p90 RTT | Round-trip payload rate | Loss / invalid |
|---|---:|---:|---:|---:|---:|
| UDP | 64 B | 7.258 ms | 10.097 ms | 0.141 Mbps | 0 / 0 |
| UDP | 256 B | 6.663 ms | 10.908 ms | 0.615 Mbps | 0 / 0 |
| UDP | 1,024 B | 7.095 ms | 8.661 ms | 2.309 Mbps | 0 / 0 |
| UDP | 1,400 B | 7.371 ms | 12.688 ms | 3.039 Mbps | 0 / 0 |
| TCP | 256 B | 9.266 ms | 15.167 ms | 0.442 Mbps | 0 / 0 |
| TCP | 1,024 B | 8.936 ms | 13.601 ms | 1.833 Mbps | 0 / 0 |
| TCP | 4,096 B | 11.325 ms | 15.404 ms | 5.787 Mbps | 0 / 0 |
| TCP | 16,384 B | 49.571 ms | 58.917 ms | 5.288 Mbps | 0 / 0 |
| TCP | 32,768 B | 105.576 ms | 115.261 ms | 4.966 Mbps | 0 / 0 |

## Consequences for the ESP32 cluster

- UDP is suitable for small control messages and head outputs that fit below
  the no-fragmentation payload ceiling. It held roughly 7 ms median RTT with no
  loss in this close-range run.
- TCP is more efficient for multi-kilobyte activation transfers. The best
  measured rate was 5.787 Mbps at 4 KiB; larger writes increased latency and
  slightly reduced goodput.
- A 1,024-byte activation round trip takes about 7.1 ms over UDP or 8.9 ms over
  TCP. This is material but smaller than the 43.4 ms optimized non-causal
  attention-layer time at the current fixture size.
- These are one-worker link results, not multi-board speedup. Four workers may
  contend for the same 2.4 GHz channel, so the final benchmark must measure
  concurrent transfers rather than multiplying this throughput by four.

UDP broadcast discovery did not traverse this WSL/AP path, while direct UDP and
TCP worked without loss. The coordinator therefore supports an explicit worker
IP; later cluster configuration should also allow static leases or serial
provisioning instead of depending only on broadcast discovery.
