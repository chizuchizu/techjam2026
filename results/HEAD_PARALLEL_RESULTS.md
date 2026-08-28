# Head-parallel protocol result

The authoritative one-worker capture is
[`esp32c3_head_parallel_v1.csv`](esp32c3_head_parallel_v1.csv). The host
projected and quantized the deterministic `N=16, d_model=32, 4 heads` fixture,
sent one int8 Q/K + int16 V head at a time over UDP, reconstructed the four
returned contexts, applied the output projection, and compared every element
with the independent float reference.

Each value below is the median of seven complete dispatches after one warm-up.
The available physical board processed the four heads sequentially.

| Mask | Four-head wall time | Worker compute total | Decode total | Approx. protocol/network remainder | Full max abs | Gate |
|---|---:|---:|---:|---:|---:|---|
| Padding | 58.138 ms | 27.460 ms | 0.072 ms | 30.606 ms | 0.000180 | pass |
| Padding + causal | 45.538 ms | 14.766 ms | 0.072 ms | 30.700 ms | 0.000232 | pass |

All eight individual head cases also pass. Every head uses a 534-byte request
and a 528-byte response, both below the measured 1,400-byte UDP payload. Median
per-head compute is 6.83–6.89 ms non-causally and 3.68–3.71 ms causally. Median
request/response RTT is 13.44–15.22 ms and 10.38–12.26 ms respectively.

## Interpretation

- This proves the binary task format, padding/causal semantics, quantized
  arithmetic, result reassembly, and independent accuracy check over the real
  LAN.
- One worker is not a speedup configuration: four serialized round trips make
  it slower than the local head computation.
- With four worker IPs, the coordinator already dispatches heads concurrently.
  The measured single-worker RTT suggests a possible lower critical path near
  the slowest head RTT, but that is not a four-board result: concurrent Wi-Fi
  contention, coordinator projections, output projection, and stragglers still
  need physical measurement.
- The worker is stateless, so retrying a lost `HEAD_TASK` is safe. A production
  protocol still needs authentication, run IDs beyond the request ID, and
  explicit duplicate suppression/telemetry.

