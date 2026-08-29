# ESP32 priorities

## Ready for contribution

1. Add automatic node registration, unique node IDs, heartbeats, and a worker
   registry for a fleet of ESP32-C3 boards.
2. Extend the two-board row partition in
   `benchmarks/case-02/multiboard/esp32-cluster-full` to four boards (`i % N`),
   and cut the per-board replicated weight streaming that currently stands
   between the measured 1.56x and 2x.
3. Cache worker performance profiles and retry a timed-out head on another
   healthy node.
4. Replace the remaining floating-point attention, LayerNorm, and GELU work in
   the case-02 single-board implementation with independently validated
   fixed-point kernels.
5. Add a larger trained tokenizer/model and evaluate prompts that were not in
   its training corpus.

## Completed foundations

- Complete trained tiny Transformer on a physical XIAO ESP32-C3.
- Official-size four-layer C baseline with host and device accuracy captures.
- Versioned UDP/TCP worker protocol with capability queries.
- Exact key/value sharding prototype and whole-head comparison.
- Persistent TCP head scheduling on two physical, heterogeneous ESP32 boards.
- Matched two-C3 scaling at about 2x with all outputs passing.
- Four matched C3 boards at 3.92–3.98x speedup with every measured response
  passing the accuracy gate.
- Official case-2 layer-0 LayerNorm, Q/K/V projections, and attention on two
  physical C3s at 2.00x, with five device seeds passing.
- Round-robin, calibrated-all, and latency-minimizing assignment policies.
- Complete four-layer case-2 forward distributed across two physical C3s at
  1.276 s against 1.990 s on one board (1.56x) on the opt23 kernels, 25/25 host
  seeds passing the benchmark gate, with the per-layer K/V exchange overlapped
  down to 5-88 ms of waiting.
- Batch cases 1, 3, 4 and 5 data-parallel on two physical C3s at exactly 2.00x
  (B = 64, 4, 16, 128), all 212 forwards gated individually against the torch
  reference with zero failing elements.

The retired NVIDIA H200 backlog lives on in git history (removed with the 2026 competition split).
