# ESP32 priorities

## Ready for contribution

1. Add automatic node registration, unique node IDs, heartbeats, and a worker
   registry for a fleet of ESP32-C3 boards.
2. Measure four comparable C3 boards at `N=128, H=4, d_head=32`; the matched
   two-C3 result is complete at 1.98–2.00x speedup.
3. Cache worker performance profiles and retry a timed-out head on another
   healthy node.
4. Integrate distributed heads with one complete layer of `esp32-baseline`,
   including Q/K/V projection and output projection in the measured path.
5. Replace the remaining floating-point attention, LayerNorm, and GELU work in
   `esp32-baseline` with independently validated fixed-point kernels.
6. Add a larger trained tokenizer/model and evaluate prompts that were not in
   its training corpus.

## Completed foundations

- Complete trained tiny Transformer on a physical XIAO ESP32-C3.
- Official-size four-layer C baseline with host and device accuracy captures.
- Versioned UDP/TCP worker protocol with capability queries.
- Exact key/value sharding prototype and whole-head comparison.
- Persistent TCP head scheduling on two physical, heterogeneous ESP32 boards.
- Matched two-C3 scaling at 99.17–99.99% efficiency with all outputs passing.
- Round-robin, calibrated-all, and latency-minimizing assignment policies.

The retired NVIDIA H200 backlog lives on in git history (removed with the 2026 competition split).
