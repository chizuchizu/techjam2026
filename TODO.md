# ESP32 priorities

## Ready for contribution

1. Add automatic node registration, unique node IDs, heartbeats, and a worker
   registry for a fleet of ESP32-C3 boards.
2. Integrate distributed heads with one complete layer of
   `benchmarks/case-02/optimisation/esp32-baseline`, adding output projection,
   both residuals, second LayerNorm, and FFN to the now-verified
   first-LayerNorm/Q/K/V/attention path.
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

The retired NVIDIA H200 backlog lives on in git history (removed with the 2026 competition split).
