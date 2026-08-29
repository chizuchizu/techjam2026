# Shared benchmark reference

The executable reference defines all 14 official cases and is a shared
read-only input to the case-specific implementations. The supplied problem
statement remains at the repository root.

| File | Purpose |
|---|---|
| [`torch_transformer_benchmark.py`](torch_transformer_benchmark.py) | Official PyTorch model, input generation, accuracy gate, and timing harness |
| [`../../COMPETITION_RULES.MD`](../../COMPETITION_RULES.MD) | Supplied competition rules and test-case table |

The reference has no universal latency result because timing depends on the
host. Each `case-NN/` directory owns its physical ESP32 baseline and subsequent
comparisons.
