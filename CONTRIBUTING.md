# Contributing

The active scope is ESP32. Keep changes small enough that another teammate can
reproduce and review them without access to your board.

## Workflow

1. Pick one item from [`TODO.md`](TODO.md), or describe the problem in the pull
   request before implementing a new direction.
2. Create a short branch such as `esp32/fleet-discovery` or
   `esp32/fixed-point-softmax`.
3. Keep firmware, host validation, and measurement changes in the same pull
   request when they implement one experiment.
4. Run `make check` and the component-specific validation.
5. Commit raw measurements under the owning case and approach, for example
   `benchmarks/case-02/multiboard/results/`, and explain what hardware produced
   them. Open the pull request against `main`.

## Validation expected by area

| Change | Minimum evidence |
|---|---|
| C/C++ math kernel | Independent host reference, error gate, and timing |
| ESP32 firmware | Successful compile plus physical upload/run when hardware is required |
| Cluster protocol | Per-head validation, complete reconstructed output, payload sizes, and wall time |
| Python coordinator | `make check` plus a deterministic fixture or physical capture |
| Documentation only | Working relative links and commands consistent with the current paths |

The numerical gate is applied per output element:

```text
absolute_error <= 0.002 OR relative_error <= 0.02
```

Report failed-element count and maximum absolute and relative error. A speedup
without the same input shape and passing accuracy is not a valid comparison.

## Result naming

Use lowercase descriptive names containing the hardware and version, for
example:

```text
benchmarks/experiments/attention-layer/results/esp32c3_attention_v4.csv
benchmarks/case-02/multiboard/results/four_c3_head_parallel_v1.csv
```

Reports should distinguish measured results from projections and state whether
setup, calibration, communication, and warm-up are included in timing.

## Repository hygiene

- Do not commit `secrets.h`, credentials, MAC addresses, private IP addresses,
  build output, virtual environments, or serial-port-specific configuration.
- Do not copy old H200 benchmark work back into the active case directories.
  The official problem statement and PyTorch reference stay at the repository
  root; retired GPU work stays under `archive/`.
- Put case-specific firmware, tools, and results under the matching
  `benchmarks/case-NN/` directory. Do not mix results from different official
  shapes.
- Do not silently rewrite another teammate's result. Add a new versioned raw
  capture and explain why the conclusion changed.
- Prefer portable C/C++ for shared kernels and standard-library Python for
  coordinators. Document additional dependencies in `requirements.txt`.
