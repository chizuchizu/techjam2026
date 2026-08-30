# Single-board repeat audit — cases 9, 10, and 11

Date: 2026-08-30

The optimized USB single-board results for cases 9, 10, and 11 were repeated
because their published full-case times are unusually close despite using 1,
2, and 16 attention heads. Each run used a physical Seeed XIAO ESP32-C3 at
160 MHz, the case's non-WiFi `esp32-baseline` firmware, and the complete
official B=64 input batch. Device compute is the sum of all 64 firmware `us=`
counters; USB transfer time is reported only as end-to-end context.

## Results

| Case | Heads | Original compute | Repeat compute | Change | Repeat median forward | Repeat MFU/device | Accuracy |
|---:|---:|---:|---:|---:|---:|---:|---|
| 09 | 1 | 138.0273 s | 138.600 s on board A; 138.600 s on board B | +0.415% | 2.1656 s | 38.735% | Both runs 64/64 PASS; worst `max_abs=1.4403e-3` |
| 10 | 2 | 138.5358 s | 139.264 s | +0.526% | 2.1760 s | 38.551% | 64/64 PASS; worst `max_abs=1.2456e-3` |
| 11 | 16 | 138.6104 s | 139.107 s | +0.358% | 2.1735 s | 38.594% | 64/64 PASS; worst `max_abs=1.3076e-3` |

All repeats are within 0.53% of the original physical measurements. The
original headline values remain valid; the repeat artifacts are retained as
independent reproducibility evidence rather than replacing the first captures.

## Why the similar score is expected

For all three shapes, `B=64`, `S=D=F=128`, and `L=4`. The repository's model
work formula gives the same **8,589,934,592 FLOP** for each complete case:

```text
2 * B * L * S * (4*D^2 + 2*S*D + 2*D*F)
```

`H` does not appear because `H * head_dim = D`: changing the number of heads
partitions the QK and PV work without changing their total MAC count. It adds
some softmax and loop overhead, but projections dominate this optimized
single-board path. MFU is therefore expected to cluster near 39%.

This does not contradict the larger difference in the tiled WiFi workers.
That memory-first schedule repeats tile normalization and projection staging
per head, so case 11's 16 heads amplify scheduling overhead and make each
tiled replica substantially slower.

## Reproducibility fix found during the audit

Case 9's default build referenced `patched_sdk_libs_current/`, archives from a
different RISC-V toolchain ABI. PlatformIO Espressif32 7.0.1 could not link
them. The target is now pinned to 7.0.1 and uses the compatible in-repository
`patched_sdk_libs/`; both the optimized USB and opt-in tiled WiFi targets link.
The rebuilt USB firmware passed both physical 64-frame repeats above.

## Raw repeat evidence

- Case 9: [`results_case09_single_c3_repeat_a.json`](case-09/optimisation/results/results_case09_single_c3_repeat_a.json)
  and [`results_case09_single_c3_repeat_b.json`](case-09/optimisation/results/results_case09_single_c3_repeat_b.json)
- Case 10: [`results_case10_single_c3_repeat.json`](case-10/optimisation/results/results_case10_single_c3_repeat.json)
- Case 11: [`results_case11_single_c3_repeat.json`](case-11/optimisation/results/results_case11_single_c3_repeat.json)

The gate is the competition's elementwise OR rule: absolute error at most
0.002 or relative error at most 0.02. No run had a missing frame, transport
retry, non-finite value, or failing element.
