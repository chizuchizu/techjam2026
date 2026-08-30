# Cases 1–5 with eight WiFi ESP32-C3 nodes available

Eight Seeed XIAO ESP32-C3 boards ran the 16-row tiled full-forward firmware
over persistent WiFi TCP. Independent batch inputs were assigned round-robin
to full-forward replicas: **8-node WiFi data parallel** (`8-node WiFi DP`).

## Complete-case measurements

| Case | B | Active nodes | Compute wall | End-to-end WiFi wall | Scaling vs 1 tiled node | Gate |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 64 | 8 | **33.713 s** | 40.0 s | 8.00x | 64/64 PASS |
| 2 | 1 | 1 of 8 | **4.220 s** | 4.9 s | 1.00x | 1/1 PASS |
| 3 | 4 | 4 of 8 | **4.218 s** | 5.0 s | 4.00x | 4/4 PASS |
| 4 | 16 | 8 | **8.438 s** | 10.3 s | 8.00x | 16/16 PASS |
| 5 | 128 | 8 | **67.451 s** | 81.4 s | 8.00x | 128/128 PASS |

All **213/213** forwards completed with no missing inputs and zero failing
elements. The worst maximum absolute error was `1.3031e-3`.

Compute wall excludes host transport, matching the repository's one-board,
two-board and four-node timing columns. End-to-end wall includes every 64 KB
input and output over WiFi TCP. The raw results are:

- [`results_cases_1_3_4_5_eight_c3_wifi.json`](results_cases_1_3_4_5_eight_c3_wifi.json)
  — the all-eight run; the harness skipped B=4 because B was smaller than N
- [`results_case3_four_active_eight_available_c3_wifi.json`](results_case3_four_active_eight_available_c3_wifi.json)
  — case 3 on its maximum useful four nodes
- [`results_case2_one_active_eight_available_c3_wifi.json`](results_case2_one_active_eight_available_c3_wifi.json)
  — case 2 on its maximum useful one data-parallel node

## Interpretation

Data parallel utilization is `min(B, N)`. Cases 1, 4 and 5 therefore use all
eight boards and retain essentially perfect 8x compute scaling. Case 3 has
only four independent inputs and saturates at four boards; case 2 has one and
saturates at one. Idle boards are not counted as a speedup.

Doubling the cluster from four to eight nodes almost exactly halves compute
wall for the fully utilized cases: 67.465 to 33.713 s (case 1), 16.853 to
8.438 s (case 4), and 134.887 to 67.451 s (case 5). WiFi-inclusive wall is
18–22% above device compute wall in those runs, but scaling is still linear
because nodes exchange no data during a forward.

The tiled WiFi forward remains about 2.12x slower per node than the opt23
non-WiFi forward. Eight-way parallelism overcomes that local penalty for the
larger batches: cases 1, 4 and 5 are about 3.78x faster than one opt23 board.
Case 3 is 1.89x faster with its four usable nodes. Case 2 cannot benefit from
data parallelism; its token-row split is still the appropriate multiboard
method.
