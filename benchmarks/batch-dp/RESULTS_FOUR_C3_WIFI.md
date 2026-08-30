# Cases 1–5 on four WiFi ESP32-C3 nodes

Four Seeed XIAO ESP32-C3 boards ran the tiled full-forward firmware over
persistent WiFi TCP. The standard term used in the repository tables is
**4-node WiFi data parallel** (`4-node WiFi DP`): independent batch inputs are
assigned round-robin to four full-forward replicas.

## Complete-case measurements

| Case | B | Active nodes | Compute wall | End-to-end WiFi wall | Scaling vs 1 tiled node | Gate |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 64 | 4 | **67.465 s** | 80.8 s | 4.00x | 64/64 PASS |
| 2 | 1 | 1 of 4 | **4.2137 s** | 4.9 s | 1.00x | PASS, 0 failing elements |
| 3 | 4 | 4 | **4.215 s** | 5.0 s | 4.00x | 4/4 PASS |
| 4 | 16 | 4 | **16.853 s** | 19.8 s | 4.00x | 16/16 PASS |
| 5 | 128 | 4 | **134.887 s** | 157.7 s | 4.00x | 128/128 PASS |

Cases 1, 3, 4 and 5 completed all **212/212** forwards with no missing inputs
and zero failing elements. Worst maximum absolute error was `1.3031e-3`.
Case 2 has only one independent input, so data parallelism cannot activate the
other three boards; its earlier distributed physical gate passed 25/25 seeds
with worst `max_abs=1.2370e-3`.

Compute wall excludes host transport, matching the existing one-board and
two-board columns. End-to-end wall includes every 64 KB input and output over
WiFi TCP. The raw results are:

- [`results_cases_1_3_4_5_four_c3_wifi.json`](results_cases_1_3_4_5_four_c3_wifi.json)
- [`results_case2_wifi_tiled.json`](results_case2_wifi_tiled.json)

## Interpretation

The tiled forward is 2.12x slower per node than opt23 because repeated row-tile
and head-sequential sweeps are the cost of freeing enough SRAM for WiFi.
Despite that penalty, aggregate compute scales linearly through the measured
four nodes. For B=4, four WiFi nodes take 4.215 s versus 7.962 s for one opt23
node, a 1.89x absolute compute improvement. Case 2 does not benefit because
`B=1`; its separate token-row split remains the faster multiboard strategy.
