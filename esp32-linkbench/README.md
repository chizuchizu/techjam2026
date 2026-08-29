# esp32-linkbench — ultra-low-latency ESP-NOW link

Two Seeed XIAO ESP32-C3 boards talk ESP-NOW over WiFi **with no router**:
one runs a softAP (`LINKFAST`, ch1), the other joins as a STA client. Built
for a TikTok hackathon where round-trip latency on the link is the metric
(`linkfast` branch).

Optimized firmware (`src/main.cpp`): unicast to a learned peer, PHY rate
forced to `WIFI_PHY_RATE_MCS7_SGI`, `WIFI_PS_NONE` + max wake window, HT40,
TX power maxed. A `-DLF_COMPAT=1` build keeps the old defaults as the
baseline (broadcast, no rate/PS tuning). Benchmarks are host-driven from
USB CDC; read `tools/run_bench.py`.

## Measured (on-board, ch=1, no router)

PING 2000 rounds/payload:

| P | OPT med | OPT min | OPT lost | COMPAT med | COMPAT lost | speedup |
|---|---|---|---|---|---|---|
| 0   | 838 µs | 682 µs | 0 | 1.9 ms | 1 | 2.3x |
| 16  | 851 µs | 674 µs | 0 | 2.0 ms | 0 | 2.4x |
| 64  | 846 µs | 662 µs | 0 | 2.4 ms | 1 | 2.9x |
| 240 | **861 µs** | 702 µs | 0 | **3.9 ms** | 2 | **4.5x** |

STREAM 300 packets/payload: OPT 79.6 / 149.6 / **254.5 KB/s** (P=64/128/240),
300/300 ACKed, 0 fail — COMPAT 26.5 / 42.7 / 60.9 KB/s (**3.0–4.2x** faster).
Server ground truth matched in both runs (304 pkts / 74,720 B).

End-to-end RTT ≈ **0.84 ms** — ~7x under the ESP-NOW 1 Mbps default (~6 ms).
Details: `docs/OPTIMIZATION_GUIDE.md`, `docs/ESP32_LINKFAST_FINAL_REPORT.md`.

## Build & run

Build: `pio run -e link-server -e link-client -e link-server-compat -e link-client-compat`.

A/B benchmark (server → `/dev/cu.usbmodem101`, client → `/dev/cu.usbmodem1101`;
adjust `P_SERVER`/`P_CLIENT` in `tools/run_bench.py` for your ports):

    python3 tools/run_bench.py opt     --timeout 150   # optimized
    python3 tools/run_bench.py compat  --timeout 150   # baseline
    # results -> .logs/result_<mode>.json (refuses to overwrite empty captures)

The harness flashes each env, forces a clean boot (`esptool read_mac`), waits
for the client's periodic `CLIENT|READY`, sends `B`, and cross-checks server
ground truth. No router, no serial break-out needed.
