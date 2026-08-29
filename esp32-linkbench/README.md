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
| 0   | 850 µs | 661 µs | 0 | 1.9 ms | 1 | 2.3x |
| 16  | 860 µs | 670 µs | 0 | 2.0 ms | 0 | 2.4x |
| 64  | 853 µs | 683 µs | 0 | 2.4 ms | 1 | 2.9x |
| 240 | **878 µs** | 704 µs | 0 | **3.9 ms** | 2 | **4.4x** |

STREAM 300 packets/payload: OPT 58.9 / 111.0 / **197.0 KB/s** (P=64/128/240),
300/300 ACKed, 0 fail — COMPAT 26.5 / 42.7 / 60.9 KB/s (**2.2–3.2x** faster;
best repeat run 3.0–4.2x). Server ground truth matched in both runs
(304 pkts / 74,720 B).

End-to-end RTT ≈ **0.85 ms** — ~7x under the ESP-NOW 1 Mbps default (~6 ms).
Repeat runs land at 818–878 µs median; 0 loss in the full-set run.
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


## Same-WiFi IP mode (`link-station`) — boards find each other over a router/hotspot

For when every board joins the **same WiFi network** (e.g. a venue hotspot or
router) instead of a private AP. Flash the *identical* `link-station` image
onto any number of boards:

    # edit the two lines in platformio.ini:  -DLF_WIFI_SSID="..." -DLF_WIFI_PASS="..."
    pio run -e link-station -t upload --upload-port /dev/cu.usbmodem101    # board 1
    pio run -e link-station -t upload --upload-port /dev/cu.usbmodem1101   # board 2
    ... (same one line for every extra board)

Each board (firmware: `src/station_comm.cpp`):
- joins the configured STA network (auto-pins to the strongest matching AP),
- advertises itself via mDNS → a stable `esp32-<mac>.local` hostname and IP,
- broadcasts a UDP beacon (port 42100) every 2 s, so every board on the same
  network learns every other board's IP with zero config,
- answers UDP PING/PONG; press `H` on any board's serial to see measured RTT,
- if the configured network is missing, it self-hosts an AP named **LINKNET**
  (pw `linkfast123`). The first board to boot becomes the AP, later boards
  join it — boards can always talk to each other.

Serial commands: `S` scan nearby WiFis · `I` own IP/hostname/peers · `H` ping.

Verified on the two XIAO boards (self-hosted LINKNET, no external router):
both auto-linked (192.168.4.1 ↔ 192.168.4.2), mDNS + beacon discovery worked
in both directions, UDP RTT **5–9 ms** once warm. Over IP the round-trip is
~6–10x the 0.85 ms ESP-NOW link — for the very lowest latency use the
ESP-NOW env above; use `link-station` when boards must share a normal WiFi
network or when you need IP/mDNS addressing at all.

> Note: some guest/campus networks (e.g. NTUGUEST) segment clients — boards
> get IPs but cannot reach each other. Use your own hotspot/router and verify
> with `H` (it should answer with `rtt`).
Role-by-role setup (hub vs client), exact `platformio.ini` env snippets,
expected serial output, and troubleshooting: **[docs/STATION_SETUP.md](docs/STATION_SETUP.md)**.

