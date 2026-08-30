# Computer-master WiFi transport — Firecrawl research (2026-08-30)

Question: enable WiFi comms between 2x ESP32-C3 boards with the computer as the
master node, keeping SRAM/flash footprint extremely small.

## Findings

1. **ESP-NOW is the smallest/fastest ESP32<->ESP32 transport.**
   Espressif's ESP-NOW User Guide states ESP-NOW "occupies less CPU and flash
   resource" than a full Wi-Fi IP stack; it has no network/transport/session/
   presentation/application layers, supports unicast and broadcast, and
   responds "in milliseconds".
   Source: https://documentation.espressif.com/esp-now/master/User_Guide.md

2. **ESP-NOW vs standard Wi-Fi (UDP/TCP).**
   ESP-NOW is connectionless, needs no AP, and carries application data
   directly in 802.11 action frames. Wi-Fi UDP/TCP has higher per-packet CPU
   and RAM cost (lwIP + sockets) and needs an association/AP, but is what a
   normal computer can actually speak.
   Sources:
   - https://developer.espressif.com/blog/reliability-esp-now/ (ESP-NOW vs Wi-Fi)
   - https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-guides/wifi-driver/wifi-performance-and-power-save.html
     (LwIP/Wi-Fi peak heap depends on max connections/sockets)
   - https://www.reddit.com/r/esp32/comments/lafnlw/espnow_performance_vs_wifi_w_udp/
     (hundreds of ESP-NOW msgs/s, ack within ~2-3 ms on Arduino)

3. **ESP-NOW tuning.** Unicast peers, 802.11n + HT40, MCS7 short-GI fixed TX
   rate, and `WIFI_PS_NONE` are the latency/throughput knobs. Already
   implemented and measured in this repo (`esp32-linkfast`, median RTT 0.85 ms).
   Source: https://docs.espressif.com/projects/arduino-esp32/en/latest/api/espnow.html

4. **RAM minimisation.** ESP-IDF "Minimizing RAM Usage" guide: understand
   static vs dynamic usage, stack vs heap. Our firmware keeps all hot buffers
   static and does no allocation/printf in receive/send callbacks.
   Source: https://docs.espressif.com/projects/esp-idf/en/stable/esp32c3/api-guides/performance/ram-usage.html

## Decision

- **ESP32 <-> ESP32: ESP-NOW** (no AP, lowest CPU/flash, sub-ms, already proven
  on this exact hardware).
- **Computer (master) -> ESP32 (slave): bridge through board A's USB-CDC
  serial**, because a laptop has no ESP-NOW radio. The computer sends short
  machine-readable commands/payloads over USB; board A relays them to board B
  over ESP-NOW and streams results back. This keeps the computer as the
  logical master without adding lwIP/TCP to the tight benchmark firmware.
- **Alternative direct-WiFi path** (when the computer may join the boards'
  self-hosted AP): same-wiFi UDP using the existing `link-station` JOB/RESULT
  protocol (see docs/PC_MASTER_WIFI_BRIDGE.md). Implemented as
  `tools/wifi_master.py`.

## Constraint

This link layer is a **separate small firmware**; it does not claim to fit
WiFi inside the full-forward transformer node. Measured on this bench, the
full-sequence build (~274 KB SRAM) plus WiFi+lwIP (~85 KB static + ~69 KB heap)
overflows `dram0_0_seg`, and ESP-NOW bulk throughput is ~60 KB/s, so a 64 KB
input still does not stream faster than USB-CDC. That analysis and the open
directions live at `docs/WIFI_ON_A_COMPUTE_NODE.md`.

## Raw notes log

See /tmp/wifi_research/{q1,q2,q3b}.txt and espnow_userguide.md (session-local).
