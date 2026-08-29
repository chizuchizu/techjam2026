# ESP-NOW minimum achievable latency on ESP32 / ESP32-C3

Research summary for the esp32-linkbench project. **As reported** = measured by the cited source; **estimated** = derived by this author, not bench-measured. RTT = round trip A→B→A.

## 1. Reported measured numbers

| Setup | Value | Type | Source |
|---|---|---|---|
| ESP32 board-to-board, 250-byte packet, default Arduino SDK config | avg RTT ≈ 6000 µs; same test < 1000 µs on ESP8266 | reported, measured | https://esp32.com/viewtopic.php?t=9965 (wayback: web.archive.org/web/20210617180546/https://www.esp32.com/viewtopic.php?t=9965) |
| ESP32, forced `WIFI_PHY_RATE_MCS7_SGI` + AMPDU TX/RX disabled | RTT ≈ 689 µs (1000 samples) | reported, measured | same thread |
| ESP32-S3 ↔ ESP32-C3 one-way, channel 6, power save disabled | S3→C3 ≈ 3.3–3.5 ms; C3→S3 ≈ 3.1–7 ms | reported, measured | https://richj233.github.io/ESP32-wireless-link-benchmarks/ |
| S3↔C3 stop-and-wait RTT | median 4.374 ms, P95 10.910 ms, max 25.573 ms | reported, measured | same |
| Dual ESP32-S3 radio RTT (timed in-MCU) | min 2.973 ms, median 3.602 ms, mean 4.442 ms, P99 12.363 ms | reported, measured | same |
| MicroPython ESP-NOW (ESP32 family) | successful send ACK ≤ ~2 ms typical; no-ACK timeout ≈ 25 ms after ~a dozen retransmits; echo RTT 5–20 ms typical | reported, measured | https://github.com/orgs/micropython/discussions/11757 |
| ESP-NOW broadcast, 100-byte packets at MCS7 SGI | ≈ 1275–1300 fps (≈ 0.77 ms per frame air/slot, not app RTT) | reported, measured | https://github.com/nanshenwei/espnowSpeedTest |

**Estimated:** the default cost is PHY rate. ESP-NOW defaults to 1 Mbps on air (Espressif FAQ), so a 250-byte frame ≈ 2 ms one-way ≈ 4 ms RTT in air — matching the forum's ~6 ms default. At MCS7 HT20/HT40 (72/150 Mbps) the same frame ≈ 40 µs airtime, consistent with the reported 689 µs RTT.

## 2. Latency knobs (exact APIs)

- **Power save / modem sleep — first check.** Default `WIFI_PS_MIN_MODEM`; call `esp_wifi_set_ps(WIFI_PS_NONE)`. In STA, also disable `CONFIG_ESP_WIFI_STA_DISCONNECTED_PM_ENABLE`. (ESP-IDF Wi-Fi API + ESP-FAQ.) Opposite (power-saving) knobs: `esp_now_set_wake_window()`, `esp_wifi_connectionless_module_set_wake_interval()`.
- **TX rate.** `esp_now_set_peer_rate_config(peer, &esp_now_rate_config_t)` (IDF ≥ v5.2, per peer) or legacy `esp_wifi_config_espnow_rate(WIFI_IF_STA, WIFI_PHY_RATE_MCS7_SGI)`. Private fast path used for the 689 µs report: `esp_wifi_internal_set_fix_rate(WIFI_IF_STA, 1, WIFI_PHY_RATE_MCS7_SGI)` with `#include "esp_private/wifi.h"` and `CONFIG_ESP_WIFI_AMPDU_TX_ENABLED`/`CONFIG_ESP_WIFI_AMPDU_RX_ENABLED` disabled.
- **802.11n + HT40.** `esp_wifi_set_protocol(WIFI_IF_STA, WIFI_PROTOCOL_11B|WIFI_PROTOCOL_11G|WIFI_PROTOCOL_11N)`; `esp_wifi_set_bandwidth(WIFI_IF_STA, WIFI_BW_HT40)` (HT40 needs 11N). ESP32-C3 supports 20/40 MHz, MCS0–7, 150 Mbps (C3 datasheet §4.1/§6).
- **Channel.** Peers share one channel: `peer_info.channel` in `esp_now_add_peer()`, and/or `esp_wifi_set_channel()`. A quiet channel improves frame rate (espnowSpeedTest).
- **Unicast vs broadcast.** With a unicast peer the send callback returns `ESP_NOW_SEND_SUCCESS` only after the MAC ACK — direct latency feedback. Broadcast has no per-peer ACK; Hackaday notes unicast may be more reliable.
- **Callbacks / IRAM / CPU.** `esp_now_register_send_cb()`/`esp_now_register_recv_cb()` run on the high-priority Wi-Fi task; keep them short, echo inside the recv callback. Enable `CONFIG_ESP_WIFI_IRAM_OPT`/`CONFIG_ESP_WIFI_RX_IRAM_OPT`, mark callbacks `IRAM_ATTR`, run 160 MHz (`CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ`, C3 max).
- **Avoid lwIP/TCP-IP.** ESP-NOW is a one-layer protocol: vendor action frames go straight through the Wi-Fi driver, no netif/lwIP in the data path. Don't keep STA connected to an AP; if you must, AP channel must equal ESP-NOW channel.

## 3. Bottlenecks

- Default 1 Mbps PHY → ≈ 2 ms/frame airtime for large frames (estimated; forum).
- 802.11 fixed overhead: preamble + SIFS + ACK ≈ tens of µs per frame even at high rate (estimated).
- Single send queue: sending before the previous send-cb returns causes callback disorder (IDF docs) and, under load, `ESP_ERR_ESPNOW_NO_MEM`/drops (richj233 75 Hz test). Drop-oldest pacing fixed it.
- Retries/CSMA: ms-scale backoff per retry; ≈ 25 ms timeout after ~a dozen retransmits (MicroPython).
- Wi-Fi modem sleep adds wake latency and jitter.

## 4. Recommendations for a 2-board ESP32-C3 ultra-low-latency link

1. Both boards: STA, not connected; `esp_wifi_set_ps(WIFI_PS_NONE)`; disconnected-PM off.
2. Force MCS7 + HT40: `esp_wifi_set_protocol(...11N)`, `esp_wifi_set_bandwidth(..., WIFI_BW_HT40)`, `esp_now_set_peer_rate_config(..., MCS7)`; private rate API only if needed.
3. Fixed quiet channel (1/6/11), identical `peer_info.channel`; short line-of-sight.
4. One unicast peer, `encrypt = false`, small fixed payloads (≤ 32 B).
5. Echo in the recv callback; callbacks in IRAM; no blocking/logging on the Wi-Fi task; 160 MHz.
6. One frame in flight, paced by the send callback; drop-oldest on missed deadline.
7. **Expected:** tuned median RTT ≈ 0.7–2 ms (best reported 0.689 ms ESP32, 2.97 ms min S3 public API); default config ≈ 4–6 ms. Estimated from cited reports.

## Sources

1. ESP32 Forum, "ESPNOW Slower than expected RTT" — https://esp32.com/viewtopic.php?t=9965
2. RichJ233, ESP32 wireless link latency benchmarks — https://richj233.github.io/ESP32-wireless-link-benchmarks/
3. MicroPython #11757, "Characterising ESPNOW" — https://github.com/orgs/micropython/discussions/11757
4. Espressif ESP-IDF, ESP-NOW API — https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-reference/network/esp_now.html
5. Espressif ESP-FAQ, ESP-NOW — https://docs.espressif.com/projects/esp-faq/en/latest/application-solution/esp-now.html
6. espnowSpeedTest; ESP32-C3 datasheet — https://github.com/nanshenwei/espnowSpeedTest ; https://www.espressif.com/sites/default/files/documentation/esp32-c3_datasheet_en.pdf
