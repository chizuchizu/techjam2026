# ESP32-C3 Wi-Fi PHY tuning — APIs + sdkconfig for max speed, min latency

Scope: ESP-NOW 2-node link (esp32-linkbench). Project reality: this repo builds **Arduino
core 2.0.17 on ESP-IDF v4.4.7** (verified via `tools/sdk/versions.txt` and the linked firmware
`.map`). Consequences: (a) enum names follow IDF 4.4/5.x (`WIFI_BW_HT40`,
`ESP32_WIFI_AMPDU_..._ENABLED`); (b) the ESP-NOW rate API here is
`esp_wifi_config_espnow_rate()` (linked in the lib; `esp_now_set_peer_rate_config()` is IDF
5.2+ and NOT linked); (c) the Wi-Fi driver is a prebuilt static lib with a **fixed
sdkconfig** — those knobs only bite when building the ESP-IDF framework instead of Arduino.

## APIs (esp_wifi.h / esp_now.h)

**1) esp_wifi_set_protocol** — header `esp_wifi/include/esp_wifi.h`
`esp_err_t esp_wifi_set_protocol(wifi_interface_t ifx, uint8_t protocol_bitmap);`
Bits: `WIFI_PROTOCOL_11B=1, 11G=2, 11N=4, LR=8` (esp_wifi_types.h). Default 11B|11G|11N.
C3: full b/g/n + LR (11AX is C6+). Call after `esp_wifi_init()`, **before** `esp_wifi_start()`;
re-calling while connected forces reconnect. Applies to ESP-NOW frames (normal 802.11 on the
same ifx). Low latency: **yes** — drop 11B (1 Mbps airtime), keep 11N; skip
`WIFI_PROTOCOL_LR` (>100 m only, slower). 11G-only + 11B off is a clean option.
https://docs.espressif.com/projects/esp-idf/en/stable/esp32c3/api-reference/network/esp_wifi.html
and driver guide v5.4.4: https://docs.espressif.com/projects/esp-idf/en/v5.4.4/esp32c3/api-guides/wifi.html

**2) esp_wifi_set_bandwidth — `esp_err_t esp_wifi_set_bandwidth(wifi_interface_t ifx, wifi_bandwidth_t bw);`**
IDF 4.4/5.x enum: `WIFI_BW_HT20`, `WIFI_BW_HT40` (IDF6 renamed to `WIFI_BW20/WIFI_BW40`).
**C3 fact/trap:** C3 *does* support HT40 — default is HT40, datasheet quotes 150 Mbps PHY
(HT40-MCS7) vs 72 Mbps (HT20) — but no HT20/40 coexist; in STA mode HT40 only survives if the
AP also negotiates it (linkbench: set identically on both nodes). Espressif recommends HT20
in crowded RF because HT40 degrades there. Guard interval has no separate knob — SGI lives in
the PHY-rate enum (§3). Low latency: **conditional** — HT40 helps throughput, not per-frame
latency; prefer HT20 for retry robustness in a noisy bench. Cite: ESP32-C3 "Wi-Fi HT20/40",
v6.0.2: https://docs.espressif.com/projects/esp-idf/en/v6.0.2/esp32c3/api-guides/wifi-driver/wifi-mac-protocols.html

**3) TX rate — two different APIs**
`esp_err_t esp_wifi_config_80211_tx_rate(wifi_interface_t ifx, wifi_phy_rate_t rate)`
(esp_wifi.h): fixes the rate for the interface's normal 802.11 TX; called between
`esp_wifi_init()` and `esp_wifi_start()`; IDF6 forbids it under 11A/AC/AX (C3 is b/g/n, fine).
`esp_err_t esp_wifi_config_espnow_rate(wifi_interface_t ifx, wifi_phy_rate_t rate)`
(esp_now.h, IDF 4.4–5.4; deprecated in 5.4+, removed on master): **the API that governs
ESP-NOW action frames**; call *after* `esp_wifi_start()`. Linked in this firmware. IDF 5.2+
replacement: `esp_now_set_peer_rate_config(peer_addr, esp_now_rate_config_t *)` (per-peer;
fields `phymode, rate, ersu, dcm`). **ESP-NOW defaults to 1 Mbps** — the biggest latency
killer; must be overridden. Enum: `WIFI_PHY_RATE_MCS0..7_LGI` (0x10–0x17) and
`..._SGI` (0x18–0x1F); fastest = **`WIFI_PHY_RATE_MCS7_SGI`** (HT40 ~150 / HT20 ~72 Mbps).
Uncertain: whether `esp_wifi_config_80211_tx_rate` alone affects ESP-NOW frames (reports
vary) — use the espnow-rate API to be unambiguous. Low latency: **yes, biggest single win.**
https://docs.espressif.com/projects/esp-idf/en/stable/esp32c3/api-reference/network/esp_now.html
and esp-idf source `components/esp_wifi/include/esp_now.h` (v4.4.7/v5.4).

**4) TX power — `esp_err_t esp_wifi_set_max_tx_power(int8_t power);` / `esp_wifi_get_max_tx_power(int8_t *power);`**
**Unit trap: NOT dBm.** Unit is **0.25 dBm**, range [8,84] → 2–20 dBm, via coarse mapping
(80 → 20 dBm; 8–19 → 2 dBm). Call after `esp_wifi_start()`. Only matters for range; keep
default for linkbench (co-located). Low latency: **usually no**.

**5) Power save + sdkconfig**
`esp_err_t esp_wifi_set_ps(wifi_ps_type_t type);` — default is **WIFI_PS_MIN_MODEM**; call
`esp_wifi_set_ps(WIFI_PS_NONE)` after start to kill modem-sleep wake latency on STA-mode
ESP-NOW. Low latency: **yes**.
sdkconfig (IDF 4.4 names = what this project ships; IDF5+ prefix `CONFIG_ESP_WIFI_*`,
AMPDU suffix `_ENABLED`):
* `ESP32_WIFI_STATIC_RX_BUFFER_NUM` (=8): HW DMA RX bufs (~1.6 KB each); raise for burst RX; not <6 while AMPDU is on.
* `ESP32_WIFI_AMPDU_TX_ENABLED` / `AMPDU_RX_ENABLED` (real names end `_ENABLED`, not `_ENABLE`): see §6.
* `ESP32_WIFI_TX_BUFFER` (static/dynamic, type 0/1): dynamic default; static reduces alloc latency.
* `ESP32_WIFI_IRAM_OPT` / `RX_IRAM_OPT`: put Wi-Fi code in IRAM → fewer flash-cache stalls on TX path.
* `CONFIG_ESP_TIMER_IMPL_SYSTIMER`: the driver schedules retries/beacons via esp_timer; keep enabled.
All compile-time; in Arduino they are baked into the prebuilt lib — use the `espidf` framework
to tune. Cite: v5.4.4 wifi guide (Wi-Fi Buffer Usage / Menuconfig) URL above.

**6) AMPDU/AMSDU vs tiny-frame latency**
Espressif's C3 doc, verbatim: AMPDU "can greatly improve the Wi-Fi throughput. Generally,
the AMPDU should be enabled. **Disabling AMPDU is usually for debugging purposes.**" No
official latency guidance favors disabling it; tiny ESP-NOW action frames go out individually
anyway, so BA-window overhead is small. The sanctioned low-latency path is **QoS AC_VO**
(socket `IP_TOS` precedence 6–7) — the same doc confirms AC_VO does **not** use AMPDU and has
highest priority. C3 supports RX AMSDU only (not TX). Community "disable AMPDU to cut
latency" posts exist but are **uncertain — measure first**. Low latency: **no (keep AMPDU on;
use AC_VO instead)**.

**Highest-impact knobs (verified against this repo's IDF 4.4.7):**
1. `esp_wifi_config_espnow_rate(WIFI_IF_STA, WIFI_PHY_RATE_MCS7_SGI)` after start — 1 Mbps default → up to 72/150 Mbps per frame.
2. `esp_wifi_set_ps(WIFI_PS_NONE)` — kills default MIN_MODEM wake latency.
3. `esp_wifi_set_protocol(STA, 11B|11G|11N)` before start — removes 1 Mbps 11B basics.
HT40: supported on C3 but a throughput/robustness trade in noise; both peers must match.
