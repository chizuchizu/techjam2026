# ESP-NOW Pairing + Channel-Lock Best Practices for a 2-Board ESP32-C3 Link

**Recommended recipe (no external router, lowest stable latency):** one board runs **SoftAP** (channel authority); the second runs **Station and associates to that SoftAP**. Both stay on the AP's fixed channel and ESP-NOW unicast rides it. Sender and receiver being on the same channel is the only hard requirement at the frame level.

## 1) Two unassociated STAs on the same channel?

Workable, but only if you force the channel yourself and nothing moves it. Espressif's own `examples/wifi/espnow` runs **both** devices as STAs, never connects to an AP, and pins the channel with `esp_wifi_set_channel(CONFIG_ESPNOW_CHANNEL, WIFI_SECOND_CHAN_NONE)` after `esp_wifi_start()` ([example main.c](https://github.com/espressif/esp-idf/blob/master/examples/wifi/espnow)). ESP-NOW works connectionless with no router ([ESP-FAQ](https://docs.espressif.com/projects/esp-faq/en/latest/application-solution/esp-now.html)). The risk is **channel drift**: once connected to Wi-Fi a device "can only transmit and receive data on the current Wi-Fi channel" ([ESP-FAQ](https://docs.espressif.com/projects/esp-faq/en/latest/application-solution/esp-now.html)), but an *unassociated* STA has no anchor — any scan, P2P, or connect-elsewhere call moves the channel and the link dies. Unassociated STAs also fall under **disconnected-state sleep** (`CONFIG_ESP_WIFI_STA_DISCONNECTED_PM_ENABLE`, on by default [per IDF](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-guides/wifi-driver/wifi-performance-and-power-save.html)); the default max wake-window keeps RF on, but it is subtle.

## 2) Most robust pairing pattern

(a) **SoftAP + connected STA** — best for a router-less 2-board link. Association locks the STA's channel and it follows the AP if the AP ever moves; TX/RX windows stay TBTT-aligned; no scanning in steady state. Arduino-ESP32 docs ship exactly this and note devices "can be in different modes (AP or Station)... the only requirement is that the devices are on the same Wi-Fi channel" ([Arduino ESP-NOW docs](https://docs.espressif.com/projects/arduino-esp32/en/latest/api/espnow.html)).

(b) **Both STAs on an existing AP** — most robust when a router exists: IDF says both stations on the **same** AP align connectionless wake intervals to TBTT and "help align the connectionless modules transmission window", improving reception ([Wi-Fi Performance & Power Save](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-guides/wifi-driver/wifi-performance-and-power-save.html)). Not our scenario.

(c) **Both AP on the same channel** — works in principle but two beacon masters contend over one channel; least documented config; avoid. *(uncertain: no official reference config)*

Pick **AP+STA** for this bench.

## 3) `esp_now_add_peer()` and ifidx gotchas

`esp_now_peer_info_t` fields: `peer_addr`, `lmk`, `channel`, `ifidx`, `encrypt`, `priv` ([API ref](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/network/esp_now.html)). `channel` 0 means "send on the current channel"; otherwise it "must be set as the channel that the local device is on" — never set a foreign channel. `ifidx` is the **local** interface ESP-NOW uses for that peer: `WIFI_IF_STA` for the STA node, `WIFI_IF_AP` for the AP node ("you can send ESP-NOW data via both the Station and the SoftAP interface"). `encrypt = false` sends an unencrypted vendor action frame (no LMK) — leave it off for a latency bench. A **broadcast peer** (`FF:FF:FF:FF:FF:FF`) "must be added before sending broadcast data". The receiver needs no peer entry for unencrypted unicast/broadcast, but must add the peer (matching LMK) to receive encrypted unicast.

**ifidx mismatch is not silent:** `esp_now_send()` returns `ESP_ERR_ESPNOW_IF` ("current Wi-Fi interface doesn't match that of peer") and `ESP_ERR_ESPNOW_CHAN` on channel mismatch — always check the return ([API ref](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/network/esp_now.html)). Classic bugs: using the STA MAC of an AP-only node (SoftAP MAC differs; use `esp_wifi_get_mac(WIFI_IF_AP)`), or sending before the interface is started.

## 4) Cutting jitter

- **Disable modem-sleep** (default `WIFI_PS_MIN_MODEM`): it adds up to a DTIM/listen-interval delay to RX. `esp_wifi_set_ps(WIFI_PS_NONE)` "minimizes the delay... in real time" ([IDF](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-guides/wifi-driver/wifi-performance-and-power-save.html)). Measured: 8.2 ms RTT high-jitter → jitter gone after `WIFI_PS_NONE` ([jcallano](https://github.com/jcallano/ESPNOW-Latency-test-for-turnaround-comm-betwen-esp32S3-and-esp32C6)).
- **Keep wake-window max (default)** so RF is always on ([API ref](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/network/esp_now.html)).
- **Fix channel, never scan** after startup.
- **Raise the PHY rate**: default 802.11b 1 Mbps ≈ 2 ms airtime for 250 B. Set per-peer rate via `esp_now_set_peer_rate_config()` (IDF ≥5.2; earlier `esp_wifi_config_espnow_rate()`) and/or ban 11b: `esp_wifi_set_protocol(..., WIFI_PROTOCOL_11G|WIFI_PROTOCOL_11N)`. Same source: 8.2 ms → 3.05 ms RTT.
- **Send one packet per send-callback** ("too short an interval may lead to disorder of sending callback"); never do heavy work in the Wi-Fi callback — post to a queue ([API ref](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/network/esp_now.html)).
- Do **not** enable Long-Range (forced 1 Mbps).

Reference numbers: dual-S3 ESP-NOW board RTT mean ~4.4 ms / P99 ~12 ms ([RichJ233](https://richj233.github.io/ESP32-wireless-link-benchmarks/)); error rate rises with distance and encryption — add application ACK/retry and prefer **unicast** to broadcast for reliability ([Hackaday log](https://hackaday.io/project/164132-hello-world-for-esp-now/log/160572-latency-and-reliability-testing/)).

## Concrete recipe (ESP-IDF C, ESP32-C3, channel 6)

**AP node (master):**
```c
esp_wifi_set_mode(WIFI_MODE_AP);
esp_wifi_set_config(WIFI_IF_AP, &(wifi_config_t){
    .ap = { .ssid="link", .ssid_len=4, .channel=6, .max_connection=4 } });
esp_wifi_start();
esp_now_init();
esp_now_peer_info_t peer = {0};
memcpy(peer.peer_addr, STA_PEER_MAC, 6); // STA node's STA MAC
peer.channel = 0;                        // use current (AP) channel
peer.ifidx   = WIFI_IF_AP;               // AP node's local interface
peer.encrypt = false;
esp_now_add_peer(&peer);
```

**STA node (client):**
```c
esp_wifi_set_mode(WIFI_MODE_STA);
esp_wifi_connect();                      // associates -> channel locked to 6
while (!connected_to_ap()) { vTaskDelay(pdMS_TO_TICKS(50)); }
esp_wifi_set_ps(WIFI_PS_NONE);           // no modem-sleep
esp_now_init();
esp_now_peer_info_t peer = {0};
memcpy(peer.peer_addr, AP_PEER_MAC, 6);  // AP node's SOFTAP MAC
peer.channel = 0;
peer.ifidx   = WIFI_IF_STA;
peer.encrypt = false;
esp_now_add_peer(&peer);
```
Arduino equivalent: `WiFi.setSleep(false)`; broadcast peer `ESP_NOW_Peer(ESP_NOW.BROADCAST_ADDR, 6, WIFI_IF_STA, nullptr)`; unicast peer with the partner's **interface-correct** MAC ([Arduino docs](https://docs.espressif.com/projects/arduino-esp32/en/latest/api/espnow.html)).

## Pitfalls checklist

- AP/STA MAC confusion: use the MAC of the interface that carries ESP-NOW (SoftAP MAC for the AP node).
- Not checking `esp_now_send()` return (`ESP_ERR_ESPNOW_IF` / `ESP_ERR_ESPNOW_CHAN`).
- Scanning, DHCP, or reconnect after startup → channel change → dead link.
- Modem-sleep or Long-Range adds latency/jitter.
- Broadcast is less reliable than unicast; add ACK + sequence numbers for real-time links.

## Sources

- [ESP-IDF ESP-NOW API reference](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/network/esp_now.html)
- [ESP-FAQ: ESP-NOW](https://docs.espressif.com/projects/esp-faq/en/latest/application-solution/esp-now.html)
- [IDF Wi-Fi Performance & Power Save](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-guides/wifi-driver/wifi-performance-and-power-save.html)
- [Arduino-ESP32 ESP-NOW docs](https://docs.espressif.com/projects/arduino-esp32/en/latest/api/espnow.html)
- [IDF examples/wifi/espnow](https://github.com/espressif/esp-idf/tree/master/examples/wifi/espnow)
- [jcallano ESP-NOW latency optimization](https://github.com/jcallano/ESPNOW-Latency-test-for-turnaround-comm-betwen-esp32S3-and-esp32C6)
- [RichJ233 ESP32 link benchmarks](https://richj233.github.io/ESP32-wireless-link-benchmarks/)
- [Hackaday ESP-NOW latency/reliability log](https://hackaday.io/project/164132-hello-world-for-esp-now/log/160572-latency-and-reliability-testing/)
- [ESP32 Forum: ESP-NOW RTT & bit-rate](https://esp32.com/viewtopic.php?t=9965)
