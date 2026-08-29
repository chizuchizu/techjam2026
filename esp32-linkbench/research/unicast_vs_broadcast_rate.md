# Unicast vs Broadcast ESP-NOW on 2.4 GHz: Rate, Airtime, Latency

Context: `esp32-linkbench` compares two transports on 2x Seeed XIAO ESP32-C3:
COMPAT (`-DLF_COMPAT=1`, all sends to the `BROADCAST` MAC) vs OPT (unicast to a
paired peer, `WIFI_PHY_RATE_MCS7_SGI`, HT40, `WIFI_PS_NONE`).
**All numbers are ESTIMATES** computed from IEEE 802.11 PHY parameters; the
linkbench bench runs are the measured source of truth.

## 1. Broadcast/multicast at the basic rate, unicast at the data rate

802.11 has no per-receiver rate feedback for group traffic. Multicast frames
must be decodable by the slowest associated station, so RFC 9119 defines the
**basic rate as "the slowest rate of all the connected devices at which
multicast and broadcast traffic is generally transmitted"** and notes there
are **no ACKs for multicast packets** (no retransmission, no delivery
knowledge), so rate adaptation cannot apply — "more than 3 orders of
magnitude" below optimal unicast forwarding. Juniper's Mist rate docs state
the same: "the minimum basic rate (MBR) is the rate at which beacons, probes,
management, control, broadcast and multicast frames are sent", and warn that
the 802.11b 1 Mbps rate costs considerable capacity. Unicast instead gets
rate adaptation up to MCS7 (65/72.2 Mbps HT20, 135/150 Mbps HT40, per the
Espressif rate table).

## 2. ESP-NOW specifics (Espressif)

- Espressif ESP-IDF ESP-NOW docs: **"The default ESP-NOW bit rate is 1 Mbps."**
  ESP-NOW rides standard 802.11 PHY timing (a vendor-specific action frame;
  optional CCMP encryption).
- Broadcast ESP-NOW has **no peer entry, no 802.11 ACK, and no delivery
  confirmation** — fire-and-forget at the default low rate.
- Unicast to a **paired** peer (`esp_now_add_peer`) can be sent faster:
  ESP-IDF exposes `esp_now_set_peer_rate_config()` (per-peer) and the
  interface-wide `esp_wifi_config_80211_tx_rate()` (11b/g/n only). `linkbench`
  OPT uses the latter with `WIFI_PHY_RATE_MCS7_SGI` (best effort — drivers may
  still arbitrate on range).
- Espressif's `wifi/espnow` example broadcasts only to discover the peer, then
  switches to unicast — the same pattern as linkbench.

## 3. Airtime for a 250-byte frame (1 SS, 2.4 GHz)

Frame = 250 B payload + 24 B MAC header + 4 B FCS = 278 B MPDU (2246 PSDU
bits incl. SERVICE+TAIL). Preamble/hdr: legacy 20 us (OFDM), 802.11b long
192 us, HT-mixed 36 us (1 spatial stream).

| PHY mode (TX rate)        | Preamble+PHY hdr | Data symbols | PPDU airtime |
|---------------------------|------------------|--------------|--------------|
| 802.11b 1 Mbps (broadcast basic rate) | 192 us | 2438 us | **2438 us (2.44 ms)** |
| 802.11b 2 Mbps            | 192 us          | 1123 us      | 1315 us |
| 802.11b 5.5 Mbps          | 192 us          | 408 us       | 600 us |
| 802.11b 11 Mbps           | 192 us          | 204 us       | 396 us |
| OFDM 6 Mbps (11g basic rate) | 20 us        | 376 us       | 396 us |
| OFDM 54 Mbps              | 20 us           | 44 us        | 64 us |
| HT MCS7, 20 MHz, LGI 65 Mbps | 36 us        | 36 us        | 72 us |
| HT MCS7, 20 MHz, SGI 72 Mbps | 36 us        | 32.4 us      | **68 us** |
| HT MCS7, 40 MHz, LGI 135 Mbps | 36 us       | 20 us        | 56 us |
| HT MCS7, 40 MHz, SGI 150 Mbps | 36 us       | 18 us        | **54 us** |

Unicast also pays **SIFS (10 us) + an 802.11 ACK** per data frame: ~44 us at
the 6 Mbps basic rate (~28 us at 24 Mbps), up to ~326 us if forced to 1 Mbps.
Broadcast pays no ACK and no retransmission (no recovery on loss).

## 4. Conclusion: unicast vs broadcast impact (estimates)

One-way PHY airtime for 250 B: **~2.44 ms broadcast (1 Mbps) vs ~54-68 us
unicast MCS7 HT40/HT20-SGI — ~36-45x less**. Full request+response over air
(incl. SIFS+ACK and ~96 us DIFS+backoff per frame): broadcast ~5.1 ms vs
unicast ~0.4 ms — **~12-13x lower RTT**. Back-to-back one-direction payload
throughput (250 B, no A-MPDU): broadcast ~0.8 Mbps vs unicast MCS7-HT40-SGI
~9.8 Mbps (HT20 ~9.2 Mbps) — **~12x higher**. So this bench should see ping
RTT drop by roughly an order of magnitude and stream throughput rise ~10x on
air. Caveats: fixed high MCS trades range for speed (use MCS5-7 per RSSI in
the field), `esp_wifi_config_80211_tx_rate` is best-effort, and ESP32-C3
firmware caps real goodput well below PHY rates — trust the linkbench measured
numbers, not these estimates.

## Sources

- RFC 9119 "Multicast Considerations over IEEE 802 Wireless Media" (IETF):
  https://datatracker.ietf.org/doc/rfc9119/
- Juniper Mist — Wi-Fi Data Rate Configuration (MBR; broadcast/multicast at
  minimum basic rate): https://www.juniper.net/documentation/us/en/software/mist/mist-wireless/topics/ref/mist-data-rates.html
- Espressif — ESP-NOW API docs ("default ESP-NOW bit rate is 1 Mbps";
  unicast/broadcast; `esp_now_set_peer_rate_config`):
  https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/network/esp_now.html
- Espressif — Wi-Fi API docs (HT20/HT40 MCS + LGI/SGI rate table,
  `esp_wifi_config_80211_tx_rate`):
  https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/network/esp_wifi.html
- Espressif — `wifi/espnow` example (broadcast discovery then unicast):
  https://github.com/espressif/esp-idf/tree/master/examples/wifi/espnow
- Omada — "How to Configure Multicast/Broadcast Rate Limit" (airtime cost of
  group traffic): https://support.omadanetworks.com/en/document/112038
- Shankar Wi-Fi — PPDU Formats / preamble duration reference:
  https://shankarwifi.com/ppdu-formats/
- embeddedcalc — Wi-Fi data-rate/airtime calculator (sanity check):
  https://embeddedcalc.com/tools/wifi-data-rate-calculator/
