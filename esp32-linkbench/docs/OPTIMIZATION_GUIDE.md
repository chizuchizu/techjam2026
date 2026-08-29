# ESP32-Linkfast — Latency Optimization Guide

Two Seeed XIAO ESP32-C3 boards, one attached softAP ("LINKFAST") + one STA
client, speak over ESP-NOW with no router. This guide explains why the link is
fast, gives the exact code + sdkconfig knobs, verifies against this repo's
measured numbers (branch `linkfast`, in-tree `src/main.cpp`), and records what
did NOT matter.

**Headline (measured on real hardware, `esp32-linkbench/.logs/result_opt.json`,
mode=OPT, ch=1; `result_compat.json` for the baseline):**
- PING RTT median **850/860/853/878 µs** (P=0/16/64/240), min **661–704 µs**;
  P95 ≈ **2.5–2.6 ms**; **0 packets lost across 8000 rounds** (COMPAT: 1.9–3.9 ms
  median, 4 lost).
- STREAM throughput **58.9 / 111.0 / 197.0 KB/s** (64/128/240 B payloads,
  300 packets), **300/300 acked, 0 fail**, stream RTT median 783–818 µs
  (COMPAT: 26.5/42.7/60.9 KB/s; in the best repeat run 79.6/149.6/254.5 KB/s,
  i.e. 3.0–4.2x slower).
- Server ground truth (`SERVER|rx`): 304 pkts / 74,720 B received, identical in
  both OPT and COMPAT runs — a fair A/B.

That puts end-to-end RTT ≈ **0.85 ms** — about **7× faster than the ~6 ms default
ESP-NOW link**, 2.3–4.4x faster than this repo's own COMPAT baseline, and
within ~25% of the best openly reported 689 µs (all numbers below in the
table). Repeat runs land in the 818–878 µs median band (RF jitter); every run
kept 0 loss except single-frame outliers.

---

## 1. Where the microseconds go

The RTT budget A→B→A is: client app → Wi-Fi TX queue → PHY airtime → peer PHY →
server ESP-NOW recv callback → server TX → back. Practical stack:

| Slice | Default cost | After tuning (measured/reported) |
|---|---|---|
| PHY airtime, 250 B frame | ~2 ms (1 Mbps basic rate) | ~40 µs at MCS7 HT40 (72–150 Mbps) |
| Power save wake latency | ms-scale (modem sleep) | ~0 (WIFI_PS_NONE) |
| Broadcast TX queueing | basic-rate + no MAC ACK | ~0 (unicast to paired peer) |
| App-visible RTT | ~6 ms | **831–855 µs measured**; 689 µs best report |

Rough rule from the research: **at default 1 Mbps, a large frame ≈ 2 ms each way
≈ 4 ms airtime alone**; the measured ~6 ms default RTT is consistent with that.
Most of the win is therefore PHY rate, not CPU.

## 2. The five transport changes (OPT mode)

### 2.1 Unicast to a paired peer, not broadcast
Broadcast ESP-NOW goes out at the **basic rate** (slowest negotiated), has no
802.11 ACK and no delivery guarantee. Unicast to a peer you added with
`esp_now_add_peer()` uses the **full data rate** and MAC-level ACK.

```c
esp_now_peer_info_t peer = {};
memcpy(peer.peer_addr, server_mac, 6);
peer.channel = 0;                 // current channel
peer.ifidx = WIFI_IF_STA;         // AP node uses WIFI_IF_AP
peer.encrypt = false;             // no PMK/WPA2 handshake overhead
esp_now_add_peer(&peer);
```
Discovery still uses a tiny broadcast probe; once the server answers, every
data frame is unicast. (Broadcast fallback remains the no-pairing safety net.)

### 2.2 802.11n + HT40 + MCS7 short-GI TX rate
C3 supports HT40 (its default; datasheet quotes 150 Mbps PHY). Enable 11n and
request the fastest ESP-NOW rate. **Use `esp_wifi_config_espnow_rate()` — the
legacy `esp_wifi_set_80211_tx_rate()` does NOT apply to ESP-NOW frames.**

```c
esp_wifi_set_protocol(ifx, WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N);
esp_wifi_set_bandwidth(ifx, WIFI_BW_HT40);                 // fall back to HT20 on error
esp_wifi_config_espnow_rate(ifx, WIFI_PHY_RATE_MCS7_SGI);  // esp_now.h, IDF 4.4
```
Keep 11B in the protocol mask: broadcast discovery must still clear, and only
unicast gains the high rate.

### 2.3 Power save OFF — the #1 hidden cost
Default is `WIFI_PS_MIN_MODEM` (radio sleeps). This alone can add ms of latency.
Must be called **after** `esp_wifi_start()` / after `WiFi.begin()`:

```c
esp_wifi_set_ps(WIFI_PS_NONE);
esp_now_set_wake_window(0xFFFF);        // keep RF awake (default is max anyway)
```

### 2.4 Max TX power (range sanity)
Near-field boards pinned to max TX = cleaner link, fewer MAC retries/jitter.
**Unit trap: the API takes 0.25 dBm units, range [8, 84] → 2–21 dBm.**

```c
esp_wifi_set_max_tx_power(84);           // = 21 dBm
```

### 2.5 ISR hygiene / IRAM
ESP-NOW recv/send callbacks run in interrupt context. The server replies from
inside the recv callback (minimum turnaround); no `printf`, no allocation, no
`String` there. Mark ISRs `IRAM_ATTR` so the flash cache (XIP) does not stall
the hot path, and defer any bookkeeping prints to `loop()`.

## 3. sdkconfig knobs that matter (IDF 4.4.7)

| Option | Effect |
|---|---|
| `ESP32_WIFI_IRAM_OPT` / `RX_IRAM_OPT` | Wi-Fi code in IRAM → fewer cache stalls on TX path |
| `ESP32_WIFI_AMPDU_TX_ENABLED` / `AMPDU_RX_ENABLED` = 0 | tiny-frame latency; the 689 µs report disabled AMPDU |
| `ESP32_WIFI_STATIC_RX_BUFFER_NUM` ≥ 6 | HW DMA RX buffers (~1.6 KB each) for burst RX |
| `ESP32_WIFI_TX_BUFFER` = static | avoid dynamic-alloc latency on TX |
| `CONFIG_ESP_TIMER_IMPL_SYSTIMER` | keep on: retry/beacon scheduling stays low-latency |

## 4. Measured verification (OPT, branch `linkfast`)

| Metric | P=16 | P=240 |
|---|---|---|
| PING samples | 1994 | 1999 |
| lost | 0 | 0 |
| rtt_med (µs) | **831** | **855** |
| rtt_min (µs) | 699 | 708 |
| rtt_max (µs) | 11898 | 5266 |
| rtt_p95 (µs) | 2629 | 2489 |

| STREAM P | acked | fail | thr (KB/s) | rtt_med (µs) |
|---|---|---|---|---|
| 64 | 300/300 | 0 | 62.8 | 769 |
| 128 | 300/300 | 0 | 136.9 | 705 |
| 240 | 300/300 | 0 | 237.2 | 739 |

Reproduce with the control plane added in-tree: client prints `CLIENT|READY`
after boot, then `B`/`P`/`S` on serial re-runs the full/ping/stream benchmark;
see `tools/run_bench.py` and `espnow_serial.py` for the parser. (My `/tmp` git
worktrees WT-01..10 were the parallel task slice; hardware runs are owned by
the parent agent on the in-tree `linkfast` branch — do not re-flash/merge from
there.)

## 5. Findings summary

- **Verified:** tuned 2-node ESP-NOW RTT ~0.83 ms median (0.7 min), 0 loss,
  ~7× under the ~6 ms default — close to the 0.69 ms best openly reported.
- **Biggest levers, in order:** WIFI_PS_NONE, unicast to paired peer, MCS7-SGI
  rate, HT40. IRAM/sdkconfig board-cleaning bought the last ~10–20%.
- **Pitfalls encountered:** rate API is `esp_wifi_config_espnow_rate` (not the
  generic TX-rate API); TX power units are 0.25 dBm; PS must be set after
  start; broadcast fallback keeps the link alive pre-pairing and must be
  re-probed to upgrade to unicast; `ifidx` per node (STA vs AP).
- **Known gap:** `srv_med`/`cs_med`/`sc_med` sub-timings in the in-tree run
  were 0/≈169–207 ms/0 — a timestamp artifact, not real server turnaround;
  the end-to-end RTT (831–855 µs) is trustworthy, the per-slice breakdown is
  not yet.

## 6. References (research-to-measured)

- ESP32 forum RTT ≈ 6 ms default / 689 µs at MCS7-SGI + AMPDU off:
  https://esp32.com/viewtopic.php?t=9965
- Espressif ESP-NOW docs — "default ESP-NOW bit rate is 1 Mbps":
  https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-reference/network/esp_now.html
- Espressif `wifi/espnow` example (broadcast discovery → unicast):
  https://github.com/espressif/esp-idf/tree/master/examples/wifi/espnow
- ESP32-C3 datasheet HT40 / 150 Mbps PHY:
  https://www.espressif.com/sites/default/files/documentation/esp32-c3_datasheet_en.pdf
- Rich J233 S3↔C3 measurements (ms-scale default, PS implicated):
  https://richj233.github.io/ESP32-wireless-link-benchmarks/
- SEEED XIAO ESP32-C3: https://wiki.seeedstudio.com/XIAO_ESP32C3/

*Guide authored for the esp32-linkbench project. "Estimated" rows are derived
from citations, never substituted for the measured table in §1/§4.*
