# techjam2026 · ESP32-C3 8-board fleet — fan-out / fan-in transport research

**Scope.** 8× ESP32-C3 (XIAO C3) on one WiFi channel: minimize wall-clock for (a) coordinator → 8 workers dispatch and (b) 8 workers → coordinator result collection. Local measured baselines:
ESP-NOW unicast 2-node RTT **0.85 ms** (MCS7-SGI, HT40, `WIFI_PS_NONE`, AMPDU off); UDP unicast RTT ≈ **7 ms**; UDP 1 KB goodput ≈ **2.3 Mbps**; TCP ≈ **5.8 Mbps** (>4 KiB).

**Build verified (ground truth for our firmware, PlatformIO):** Arduino core **3.0.7 → ESP-IDF 5.1.0**; C3 headers in
`framework-arduinoespressif32-libs/esp32c3/include/esp_wifi/include/esp_now.h` (see §1, §5). Core ships the official `ESP_NOW` library under `libraries/ESP_NOW` with `ESP_NOW_Broadcast_Master/Slave` examples.

---

## Bottom line (TL;DR)

| Direction | Best transport | Expected wall-clock @ N=8 | Drop / retry implications | Pitfalls |
|---|---|---|---|---|
| **Dispatch 1→8** | **ESP-NOW true broadcast** — one frame to peer `FF:FF:FF:FF:FF:FF` (registered once) | **~0.2–0.5 ms** at MCS7-SGI/HT20 (72.2 Mbps); ~2–3 ms if left at default 1 Mbps PHY rate | Broadcast has **no 802.11 ACK and no retry** per receiver; a worker that missed the frame knows nothing → add unicast ack/retry only for workers that did not answer | Must `esp_now_add_peer` the broadcast MAC first; disable AMPDU; `WIFI_PS_NONE`; pace `esp_now_send` by the **send-complete callback** (too-close sends → `ESP_ERR_ESPNOW_NO_MEM` / `esf_buf: no eb TXQ_BLOCK=2000`) |
| Dispatch 1→8 | ESP-NOW unicast ×8 in a loop | ~8 × one-way ≈ **3–5 ms** (each ≈0.42 ms from the 0.85 ms RTT baseline + pacing gap) | Unicast gets 802.11 ACK + Wi-Fi retries → reliable per-peer delivery | Serial latency; must wait per send (async but paced) or NO_MEM errors; no real speed advantage over broadcast |
| Dispatch 1→8 | `esp_now_send(NULL, …)` (send to **all paired peers**) | Same as unicast loop (~3–5 ms) | Same as unicast (per-peer ACK) | One API call but still N sequential frames on air; all 8 MACs must be added peers |
| Dispatch 1→8 | UDP broadcast via AP (one AP, 8 STAs) | ~2–6 ms steady-state per cycle **plus** WiFi join (seconds at boot); broadcast goes at **lowest mandatory/basic rate** (1–6 Mbps) with **no per-recipient ACK** | No delivery guarantee to any specific STA; worst airtime | Needs AP + associations; lwIP jitter (see §4); useless for AP-less operation |
| **Collect 8→1** | **ESP-NOW TDMA / staggered unicast** — worker *i* transmits in its own slot (`t0 + i×Δ`, Δ≈0.5–1 ms) | deterministic **~3–6 ms** total (∑ airtime + per-frame turnaround) | Each unicast ack/retried; coordinator can be sure of which workers answered (missing = retry/whole-miss list) | Requires slot discipline; no HW QoS on C3 (only DCF); keep callback fast or frames drop (see §3) |
| Collect 8→1 | ESP-NOW unsynchronized burst | 1–40 ms **unbounded tail** (CSMA/CA collisions → backoff + retries; NO_MEM backpressure) | Drops; broadcast result frames are **silently lost** on collision (no ACK) | No arbitration beyond DCF; hidden-node-ish drops in a star; don't do this |
| Collect 8→1 | UDP unicast up via AP | ≈ **7–25 ms** (8 uploads × backoff/airtime + AP bridge + 7 ms RTT baseline) | 802.11 ACK/retry makes STA→AP leg reliable; AP→coordinator local and fast | lwIP `UDP_RECVMBOX_SIZE=6` per socket → coordinator drops a 7th–8th un-drained datagram in a burst; AP adds jitter; fragmentation off |

**Recommended.** ESP-NOW only, no AP. Dispatch with one broadcast frame; collect with a deterministic per-worker slot
(unicast). One full round ≈ **3–6 ms** typical (sub-ms broadcast fan-out + ~3–5 ms staggered collect), versus ≥ 7 ms for a single UDP round trip class. Keep each ESP-NOW frame ≤ **244 B** (hard ceiling 250 B in ESP-IDF 5.1 / core 3.0.7 — v2.0's 1470-B frames need IDF ≥ 5.4). Port bulk results out-of-band via TCP/UDP if they exceed 250 B.

*(Wall-clock figures are estimates: single-link measured baselines (0.85 ms RTT) + published airtime numbers (§2–§4). Validate on hardware — plan a 1→8 + 8→1 timing sweep.)*

---

## Q1. ESP-NOW group semantics: true broadcast? peer limits on C3?

**Yes — it is a true single-transmission broadcast, not N enqueued copies.**
- ESP-NOW sends direct 802.11 vendor-specific action frames; FromDS/ToDS=0; the third address field carries `ff:ff:ff:ff:ff:ff` for broadcast. [ESP-IDF ESP-NOW API ref, esp32c3]
- **"A device with a broadcast MAC address must be added before sending broadcast data."** One `esp_now_send` to that MAC reaches every listening ESP32 in range on that channel. [ESP-IDF ESP-NOW API ref]
- Official `espressif/esp-now` user guide: transmission mode is "flexible including unicast and broadcast, and supports one-to-many and many-to-many device connection." [github.com/espressif/esp-now/blob/master/User_Guide.md]
- Arduino core 3.0.7 ships an official `ESP_NOW.Broadcast_Peer`/`ESP_NOW_Broadcast_Master` implementation that registers `ESP_NOW.BROADCAST_ADDR` and sends one frame to all receivers. [developer.espressif.com blog "Using ESP-NOW in Arduino" (Aug 2024); local `libraries/ESP_NOW/examples/ESP_NOW_Broadcast_Master`]
- **No device-count limit on broadcast recipients** — every device hears the frame. The "max peers" limit applies only to the **paired-device table**. ESP-FAQ: to address >20 specific devices, use broadcast + put destination addresses in the payload. [ESP-FAQ esp-now]

**Peer limits (exact for our C3 build, header-verified):**
```
ESP_NOW_MAX_TOTAL_PEER_NUM   20   /* max total peers             */
ESP_NOW_MAX_ENCRYPT_PEER_NUM  6   /* max encrypted peers (C3)    */
ESP_NOW_MAX_DATA_LEN         250  /* v1.0 max payload per frame  */
```
[`framework-arduinoespressif32-libs/esp32c3/include/esp_wifi/include/esp_now.h`]
- So: **20** unencrypted paired peers (the "10" some tutorials cite is a legacy/mode-dependent figure; Random Nerd Tutorials and the ESP-IDF docs report 10 encrypted/STA or 6 encrypted/SoftAP, with total < 20 unencrypted). For N=8 unencrypted fleet: coordinator needs 8 worker peers + 1 broadcast peer = **9 ≤ 20** — fits.
- **Encryption is the trap:** 6 encrypted peers < 8 workers. Do **not** enable ESP-NOW encryption for this fleet unless you drop below 6, or encrypt only the control unicast and leave broadcast open. [local header; IDF docs "paired encryption devices no more than 17, default 7", C3 build = 6]
- IDF 5.1 = ESP-NOW **v1.0 only** → 250-B max payload; 1470-B v2.0 frames require IDF ≥ 5.4 and the newer `esp_now_send_v2`. Our core 3.0.7/IDF 5.1 stays at 250 B. [ESP-IDF esp_now.h; ESP-FAQ esp-now]

---

## Q2. `esp_now_send` to N peers in a loop: blocking? pacing? frame rates?

**Non-blocking but queue-bound, and rate-limits real.**
- `esp_now_send` is asynchronous: it copies the buffer into the Wi-Fi TX path and returns immediately (the data pointer need not stay valid after return). [ESP-IDF ESP-NOW API ref]
- When the TX queue is full it returns **`ESP_ERR_ESPNOW_NO_MEM`**, whose doc literally says **"delay a while before sending the next data"**. Hot-looping without pacing therefore either errors or must key off the **send-complete callback**. [local header `esp_now.h`]
- **Pacing rule (from the API doc):** "too short interval between sending two ESP-NOW data may lead to disorder of sending callback function … sending the next ESP-NOW data **after the sending callback function of the previous sending has returned**." The send/recv callbacks run on the high-priority Wi-Fi task — no lengthy work there; post to a queue. [ESP-IDF ESP-NOW API ref]
- The classic symptom of no pacing: `esf_buf: no eb, TXQ_BLOCK=2000` — seen repeatedly when users fire `esp_now_send` in a `For` loop without any delay. [Random Nerd Tutorials, ESP-NOW one-to-many]

**Measured frame timing (published):**
- Default ESP-NOW PHY rate is **1 Mbps**. At 250 B that puts ~2 ms of payload airtime one-way on the wire; an Espressif engineer's estimate for a 250-B request/reply at 1 Mb/s was `(250*8/1M)*2 ≈ 4 ms`. [ESP-IDF issue #3238 / IDFGH-889]
- User measurement at **default rate**: 250-B request/reply **RTT 5.4–7.5 ms**. Same user then disabled AMPDU and called
  `esp_wifi_internal_set_fix_rate(iface, 1, WIFI_PHY_RATE_MCS7_SGI)` → **"under 700 µs round-trip"**. [ESP-IDF issue #3238]
- `esp-wifi-speed-test` follow-up (#7687): a **non-pipelined** request/reply ping-pong stays ≈ **6 ms even at MCS7-SGI** — i.e. naive per-packet wait loops dominate end-to-end latency, not the PHY. Pipeline or fire-and-forget rewards. [ESP-IDF issue #7687]
- One-to-one app-level throughput at default 1-Mbps PHY: ESP-FAQ measured **~214 Kbps open env / ~555 Kbps shielded box**; "if you require a higher rate, configure the TX rate" ([How do I set… docs]). [ESP-FAQ esp-now]
- Our own 0.85 ms 2-node RTT at MCS7-SGI/HT40/AMPDU-off matches the post-tuning regime: single-frame latency < 1 ms; the soft path (not airtime) dominates.
- Per-peer MCS7-SGI at HT20 = **72.2 Mbps**, HT40 = **150 Mbps** (C3 enum table). [local `esp_wifi_types.h`]

**Consequence for the loop:** 8 sequential unicast sends ≈ 8 × (one-way ~0.42 ms + pacing/ACK gap) ≈ **3–5 ms**; one broadcast replaces all 8 with a single ~0.2–0.5 ms frame.

---

## Q3. Fan-in: 8 concurrent senders → 1 coordinator: drops, airtime, CSMA, callback depth

**No scheduler exists; 802.11 DCF (CSMA/CA) is the only arbitration.**
- 802.11 uses carrier-sense multiple access with collision avoidance: each transmitter must sense the channel idle for DIFS + random backoff slots before sending. Two nodes that pick the same slot collide. There is no AP/central broker in ESP-NOW and no QoS/traffic classes on C3 (DCF only). [802.11 CSMA/CA background — Wikipedia/standard texts]
- For **broadcast** frames a collision is **silent**: no per-recipient ACK, so nobody knows it died — delivery is best-effort. For **unicast**, the Wi-Fi MAC ACKs and retries the frame at the driver level (more robust, still no application-level guarantee, and retries consume airtime). [ESP-FAQ; pschatzmann/arduino-audio-tools discussion #605: broadcast "disables retransmissions/error correction"].
- With 8 workers bursting at once, you get random collisions → exponential backoff, retries (unicast), and **silently dropped frames** (broadcast). Wall-clock becomes a lottery. Deterministic result collection therefore **requires app-level scheduling** (staggered slots / TDMA keyed off the coordinator's dispatch, or per-worker backoff windows of a few ms).

**Can one receive callback handle 8 back-to-back frames?**
- The receiving callback runs on the **Wi-Fi task**; ESP-IDF: "**do not do lengthy operations in the callback function. Instead, post the necessary data to a queue and handle it from a lower priority task.**" [ESP-IDF ESP-NOW API ref]
- 8 back-to-back small (≤250 B) frames are individually fine **if** the callback just memcpy+queue (a few µs each). Our build has `CONFIG_ESP32_WIFI_STATIC_RX_BUFFER_NUM=8` static RX buffers and AMPDU-RX compiled on; a slow callback or high RX rate exhausts RX buffers/queue depth and frames drop. Keep a deep FreeRTOS queue (≥16 entries) and a dedicated drain task. [local `sdkconfig`]
- Same rule the other way: send callback + recv callback both on the Wi-Fi task → don't echo receipts from inside the callback, or you throttle RX.

**Published fan-in data points:**
- **Espressif DevCon 2024 community study (8→1):** 8 ESP32 sensor nodes → 1 central node (ESP32-S3/ESP32-C6) over ESP-NOW at 4 pkts/s; indoor result: only a subset (best links: nodes 1, 2, 4, 6 — about **50% of nodes reliable**), loss strongly correlated with walls/building structure; recommendation: ACK-based delivery control + multiple central units. [developer.espressif.com blog "Data transmission reliability over ESP-NOW protocol in indoor environment", DevCon 2024]
- **Indoor Performance Evaluation of ESP-NOW** (Urazayev, Zorbas, et al., 2023): packet-success-rate experiments vs distance/placement (indoor). [IEEE/ResearchGate publication 373470899 / 369626626]
- Multiple-senders → one receiver is the standard ESP-NOW "sensor dashboard" pattern (Random Nerd Tutorials ESP-NOW Web Server Dashboard; several boards → 1 receiver). Works, but relies on low duty cycle; nothing published at our "8 full-speed concurrent burst" rate on a C3 — the reliable-engineered answer is staggering.

---

## Q4. UDP broadcast fan-out on one AP — works? latency vs unicast? packet size on C3?

**Yes it works, at a real cost.**
- An AP relays broadcast/multicast frames to every STA in its BSS; ESP32 Arduino can send with `WiFiUDP.beginBroadcast()`/`write` or AsyncUDP `broadcastTo()`. [arduino-esp32 AsyncUDP; esp32 forum]
- **Latency/rate penalty:** 802.11 multicast/broadcast (and management) frames are sent at the **lowest mandatory / basic rate** — 1–6 Mbps in 2.4 GHz — with **no per-recipient ACK/retry**. [Cisco WLAN site-survey guidelines; MikroTik "Wireless Interface" docs; AlgoMaster 802.11 basics; ScienceDirect "Multicast Transmission" overview]
  => Higher airtime per frame than a unicast at 72–150 Mbps, delivery to any specific STA is best-effort, and airtime balloons as frames grow.
- **UDP jitter on ESP32 is real:** ESP-IDF #15345 (UDP audio bridge on ESP32-C5/C6) reports steady ~1.36 ms cadence ±500 µs, but **random 10–20 ms pile-ups** caused by TX-path buffering (fixed in IDF for rev≥1.0 C5/C6; nothing for C3's lwIP path). For dispatch that jitters your whole fleet; ESP-NOW is immune (no lwIP in the path). [github.com/espressif/esp-idf/issues/15345, IDFGH-14588]
- **Packet size on our C3 build:**
  - Safe non-fragmenting UDP payload = **1472 B** (1500 MTU − 20 IP − 8 UDP). In practice tcpdump on ESP32 shows 1460-B segments (MTU-minus-headers behavior; arduino-esp32 #3874 camera example: 1460-B UDP datagrams). [arduino-esp32 #3874]
  - **IP transmit fragmentation is compiled out** in our build (`CONFIG_LWIP_IP_FRAG` absent). Oversize sends are **not** fragmented — they are "broken into pieces and sent as separate UDP packets" (each ≤ MTU), or dropped, per lwIP config. [esp32.com t=10656; t=10289 UDP-fragmentation-disabled threads]
  - RX **reassembly is present** (`CONFIG_LWIP_IP_REASS_MAX_PBUFS=10`) for any externally-fragmented datagrams. [local `sdkconfig`]
  - So: **use UDP payloads ≤ 1460–1472 B; below ~1440 is the safe zone used by real projects** (the #15345 stack chose 1452 B). For our fleet the intent is tiny control/results → this is a non-issue; it matters only for bulk TCP/UDP channels (TCP MSS=1436 on this build).

---

## Firmware recommendation (concrete)

1. **Transport:** ESP-NOW end-to-end; **no AP**, all 9 devices on one channel. Dispatch = one **broadcast** frame; collect = **staggered unicast slots**.
2. **PHY tuning (per device):** disable AMPDU (menuconfig `WIFI_AMPDU_TX_ENABLED`/`_RX_ENABLED` = n), set `WIFI_PS_NONE`, then set the rate before use with
   `esp_now_set_peer_rate_config()` (modern, per-peer; present in this IDF 5.1) — or the deprecated-but-working `esp_wifi_config_espnow_rate(ifx, WIFI_PHY_RATE_MCS7_SGI)` (72.2 Mbps @ HT20; 150 @ HT40). Same as the 0.85 ms-RTT baseline.
3. **Dispatch (coordinator):** `esp_now_add_peer(&broadcast)` once; send one ≤244-B frame to `FF:FF:FF:FF:FF:FF`. No loop, no pacing needed for a single frame; measure round-trip to rebuild a **missing-worker list** (`8 - answered`) and unicast-retry only those.
4. **Collect (workers):** each worker transmits its result (≤244 B) in a fixed slot: `delay = worker_index × Δ` (Δ ≈ 0.5–1 ms from dispatch receipt). Unicast to the coordinator so 802.11 ACK/retry protects each frame. Coordinator: recv callback does **only** `memcpy` + queue; a lower-priority task drains a `xQueueCreate(16,…)` and processes. Missing slot = worker missed the dispatch → add to retry list.
5. **Payload budget:** ≤ 244 B/frame in ESP-IDF 5.1 (250-B ceiling). If results exceed that, either chunk + sequence-number over several frames, or open a separate UDP/TCP bulk path (TCP 5.8 Mbps >4 KiB measured) only for big payloads; keep control on ESP-NOW.
6. **No encryption** (6-peer encrypted cap < 8 workers). 
7. **Expected budget:** ~0.2–0.5 ms fan-out + ~3–5 ms staggered collect + ~1 ms coordinator processing ≈ **3–6 ms per full round**; measure with `esp_timer_get_time()` stamps on coordinator log.

---

## Sources
- ESP-IDF ESP-NOW API reference (esp32c3) — docs.espressif.com (broadcast MAC peer, pacing note, callback warning, send-to-all-peers `esp_now_send(NULL,…)`, max 20 paired / 17 encrypted default 7).
- Local C3 headers (this build): `esp_now.h` (20/6/250, `ESP_ERR_ESPNOW_NO_MEM` doc, `esp_now_set_peer_rate_config`, deprecated `esp_wifi_config_espnow_rate`, MCS7_SGI 72.2/150 Mbps) — `framework-arduinoespressif32-libs/esp32c3`.
- Local C3 `sdkconfig`: no `LWIP_IP_FRAG`; `LWIP_IP_REASS_MAX_PBUFS=10`; `LWIP_MAX_UDP_PCBS=16`; `LWIP_UDP_RECVMBOX_SIZE=6`; `ESP32_WIFI_STATIC_RX_BUFFER_NUM=8`; AMPDU TX/RX on; TCP MSS 1436.
- ESP-FAQ — esp-now: default 1 Mbps → ~214 Kbps (open) / ~555 Kbps (box); broadcast no device-count limit (address-in-payload); v1.0 250-B limit stems from the vendor-element Length field; IDF < 5.4 = v1.0 only.
- `espressif/esp-now` User_Guide.md — unicast/broadcast, one-to-many & many-to-many.
- ESP-IDF #3238 (IDFGH-889) — default-rate RTT 5.4–7.5 ms; AMPDU-off + MCS7_SGI → <700 µs RTT; `(250*8/1M)*2` estimate.
- ESP-IDF #7687 — non-pipelined request/reply ≈6 ms even at MCS7-SGI; espnow-speed-test.
- Random Nerd Tutorials (one-to-many) — peer limits, `esf_buf: no eb, TXQ_BLOCK=2000` when sending without delay.
- Espressif blog "Using ESP-NOW in Arduino" (2024) — official Arduino `ESP_NOW` lib, `Broadcast_Peer`, broadcast master/slave examples, multi-broadcaster→receiver.
- Espressif blog / DevCon 2024 — 8-node→1-central ESP-NOW reliability study (50% reliable indoors, ACK recommendation).
- "Indoor Performance Evaluation of ESP-NOW" (Urazayev, Zorbas et al., 2023) — indoor PSR experiments.
- pschatzmann/arduino-audio-tools discussion #605 — broadcast needed for synchronized one-to-many; broadcast disables retransmissions/error correction; node time-shift otherwise.
- Cisco WLAN site-survey guidelines / MikroTik "Wireless Interface" / AlgoMaster 802.11 — broadcast & management frames at lowest mandatory basic rate, no per-recipient ACK.
- ESP-IDF #15345 (IDFGH-14588) — UDP send jitter ±500 µs with random 10–20 ms pile-ups on ESP32; 1452-B no-fragment payload used.
- esp32.com t=10656 / t=10289 — lwIP fragmentation disabled; oversized UDP split into separate ≤MTU packets.
- arduino-esp32 #3874 — 1460/1500-B MTU behavior; UDP datagrams fragmented into 1460-B segments.
- StackOverflow 66103497 — ESP32 UDP broadcast RX pitfalls (association/interface filter).

*Research performed 2026-08-31 via firecrawl; working scrapes in `.firecrawl/fleet-*.md|json` in this repo.*
