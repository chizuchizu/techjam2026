# Fleet collectives for N=8 ESP32-C3 — dispatch + collect on one 2.4 GHz channel

**Scope.** 8 single-core ESP32-C3 (160 MHz, 320 KB SRAM) sharing one 2.4 GHz channel via one AP,
coordinated by a coordinator (lowest MAC). Data-parallel jobs with small exchange (partial sums /
small vectors per node). Today: O(N) UDP unicast dispatch (JOB) + sequential collect (RESULT) =>
end-to-end wall is sublinear even though compute scales linearly. Goal: minimize wall-clock.

**Sources.** Marked `[FETCHED]` = scraped full page with firecrawl during this research.
`[FOUND]` = returned by a firecrawl web search this session (title/URL/abstract-level).
All URLs are real.

- `[FETCHED]` ESP-IDF ESP-NOW API — https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-reference/network/esp_now.html
- `[FETCHED]` ESP-IDF Wi-Fi driver perf (ESP32 single-channel numbers) — https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-guides/wifi-driver/wifi-performance-and-power-save.html
- `[FETCHED]` Electric UI: latency benchmark of wireless links (12B/128B/1kB) — https://electricui.com/blog/latency-comparison
- `[FETCHED]` Wikipedia CSMA/CA (DCF, backoff, RTS/CTS) — https://en.wikipedia.org/wiki/Carrier-sense_multiple_access_with_collision_avoidance
- `[FETCHED]` espressif/esp-idf issue #15345: UDP send jitter/pile-ups on ESP32 (C5/C6) — https://github.com/espressif/esp-idf/issues/15345
- `[FETCHED]` RandomNerdTutorials ESP-NOW getting started — https://www.randomnerdtutorials.com/esp-now-esp32-arduino-ide/
- `[FETCHED]` OpenELAB "ESP-NOW: Eight Points You Must Know" — https://openelab.io/blogs/learn/esp-now-eight-points-you-must-know
- `[FETCHED]` RockNet: Distributed Learning on Ultra-Low-Power Devices (arXiv abs) — https://arxiv.org/abs/2510.13320
- `[FETCHED]` "Distributed Computing From First Principles" (arXiv survey, collectives/bcast/gather/RMA) — https://arxiv.org/pdf/2506.12959
- `[FOUND]` Bandwidth-optimal all-reduce algorithms for clusters of workstations — https://dl.acm.org/doi/10.1016/j.jpdc.2008.09.002
- `[FOUND]` ESP32 multicast UDP high losses (SO thread; full page blocked for us) — https://stackoverflow.com/questions/51150014/esp32-multicast-udp-high-losses-receiving

**Timing anchors used below.** Our ESP-NOW 2-node link: **0.85 ms RTT** (measured, branch linkfast).
ElectricUI typical per-packet (full software stack, ESP32): ESP-NOW 12 B = 5.6 ms, 128 B = 5.8 ms,
1 kB (5x frames) = 24 ms; tuned WiFi UDP ~9 ms median. ESP-IDF issue #15345 documents **10–20 ms
random UDP send pile-ups** on ESP32-family even in a clean single-link setup. Treat every "wall"
figure below as an **estimate** for planning; measure on the fleet.

---

## Bottom line / recommended "dispatch + collect" design (biggest win first)

Replace the sequential `dispatch -> compute -> collect` chain with:

1. **Broadcast the JOB once** to all 8 workers (one frame, O(1) channel time) — ESP-NOW broadcast
   when payload fits (<=230 B payload, default 1 Mbps) or UDP subnet broadcast via the AP for larger
   jobs. All workers decode it and start computing at (nearly) the same instant. Adds a job
   **sequence number**; missing workers pull once via unicast *(technique T1/T7)*.
2. **Workers do one-frame "flat" star collect**: each worker reduces to the minimal result (often
   a partial sum / small vector, few 10s of bytes) and pushes exactly **one RESULT datagram**
   (no request/response ping-pong) at a **fixed slot** `slot = worker_id * N + jitter` (T3 + T4).
3. **Overlap everything**: worker computes immediately after RX and pushes the previous round's
   result while computing the next (double-buffered result slots, non-blocking `sendto`);
   coordinator runs an **async RX/collect task** so dispatch of round k+1 and collect of round k
   overlap. No global barrier between rounds (T2).
4. Coordinator tracks an 8-bit **arrival bitmap**; after the slot window it unicasts a targeted
   `RESEND(job,worker)` only to missing workers, and only to gather-all for the *final* round —
   never as a per-round barrier (T1/T7 reliability).

**Expected wall clock.** With compute >> comm the wall is only `max(compute, comm)` instead of a sum:
dispatch shrinks from ~8 x per-unicast-frame to ~1 broadcast frame; collect (~8 short frames) is
mostly hidden under compute; TDMA-ish slots remove collision-tail variance. The 8x compute speedup
then shows up in end-to-end wall instead of being eaten by the serial O(N) phases.

---

## Technique ranking (N=8, expected wall-clock win, biggest first)

### T1. Broadcast dispatch — one frame to all workers (ESP-NOW broadcast or UDP broadcast)  [WIN: large]
- **What.** Coordinator sends the job descriptor (job id, params, collect-slot schedule) in one
  group-addressed frame instead of N unicasts. ESP-NOW: add peer with broadcast MAC
  `ff:ff:ff:ff:ff:ff` then `esp_now_send()` — one Wi-Fi action frame heard by all 8.
  UDP: `NetworkUDP.beginPacket(broadcastIP)` — one IP datagram the AP relays to all associated STAs
  (station broadcast frames and AP-relayed broadcasts are all 802.11 group frames).
- **Why it helps wall.** Removes the O(N) dispatch serialization: today's dispatch is the first
  offender making E2E wall sublinear. All workers start compute at the same time (tight start
  alignment), so compute overlap is maximal.
- **Expected effect.** Dispatch channel time goes from `~N * (airtime + access + SW per-send)` to
  `~1 * frame`. With ~2-4 ms typical per unicast frame (estimate; see ElectricUI 5.6-5.9 ms typical
  per ESP-NOW packet), saving ~7 frames ≈ **10-25 ms per job round**, and removes jitter from
  staggered compute starts.
- **Implementation notes (Arduino core 3.x / IDF 5.1).** ESP-NOW: `esp_now_init()`, add broadcast
  peer `{ff:ff:ff:ff:ff:ff}` (works over Station or SoftAP interface), `esp_now_send()`;
  broadcast payload limit is 250 B (`ESP_NOW_MAX_DATA_LEN`), per-peer rate configurable via
  `esp_now_set_peer_rate_config()` (default 1 Mbps). UDP broadcast: `NetworkUDP.beginPacket(IPAddress(bcast))`.
  Reliability: broadcast has **no per-receiver ACK/retry** in either transport (see T7).
- **Citations.** ESP-NOW API doc `[FETCHED]`; RNT getting started + linked one-to-many page `[FETCHED]`;
  OpenELAB `[FETCHED]`; CSMA/CA wiki (group frames contest the channel) `[FETCHED]`.

### T2. Overlap compute/collect + dispatch/collect (pipeline, double-buffer, async RX)  [WIN: large]
- **What.** Run RX processing in a separate high-priority path (on C3 the ESP-NOW recv callback and
  lwIP/WiFi RX already run in the WiFi task) while the app task computes. Workers keep two result
  slots: fill slot A during round k compute, transmit slot A during round k+1 compute. Coordinator
  never blocks waiting for all 8; it dispatches round k+1 as soon as round k has started, and
  collects round k results in the background.
- **Why it helps wall.** Converts the serial phase sum into an overlap: E2E wall -> ~`max(total
  compute, dispatch, collect)` rather than their sum. This is the standard fix for the "dispatch
  -> compute -> collect" sublinear behaviour the fleet observes.
- **Expected effect.** For any workload where compute per round >= dispatch+collect, the comm cost
  effectively disappears from the critical path (up to the combined 1 Mbps channel budget).
  RockNet reports scaling one central device to 20 ultra-low-power MCUs cuts per-device
  memory/latency/energy up to 90% by tightly integrating compute and communication.
- **Implementation notes.** Worker: `esp_now_register_recv_cb()`/UDP RX callback sets a "job ready"
  flag + copies payload (short, non-blocking); main loop computes and, when done, `sendto()`
  (non-blocking) or `esp_now_send()` into the other slot. Do not `delay()` inside compute loops
  longer than needed to keep WiFi task scheduling fair (single core: WiFi task preempts app task on
  IDF 5.x so RX stays live). Coordinator: dedicated collect task (`xTaskCreatePinnedToCore`, though
  C3 has 1 core — just give it reasonable priority) + arrival bitmap.
- **Citations.** RockNet abstract `[FETCHED]`; ESP-IDF UDP jitter issue (shows TX pile-ups, so never
  block on send) `[FETCHED]`; arXiv collectives/pipelining survey `[FETCHED]`.

### T3. One-frame flat star collect with minimal payload (no request/response)  [WIN: medium-large]
- **What.** Each worker reduces its partial result to the smallest representation (e.g. one or a few
  `uint32_t`/`float` partials) and sends exactly one short frame upstream. Identity is in the
  payload header (worker id + round id). Coordinator aggregates in a fixed-size array.
- **Why it helps wall.** Per-frame overhead dominates on a 1 Mbps-class link (see ElectricUI: an
  extra ESP-NOW packet costs ~1 ms even when it carries 12 bytes; a 5-frame 1 kB transfer costs
  24 ms vs 5.8 ms for one 128 B frame). Fewer and smaller frames = directly less channel time and
  less host processing.
- **Tree/butterfly is ruled out at N=8.** On a **single shared CSMA channel every transmission is
  serialized**, so a tree/butterfly reduce does *not* reduce channel occupancy for a gather (each
  edge transmission still reserves the channel). A star gather is 8 channel transmissions; a tree
  gather is >= (N-1) transmissions for the same bytes on the medium (more, if intermediate nodes
  forward combined data whose total bytes already moved). The `log(N)` latency win of butterfly
  trees comes from pipelined *parallel* links on switched full-duplex fabrics, and from allreduce
  (data needed by all), neither of which describes N=8 -> 1 coordinator on one CSMA channel.
  Keep trees only if intermediate CPUs must *combine* to shrink bytes AND data is large; at N=8
  with small partials the star is simpler and at least as fast.
- **Expected effect.** Collect drops to 8 short frames with no trailing request/response latency;
  combined with T2 it is hidden under compute. Against the current collect it removes per-node
  round-trips (each RESULT currently costs a send + ACK-wait pattern).
- **Citations.** ElectricUI per-packet overhead `[FETCHED]`; ACM bandwidth-optimal allreduce paper
  (found via search; the switched-network basis) `[FOUND]`; arXiv collectives/gather survey `[FETCHED]`.

### T4. Deterministic (TDMA-like) result slots — beat the 8->1 CSMA burst  [WIN: medium]
- **What.** Give each worker a fixed send window, e.g. `tx_start = round_start + worker_id * SLOT`
  (SLOT ~ per-frame airtime + margin), pushed inside the JOB broadcast. Workers transmit during
  their slot (optionally plus a few hundred µs jitter). Coordinator sets slot windows from the same
  schedule.
- **Why it helps wall.** Raw CSMA with 8 workers finishing simultaneously ("thundering herd")
  produces collisions -> exponential backoff -> tail latency and jitter (the mechanism documented in
  the CSMA/CA wiki: same-channel transmitters take turns via backoff). A static schedule turns the
  collect into a mostly collision-free sequential drain: predictable ~N*SLOT finish time and
  near-zero retries. It also decorrelates the broadcast-dispatch start (all 8 start together) from
  the all-8-finish-together result storm.
- **Expected effect.** Removes collision-driven retry tail; collect variance drops sharply; with
  1 Mbps ESP-NOW a 64 B result slot is ~1-2 ms, so N=8 -> ~10-15 ms worst-case deterministic drain
  that can even be pipelined across rounds (T2).
- **Implementation notes.** Encode SLOT in the JOB broadcast (T1) so it costs zero extra frames.
  On coordinator, per-worker receive timeout = SLOT; on expiry mark missing and let T7 handle.
  Note ESP-IDF #15345: even a single clean UDP link bursts 10-20 ms occasionally — never make a
  slot shorter than realistic jitter, and treat the collect as probabilistic with a final resend
  round rather than a hard barrier.
- **Citations.** CSMA/CA wiki `[FETCHED]`; ESP-IDF UDP jitter issue #15345 `[FETCHED]`; TDMA-vs-CSMA
  search results (TDMA-based scheduling for WSN surveys found via search).

### T5. Batch/aggregate results into fewer, larger frames  [WIN: medium (only if >1 value per round)]
- **What.** Accumulate multiple result items on the worker and flush as one frame up to the payload
  cap (250 B ESP-NOW / ~1472 B UDP), instead of one frame per item. Reassembly by (worker, round)
  + item index on coordinator.
- **Why it helps wall.** Per-frame fixed costs (PHY preamble + SIFS/DIFS/backoff + host `sendto`
  processing) are paid once per frame. On a 1 Mbps class link, 8 small frames cost ~8x the overhead
  of 1 large frame (ElectricUI's 12 B vs 128 B figures are within 0.2 ms of each other: bytes are
  nearly free, frames are not).
- **Expected effect.** For multi-partial results (e.g. 8 workers x 8 sub-results), collect channel
  time drops by up to ~8x for that data. Single-scenario result: use T3 instead (already one frame).
- **Implementation notes.** Worker: fixed staging buffer, flush on `itemCount>=THRESHOLD` or timer;
  keep UDP datagrams under MTU to avoid IP fragmentation (fragments are not retried as reliably and
  on broadcast they are especially lossy).
- **Citations.** ElectricUI per-frame vs per-byte data `[FETCHED]`; ESP-NOW 250 B frame cap `[FETCHED]`;
  SO multicast-loss thread (fragmentation/multicast reliability context) `[FOUND]`.

### T6. Hybrid control plane: ESP-NOW for dispatch/ACK/control, UDP for bulk payloads  [WIN: small-medium]
- **What.** Run two transports simultaneously: ESP-NOW (data-link, no IP, no AP turnaround, default
  1 Mbps) for JOB broadcasts, worker ACKs and control; UDP/IP via the existing AP for large job
  data and bulk RESULT uploads (WiFi PHY can carry 11-72 Mbps with big frames in the lab; ESP32
  dual-core lab numbers are ~30 Mbps UDP — C3 will be well below, but still > ESP-NOW's 1 Mbps cap).
- **Why it helps wall.** Dispatch/control latency drops to our measured sub-ms regime (0.85 ms RTT
  2-node linkfast) and frees CPU (no lwIP on control path; ESP-NOW recv cb runs in WiFi task).
  Bulk data still gets the higher-rate WiFi path. ESP32's Wi-Fi driver runs ESP-NOW RX and WiFi RX
  concurrently, so both stay live.
- **Expected effect.** Control-plane frame budget moves from "typical WiFi ~5-9 ms per round trip"
  (ElectricUI) to "sub-ms" for dispatch (our measured link). For small-exchange jobs this is mostly
  a latency/jitter win; for dispatch-heavy workloads a wall-clock win.
- **Implementation notes.** Keep ESP-NOW payload <= 230 B (leave room inside 250 B cap), add the
  `ff:ff:ff:ff:ff:ff` peer, `esp_now_send` for dispatch and per-worker unicast ACK/`RESEND`.
  Keep `NetworkUDP` only where payload size demands it. Remember ESP-NOW broadcast gives you no
  receiver ACK (T7).
- **Citations.** ESP-NOW API doc (1 Mbps default, broadcast peer, 250 B cap, recv/send callbacks)
  `[FETCHED]`; ElectricUI WiFi vs ESP-NOW latency `[FETCHED]`; our measured 2-node ESP-NOW link
  (linkfast branch, 0.85 ms RTT, project-internal).

### T7. Reliability on broadcast: sequence numbers + missing-worker unicast pull  [WIN: small but guards T1/T3]
- **What.** Broadcast drops are possible because 802.11 group-addressed frames are **not ACKed and
  not retried** (this is true for both ESP-NOW broadcast and UDP broadcast relayed by the AP | the
  MAC-layer unicast ACK loop doesn't protect group frames). Add: (a) sequential job/round id in
  every frame; (b) workers dedupe identical ids; (c) coordinator's collect task lacks workers from
  an 8-bit bitmap and unicasts `RESEND(job,worker)` to exactly the missing ones.
- **Why it helps wall.** Bounded worst case: a broadcast miss costs one unicast retry round
  (cheap), never a stall-forever. It makes broadcast dispatch *safe to adopt* — the whole reason
  T1 is usable.
- **Expected effect.** At N=8 on a clean indoor channel expect ~0-2 retries total per round
  (estimate); each retry is a single unicast, so added wall is tiny and does not scale with N.
- **Implementation notes.** Worker->coordinator RESULT headers: `{job_id, round, worker_id,
  checksum}`. Coordinator: bitmap per round, one unicast re-request per missing worker after the
  slot window (T4), don't stall other workers (T2).
- **Citations.** ESP-NOW API doc: "not guaranteed that application layer can receive the data...
  send back ack data... assign sequence number... drop duplicate" `[FETCHED]`; unicast-ACK-only
  behaviour implicit in CSMA/CA wiki (unicast frames use ACK; group frames do not) `[FETCHED]`.

---

## Ruled out / do-not-build list (why)

- **Tree / butterfly reduce for the gather at N=8.** Same-channel serialization makes it >= the
  star's channel time for a 1->N-1 root gather; keep the star (T3). Log(N) only pays on switched,
  parallel-link fabrics — the ACM allreduce result behind that idea assumes full-duplex switched
  links `[FOUND]`.
- **True PGAS / remote reads over the radio.** Random remote memory access over a 1 Mbps shared
  channel is a latency disaster. Use SPMD "owner-computes": fixed partitions, workers push only
  small partials/deltas; nothing reads remote memory on demand. (This is exactly RockNet's
  communication-efficient design: timeseries = few hundred bytes, classifier partials = tens of
  bytes, communicated once per round `[FETCHED]`.)
- **Full TDMA implementation (802.15.4-style superframes).** At N=8 on 802.11, static full-TDMA
  adds complexity (guard times, sync drift) beyond what T4's simple slots + T7's retry give.
  You get the determinism you need from per-worker slot offsets without building a scheduler.

## Open questions to measure on the fleet (next step)
- Real per-frame dispatch/collect air times for ESP-NOW vs UDP on C3 (ElectricUI numbers are for a
  different stack/hardware).
- Empirical broadcast loss rate at N=8 indoors (to size the T7 retry budget).
- Whether ESP-NOW broadcast RX stays live while the C3 core does heavy compute (test with a
  synthetic busy loop).
