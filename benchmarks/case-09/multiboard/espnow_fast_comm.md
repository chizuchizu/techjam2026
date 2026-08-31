# ESP-NOW fast dispatch — Phase 2 (measured 2026-09-02) — fleet comm fast path

## Result
The fleet coordination path (dispatch a job + collect a reply) moves from
**UDP round-trip ~40 ms** to **ESP-NOW median RTT ~4.4–5.2 ms** on the same two
boards, with **0 lost rounds over 64 and 256 round runs**.

| Figure | UDP (fleet baseline) | ESP-NOW (this work) |
|---|---:|---:|
| Dispatch+reply RTT (median) | ~40 ms | **4.4–5.2 ms** (med across runs) |
| RTT min | ~40 ms | **2.26–2.38 ms** |
| Round loss | n/a | **0 / 64, 0 / 256** |
| Dispatch cost for N boards | O(N) unicast | O(1) broadcast + N slotted unicast replies |

256-round run (link-station-espnow-256): `n=256 | min_us=2369 | avg_us=9649 |
max_us=86839 | med_us=4602 | miss=0`.
64-round production build (quiet): `n=64 | min_us=2257 | avg_us=10589 |
max_us=70002 | med_us=5226 | miss=0`.

The RTT floor (~2.3–3 ms) is worker main-loop poll latency, not radio: the
worker defers its reply to the main loop (see design note below), so the
dispatch round-trip includes one worker loop iteration. Pure RF one-way is
well under 1 ms on this link.

## Design (8-board capable)
1. Coordinator broadcast-dispatches `EB|<seq>` on ESP-NOW — one frame reaches
   all workers (O(1) dispatch instead of the fleet's O(N) UDP unicast loop).
2. Each worker replies in its own 1500 us collect slot
   (`LF_ESPNOW_SLOT_US * worker_index` after receipt) — the slotted replies
   avoid an 8→1 CSMA burst at the coordinator.
3. Replies are unicast to the sender with an `esp_now_send_cb` status check.
   Slot + unicast keeps the design valid at 8 boards.
4. Bench runs a fixed number of measured rounds after 3 warmup rounds and
   reports min/avg/max/median + miss count on `bench|done`.

## The two bugs that made Phase 2 look broken (both fixed)
Both failures were **protocol bugs, not radio problems**. A minimal standalone
app (`espnow_min.cpp`) plus a bidirectional ping mode proved ESP-NOW broadcast
AND unicast deliver ~100% in the main firmware with ~3 ms flight time.

1. **seq-parse off-by-one, worker side**: the old worker parsed the sequence
   from `data[2]` but the frame is `EB|<seq>` — index 2 is `'|'`, so `d=0`,
   every dispatch silently dropped. Frames were always arriving; "stops at
   3 received" was a debug-print cap.
2. **reply send from inside the ESP-NOW recv callback**: `esp_now_send()` in
   the recv callback stalls the wifi task on this stack (frames sat in the RX
   queue ~1 s and replies never transmitted). Fix: the callback only records
   seq + sender MAC; `eb_poll` (main loop) performs the slot-timed send.
3. **seq-parse off-by-one, coordinator side**: the reply `EBR|<seq>` parsed
   from index 3 (`'|'`) — same class of bug, fixed to skip `EBR|`.

## Envs
In `esp32-linkbench/platformio.ini` (ESP-NOW bench lives inside the normal
station firmware behind `-DLF_ESPNOW_BENCH=1`):

| Env | Role / purpose |
|---|---|
| `link-station-espnow-ap` | forced-AP worker board (this rig: A, MAC ...5AE8) |
| `link-station-espnow` | station coordinator (this rig: B, MAC ...02D8) |
| `link-station-espnow-256` / `-256ap` | 256-round statistical run |
| `link-station-espnow-dtc` | diag: 1200 ms reply window (late-reply t-shoot) |
| `link-station-espnow-ping` / `-pingap` | diag: bidirectional 300 ms ping, logs every rx |
| `espnow-min-ap` / `espnow-min-sta` | minimal standalone isolation app |

## Reproduce
```bash
cd esp32-linkbench
pio run -e link-station-espnow-ap -t upload --upload-port /dev/cu.usbmodem101
pio run -e link-station-espnow   -t upload --upload-port /dev/cu.usbmodem1101
# coordinator prints (quiet build):
#   LINKFW-E|bench|start|rounds=64|slot_us=1500
#   LINKFW-E|bench|done|n=64|min_us=...|avg_us=...|max_us=...|med_us=...|miss=0
# turn on per-round traces with -DLF_ESPNOW_DEBUG=1
```
