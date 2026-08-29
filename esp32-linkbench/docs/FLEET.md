# FLEET — multi-board parallel compute over WiFi (ESP32)

Same firmware on every board. No per-board config. Boards find each other over a shared
Wi-Fi network (by default the self-hosted `LINKNET` AP), elect a coordinator by lowest MAC,
and the coordinator splits a compute job across the fleet over UDP.

Target: 8 x ESP32-C3 / other ESP32 with WiFi. This doc describes the layer that turns the
"link-station" 2-board link into an N-board fleet.

## Build & flash

```bash
cd esp32-linkbench
pio run -e link-station                 # builds src/station_comm.cpp
# first board: starts the LINKNET AP when no SSID matches
pio run -e link-station -t upload --upload-port /dev/cu.usbmodem101
# ~20 s later, next board joins the AP, then flash it:
pio run -e link-station -t upload --upload-port /dev/cu.usbmodem1101
```

Flash order only matters for the first board (it must become the LINKNET AP first).
After that, any subset of boards can be added or removed at any time.

## Fleet protocol (UDP, port 42100)

| Msg      | Payload                                             | Who sends   |
|----------|-----------------------------------------------------|-------------|
| BEACON   | `BEACON|<id>|<mac-hex>|<uptime_ms>|<heap>`          | every node, ~1 s |
| PING     | `PING|<id>`                                         | any node    |
| PONG     | `PONG|<id>|<uptime_ms>|<heap>`                      | reply       |
| JOB      | `JOB|<jobid>|<op>|<offset>|<len>|<data>`            | coordinator -> workers |
| RESULT   | `RESULT|<worker-id>|<jobid>|<op>|<value>|<count>`   | worker -> coordinator |

- Identity: `esp32-<last2-hex-of-MAC>` (read from eFuse, unique, stable).
- Registry: peers expire after `LF_PEER_TTL_MS` (8 s) of silence; roles always recomputed
  from the current fleet.
- Coordinator: lowest MAC wins (`mac_cmp`). Re-elected automatically as boards join/leave.
- Datagram cap: 1460 B (Arduino core 3.x NetworkUDP). Fleet layer splits payloads so each
  worker gets one datagram <= 1300 B, matching ESP32 UDP goodput.
- Job timeout: `LF_JOB_TIMEOUT_MS` (3 s); `job|timeout` on expiry.

## Serial commands

| Key | Action |
|-----|--------|
| `S` | status / info line |
| `F` | fleet table (id, IP, MAC, uptime, heap, role, last-heard age) |
| `B` | run SUM16 demo on the coordinator (also auto-run, see below) |
| `I` / `H` | identify / beacon ping |

### Auto-run mode (self-orchestrating fleet)

Set `LF_DEMO_AUTO` (default 1) in `src/station_comm.cpp`. When set, the coordinator
runs the SUM16 demo on its own every `LF_DEMO_AUTO_PERIOD_MS` (6 s), starting
`LF_DEMO_AUTO_STARTUP_MS` (8 s) after the role is known - no PC in the loop. This is the
intended mode for the 8-board use case: the fleet distributes work by itself and the PC
(plugged into any one board) just watches `job|done` lines.

## SUM16 demo (end-to-end parallel compute smoke test)

`B` on the coordinator:
1. builds a 1024-uint16 pseudo-random vector (deterministic),
2. computes the expected full sum locally (self-verifying),
3. splits the vector evenly across all workers, one UDP datagram per worker,
4. workers compute their partial sum and reply with a RESULT datagram,
5. the coordinator merges partial sums and prints `job|done` with `match=1` plus per-worker RTT.

```text
LINKFW-S|job|start|id=1|workers=1|len=1024|expected=334674783
LINKFW-S|job|res|from=192.168.4.2|id=esp32-5AE8|jobid=1|part=334674783|cnt=1024|rtt=6 ms|pending=0
LINKFW-S|job|done|id=1|workers=1|sum=334674783|expected=334674783|match=1|ms=8
LINKFW-S|job|per_worker|idx=0|id=esp32-5AE8|rtt=6 ms|cnt=1024
```

Run it repeatedly from the host: `python3 tools/fleet_smoke.py --port /dev/cu.usbmodem101`.

## Scaling to 8 boards

- Works out of the box for any number of boards on the same AP (tested shape: star via AP).
- One UDP receive loop per board; keep job payloads under ~1300 B per datagram, or chunk
  into multiple JOBs (see `benchmarks/case-02/multiboard/esp32_cluster_transport/` for the
  TCP path used for >1.4 KB payloads).
- The demo vector length is `LF_DEMO_VEC_LEN` (650) in `station_comm.cpp` - the largest
  value that fits a single worker datagram (650 u16 = 1300 B payload); for N workers each
  slice is 650/N.
- The coordinator also does its own work slice if you give it one (currently it only
  distributes + merges, to keep the demo's output checks trivial).

## Limitations / next steps

- Fire-and-forget UDP: a lost JOB/RESULT triggers `job|timeout`; retransmit support is next.
- Fleet table is indexed by position; a peer dropping mid-job shifts per-worker stats (sum
  is unaffected because results are keyed by job id from the datagram).
- `/dev/cu.usbmodem101` flash from an earlier firmware can wedge the USB link of another
  board on the same hub — unplug/replug that board to recover (seen on usbmodem1101).
