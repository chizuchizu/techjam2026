# WiFi setup & how it works — computer master, 2x ESP32-C3 slaves

This page explains the WiFi layer used by the two-board ESP32-C3 benchmarks:
the computer is the **master**, and the two ESP32-C3 boards are **slaves**.

Hardware: 2x Seeed XIAO ESP32-C3 (160 MHz, 320 KB SRAM, 4 MB flash), both
plugged into the computer by USB-C.

## Choose a topology

| Mode | When | Board↔board | Computer↔board |
|---|---|---|---|
| **A. Serial bridge (ESP-NOW)** | Lowest latency, no router, tiny RAM | ESP-NOW (~0.85 ms RTT) | USB-CDC serial to board A |
| **B. Direct-WiFi (UDP station)** | Computer must talk IP to boards | UDP (5–9 ms RTT) | UDP on the LINKNET Wi-Fi |

Mode A is the default for benchmark work: it adds no lwIP/TCP to the
compute build, and the link-only firmware itself uses only ~10-13% of SRAM
(see footprint below). Mode B is for when the computer itself must send
packets over Wi-Fi.

Firecrawl research backing the transport choice:
`research/computer_master_wifi_research.md`.

---

## Mode A — serial bridge over ESP-NOW

### Topology

    computer (master) -- USB-CDC serial --> ESP32-A (bridge, "server")
                                                |
                                                |  ESP-NOW over WiFi, channel 1
                                                |  no router, no password
                                                v
                                           ESP32-B (worker, "client")

The computer sends short commands over USB to board A, board A relays payloads
to board B over ESP-NOW, B replies, A returns the result to the computer.

### Why ESP-NOW

- No access point, no association, no DHCP — boards talk in <1 ms after boot.
- Lower CPU/flash cost than a full Wi-Fi IP stack (Espressif: "occupies less
  CPU and flash resource").
- Fits in ~34–42 KB RAM (see footprint below).

### Setup (one-time)

1. Install [PlatformIO](https://platformio.org/install) (`pio` on PATH).
2. `cd esp32-linkbench`
3. Plug in both boards and note their serial ports:
   `ls /dev/cu.usbmodem*` (e.g. `cu.usbmodem101`, `cu.usbmodem1101`).

### Build & flash

    cd esp32-linkbench
    pio run -e link-server -e link-client            # build both images
    pio run -e link-server -t upload --upload-port /dev/cu.usbmodem101   # board A
    pio run -e link-client -t upload --upload-port /dev/cu.usbmodem1101  # board B

Board A (`link-server`) starts a hidden-channel softAP `LINKFAST` on channel 1.
Board B (`link-client`) joins it, discovers board A's MAC with an ESP-NOW probe,
then runs the built-in PING + STREAM demos once and prints `CLIENT|READY`.

### Using it (serial commands, 115200 baud, board B's port)

Open board B's serial monitor (`pio device monitor -b 115200 --port
/dev/cu.usbmodem1101`) and send one of these characters + Enter:

| Command | Action |
|---|---|
| `B` | run PING + STREAM benchmarks and print results |
| `P` | run PING latency profile only |
| `S` | run STREAM bandwidth test only |
| `W <N>` | **relay N bytes** (1..240) through board A to board B and back, verify bit-exactness, print `RELAY|s|N=...|rtt_us=...|ok=1` |

Example `W` output (this is the computer-as-master data round-trip):

    RELAY|s|N=240|rtt_us=1342|ok=1|seq=30718155
    CLIENT|DONE

### How Mode A works

ESP-NOW frames are sent as 802.11 action frames; the app payload is the frame
body plus a 1-byte frame type. All timestamps use `esp_timer_get_time()`.

Frame types in `src/main.cpp`:

| Type | Value | Meaning |
|---|---:|---|
| `FT_PROBE` | 0x81 | client discovers server MAC (broadcast) |
| `FT_PROBE_ACK` | 0x82 | server replies with magic |
| `FT_PING` / `FT_PING_ACK` | 0x01 / 0x02 | latency profile (seq + 3 timestamps) |
| `FT_STREAM` / `FT_STREAM_ACK` | 0x03 / 0x04 | bandwidth test (seq + us timestamp) |
| `FT_RELAY` / `FT_RELAY_ACK` | 0x11 / 0x12 | **generic PC-master payload relay** |
| `FT_RESET_ST` / `FT_STREAM_END` | 0xF0 / 0xFF | stream counter reset / end marker |

Relay flow (`W N`):

1. Computer writes `W 240\n` on board B's serial.
2. Board B builds `FT_RELAY | u32 seq | 240 bytes pattern`, unicasts to board A,
   records `t0`.
3. Board A's receive callback flips the type byte to `FT_RELAY_ACK` and sends
   the whole frame straight back (no allocations, static 250 B buffer).
4. Board B compares the echoed 240 bytes against the sent pattern in
   `FT_RELAY_ACK`, records `t3`, and prints `RELAY|s|...|rtt_us=(t3-t0)|ok=1`.

Measured relay round-trip (board↔board wireless hop, excludes host serial
time), fresh flash 2026-08-30, channel 1:

| N bytes | 1 | 16 | 64 | 240 |
|---|---:|---:|---:|---:|
| RTT | 1153 us | 1226 us | 1263 us | 1342 us |

All four runs `ok=1` (bit-exact). Same session PING profile: median RTT
0.99–1.01 ms across P=16..240, 0/2000 lost.

---

## Mode B — direct-WiFi master over UDP (`link-station`)

### Topology

    computer (master) -- Wi-Fi/UDP --> ESP32-A (board)    (all three on LINKNET)
                           `----- Wi-Fi/UDP --> ESP32-B (board)

The computer itself is a network peer. One board self-hosts a softAP named
`LINKNET` (password `linkfast123`); the other board joins it; the computer also
joins `LINKNET`. Everything communicates over UDP port 42100.

### Setup

1. Edit the SSID/password if you want a specific network, else the boards
   fall back to hosting/joining `LINKNET` automatically:

       pio run -e link-station -t upload --upload-port /dev/cu.usbmodem101
       pio run -e link-station -t upload --upload-port /dev/cu.usbmodem1101

2. Power-cycle both boards. The first to boot hosts `LINKNET` (192.168.4.1);
   the second joins and DHCP-gets 192.168.4.2.
3. On the computer, join Wi-Fi `LINKNET` (password `linkfast123`).
4. Run the master tool:

       cd esp32-linkbench
       python3 tools/wifi_master.py discover
       python3 tools/wifi_master.py ping  --ip 192.168.4.2
       python3 tools/wifi_master.py sum   --ip 192.168.4.2 --len 64
       python3 tools/wifi_master.py bench --ip 192.168.4.2 --len 650 --n 50

### How Mode B works (UDP protocol, port 42100)

| Direction | Datagram | Reply |
|---|---|---|
| any → board | `PING|<seq>` | `PONG|<seq>` |
| master → board | `JOB|<coord>|<jobid>|SUM16|<len>\|<uint16 payload>` | `RESULT|<worker>|<jobid>|SUM16|<sum>\|<count>` |
| board → all | `BEACON|<id>|<ip>|<mac>|<uptime>|<heap>` (every 2 s) | — |

`JOB` payload is `len` uint16 words in host byte order (max 650 words, one
datagram). The board sums them and replies with the sum + count; the master
checks both. Firmware lives in `src/station_comm.cpp`; the wire protocol is
implemented exactly in `tools/wifi_master.py` (standard library only, no
dependencies).

---

## Footprint (Arduino core 3.0.7 / ESP-IDF 5.1.0)

| env | RAM | Flash (app image) |
|---|---:|---:|
| link-server | 34,028 B (10.4% of 327,680 B) | 926,328 B |
| link-client | 42,084 B (12.8% of 327,680 B) | 928,858 B |

RAM is the scarce resource on the C3. These figures are the **link-only**
firmware: it carries no transformer compute, so its total RAM is tiny. WiFi does
**not** fit inside the full-forward compute node — measured on this bench, the
full-sequence build uses ~274 KB, WiFi+lwIP adds ~85 KB static plus ~69 KB heap,
and the combined image overflows `dram0_0_seg` (see
[`docs/WIFI_ON_A_COMPUTE_NODE.md`](../../docs/WIFI_ON_A_COMPUTE_NODE.md)).
That is exactly why the topology here keeps WiFi **out** of the compute
firmware: boards run this small link firmware when they act as transport, and
the compute builds keep their existing USB/serial path. Flash is dominated by
the prebuilt Wi-Fi + ESP-NOW + USB-CDC driver blobs, not by our code.

## Troubleshooting

- **"chip stopped responding" / upload fails** after a board ran link firmware:
  use esptool with `--before usb_reset`, or power-cycle the board and retry the
  upload.
- **`CLIENT|AP_JOIN_FAIL`**: board A's softAP is not up yet; power-cycle both
  boards (server first), or check channel interference and rebuild.
- **`link-station` boards can't see each other**: some guest networks segment
  clients (e.g. NTUGUEST). Use your own hotspot/router, or let boards self-host
  `LINKNET` by leaving the configured SSID unreachable.
- **Relay `ok=0`**: rare frame corruption/truncation; re-send. A lost reply
  prints `RELAY|lost|N=..`.
