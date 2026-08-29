# ESP32 station mode — same-WiFi setup (`link-station`)

Configures ESP32 boards so they **talk to each other over one WiFi network**
by IP: each board advertises its IP via mDNS, auto-discovers the other boards
via UDP broadcast, and can ping them (RTT). One identical firmware image runs
on hub **and** clients — the role is chosen at build time with one flag.

Repo: `esp32-linkbench/` on branch `linkfast` (github.com/chizuchizu/techjam2026).

## Concepts

| Role  | What it does | Default IP | Build flag |
|-------|--------------|-----------|------------|
| **hub**   | Hosts the WiFi network `LINKNET` (pw `linkfast123`); other boards join it | `192.168.4.1` | `-DLF_FORCE_AP=1` |
| **client**| Joins a WiFi network (your venue/hotspot, or the hub's `LINKNET`) | DHCP (`192.168.4.x` on a hub) | `-DLF_FORCE_AP=0` (default) |
| auto | Tries to join the configured network; if none exists after ~15 s it becomes a hub itself | `192.168.4.1` | default build (no flag) |

Every board, in any role:

- advertises `esp32-<mac.low2hex>.local` via mDNS (also seen by `pio device list --mdns`)
- broadcasts a UDP beacon every 2 s on port **42100** → all boards on the same
  network learn each other's IP automatically
- answers UDP `PING`/`PONG`; press **`H`** on a board's serial to measure RTT
  (5–9 ms verified between two XIAO ESP32-C3 boards)

## Prerequisites

- 2 or more ESP32 boards (verified on Seeed XIAO ESP32-C3), each with USB cable
- PlatformIO Core 6.x (`pio --version`)
- Repo: `git clone https://github.com/chizuchizu/techjam2026 && cd techjam2026 && git checkout linkfast`
- Board ports (macOS example): `/dev/cu.usbmodem101`, `/dev/cu.usbmodem1101`.

## "Do I need to set IP addresses?" — No. It's automatic. (read this first)

If you have never done IP setup before: **you do not have to do any of it.**
This guide never asks you to type an IP address. Here is the whole story:

1. **The hub board creates the network.** When it starts, it makes a small
   WiFi network called `LINKNET`. Every ESP32 that creates a hotspot
   automatically uses IP `192.168.4.1` for itself, and automatically gives out
   `192.168.4.2`, `192.168.4.3`, … to other boards that join. You never set
   these numbers; the board does it on its own.

2. **Client boards just "connect to the WiFi".** A client joins a WiFi network
   exactly like your phone joins your home WiFi. The network — the hub, or
   your router / phone hotspot — automatically gives the board an address
   (usually `192.168.4.2`, `192.168.4.3`, …). Again: no typing.

3. **Boards find each other's IPs automatically.** Every board shouts "here I
   am" over the network every 2 seconds, and the other boards hear it and
   remember where the shout came from. So nobody needs to write down anyone's
   IP beforehand.

So the **only** thing you configure is the *WiFi name* and *password*, in one
file. IP numbers are handled by the boards themselves.

### Where do I type the WiFi name and password?

In `esp32-linkbench/platformio.ini`, inside the `[env:link-station-hub]` /
`[env:link-station-client]` sections. Two lines:

```ini
-DLF_WIFI_SSID="LINKNET"          ; WiFi name  (SSID)
-DLF_WIFI_PASS="linkfast123"      ; password
```

- **Hub board**: SSID/PASS stay `LINKNET` / `linkfast123`, and keep
  `-DLF_FORCE_AP=1`. It makes the network. Done.
- **Client board**: set SSID/PASS to the network it should join — either the
  hub's `LINKNET` / `linkfast123`, or your venue WiFi (SSID + its password).
  Keep `-DLF_FORCE_AP=0`. Done.

### How do I know it worked? (how to SEE the IPs)

Look at the board's serial output (open it with `pio device monitor -p <port>`).
A line like this means the board got its address:

```
LINKFW-S|connected|ip=192.168.4.2|rssi=-5
```

Or press the letter `I` on that board's serial — it prints your IP, your
name, and the list of other boards it found:

```
LINKFW-S|info|ip=192.168.4.2|...|host=esp32-B7A8.local|...|peer0=192.168.4.1|
```

If you can see `peer0=...` with another board's IP, the boards found each
other and there is nothing left to set up. That's the "IP setup" — done.

(If you're curious: the automatic "please give me an address" step is called
DHCP, and the "here I am" shout is mDNS/broadcast. You don't have to remember
those words to use this guide.)

### Example: what your friend will see when it works

Hub board (after flashing + booting):
```
LINKFW-S|ap|ssid=LINKNET|ip=192.168.4.1
LINKFW-S|mdns|host=esp32-XXXX.local|port=42100
LINKFW-S|READY
```

Client board (after flashing + booting):
```
LINKFW-S|connected|ip=192.168.4.2|rssi=-5
LINKFW-S|rx|from=192.168.4.1|BEACON|esp32-XXXX|192.168.4.1
```

Press `H` on the hub or client and you get the round-trip time, e.g.
`LINKFW-S|rtt|from=192.168.4.2|6 ms`. Working = done.

## 1. Setup a board as HUB (hosts the network)

The hub creates the WiFi network. Everyone else joins it. Flash with
`-DLF_FORCE_AP=1`. Recommended `platformio.ini` env:

```ini
[env:link-station-hub]
build_flags =
  -DBUILD_ROLE_STATION
  -DLF_WIFI_SSID=\"LINKNET\"
  -DLF_WIFI_PASS=\"linkfast123\"
  -DLF_FORCE_AP=1          # <-- hub role: always host the AP
```

Flash + verify:

```bash
pio run -e link-station-hub -t upload --upload-port /dev/cu.usbmodem101
```

Expected serial output (open the terminal, e.g. `pio device monitor -p /dev/cu.usbmodem101`):

```
LINKFW-S|boot|role=STATION|fw=station-comm-v3
LINKFW-S|ap|ssid=LINKNET|ip=192.168.4.1
LINKFW-S|mdns|host=esp32-XXXX.local|port=42100
LINKFW-S|READY
```

`192.168.4.1` + `mdns|host=...` + `READY` = hub is up.

## 2. Setup a board as CLIENT (joins the network)

The client joins the hub's `LINKNET` (or your venue WiFi). Keep the default
`-DLF_FORCE_AP=0` and point SSID/PASS at the network to join.

Joining the **hub** (`LINKNET`):

```ini
[env:link-station-client]
build_flags =
  -DBUILD_ROLE_STATION
  -DLF_WIFI_SSID=\"LINKNET\"
  -DLF_WIFI_PASS=\"linkfast123\"
  -DLF_FORCE_AP=0
```

Joining a **normal WiFi** (venue hotspot/router):

```ini
[env:link-station-venue]
build_flags =
  -DBUILD_ROLE_STATION
  -DLF_WIFI_SSID=\"MyVenueWiFi\"
  -DLF_WIFI_PASS=\"thepassword\"
  -DLF_FORCE_AP=0
```

Flash + verify:

```bash
pio run -e link-station-client -t upload --upload-port /dev/cu.usbmodem1101
```

Expected serial output (client):

```
LINKFW-S|conn_pin|ssid=LINKNET|bssid=XX:XX:XX:XX:XX:XX|ch=1   # auto-pinned to the AP
LINKFW-S|connected|ip=192.168.4.2|rssi=-5
LINKFW-S|mdns|host=esp32-XXXX.local|port=42100
LINKFW-S|READY
LINKFW-S|rx|from=192.168.4.1|BEACON|esp32-XXXX|192.168.4.1     # hub found automatically
```

`connected|ip=192.168.4.2` + `rx|...|BEACON|...` = the client joined and
discovered the hub.

## 3. Auto mode (one image, self-healing — optional)

If you omit `-DLF_FORCE_AP` entirely, every board runs the same image with
SSID/PASS pre-filled. The first board to boot creates `LINKNET`; later boards
join it. No role decisions needed — useful for demos, but use explicit
hub/client envs for deterministic setups.

## 4. Verify communication (does it actually work?)

On the **client** serial, press `H`:

```
LINKFW-S|ping|targets=1
LINKFW-S|rx|from=192.168.4.1|PONG|1
LINKFW-S|rtt|from=192.168.4.1|6 ms     # round-trip to the hub
```

On the **hub** serial, press `H` (it knows the client too):

```
LINKFW-S|rtt|from=192.168.4.2|5 ms
```

Serial commands:

| Key | Action |
|-----|--------|
| `S` | scan nearby WiFi networks (SSID / RSSI / encryption / channel) |
| `I` | own IP, MAC, mDNS hostname, and full peer list |
| `H` | PING every known peer, print RTT in ms |

Troubleshooting:

- **`conn_pin` never prints / timeout** → board can't see the network. Press `S`,
  confirm the SSID is listed; check SSID/PASS spelling.
- **Connected but `targets=0` when pressing `H`** → wait a few seconds for the
  beacons (they repeat every 2 s). If still nothing, the network isolates
  clients (see next point).
- **Client-isolated network** → some guest/campus networks (e.g. NTUGUEST)
  give boards IPs but block client-to-client traffic. Boards can't reach each
  other even though both are "on WiFi". Symptom: both `connected` but no
  beacons/rx ever. Fix: use your own hotspot/router or the hub mode above.
- **mDNS not resolving** → use the UDP/IP path (the `rx` lines show real IPs);
  mDNS is a convenience. `pio device list --mdns` only works from a machine on
  the SAME network as the boards.

## Reference: build flags

| Flag | Default | Meaning |
|------|---------|---------|
| `-DLF_WIFI_SSID=\"...\"` | `YOUR_WIFI_SSID` | STA network / hub SSID to join |
| `-DLF_WIFI_PASS=\"...\"` | `YOUR_WIFI_PASS` | password (use `\"\"` for open networks) |
| `-DLF_FORCE_AP=0/1` | `0` | `1` = hub (always host `LINKNET`), `0` = client |
| `-DLF_AP_SSID` / `-DLF_AP_PASS` | `LINKNET` / `linkfast123` | hub AP name/password |
| `-DLF_UDP_PORT` | `42100` | discovery + ping UDP port |
| `-DLF_BEACON_MS` | `2000` | beacon interval |

Implementation: `src/station_comm.cpp`. ESP-NOW (0.85 ms RTT) docs stay as-is:
`README.md`, `docs/OPTIMIZATION_GUIDE.md`, `docs/ESP32_LINKFAST_FINAL_REPORT.md`.
