#!/usr/bin/env python3
"""fake_worker.py - a host-side ESP32 FLEET worker for scaling/split tests.

Lets you test the coordinator's N-way parallel split WITHOUT extra ESP32 boards:
join your machine to the fleet's LINKNET WiFi and run this; the real coordinator
discovers this peer (via BEACON), sends it a JOB slice, and this worker computes
the partial SUM16 and replies RESULT - exactly like a real board.

Checks to make:
tail -f /dev/cu.usbmodemXXXX   (the real coordinator's serial)
Expect: job|start|workers=N ... job|done|...|match=1 with N-1 fake workers.

Usage:
  networksetup -setairportnetwork en0 LINKNET linkfast123   # join the fleet WLAN
  python3 tools/fake_worker.py --id fake-1                  # one fake worker
  python3 tools/fake_worker.py --id fake-1 --id fake-2       # two fake workers
"""
import argparse
import socket
import struct
import sys
import time


def mac_of(ident):
    # deterministic high MAC so the peer never becomes coordinator
    # (coordinator = lowest MAC; real boards are 64:E8:33:8A:xx)
    b = bytes(ident, "utf-8")
    h = (sum(b) * 2654435761) & 0xFFFFFFFFFFFFFFFF
    m = bytearray(6)
    for i in range(6):
        m[5 - i] = h & 0xFF
        h >>= 8
    m[0] |= 0xFE                    # top byte >= 0xFE -> always high
    return ":".join("%02X" % x for x in m)


def local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.168.4.1", 42100))   # the LINKNET AP
        return s.getsockname()[0]
    except Exception:
        return "0.0.0.0"
    finally:
        s.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", action="append", required=True,
                    help="node id for one fake worker (repeat for several)")
    ap.add_argument("--port", type=int, default=42100)
    ap.add_argument("--beacon-ms", type=int, default=1000,
                    help="heartbeat period (fleet TTL is 8 s)")
    ap.add_argument("-t", "--timeout", type=float, default=0,
                    help="run for N seconds then stop (0 = forever)")
    args = ap.parse_args()

    workers = {}
    macs = {}
    for ident in args.id:
        workers[ident] = {"mac": mac_of(ident), "jobs": 0, "sum": 0}
        macs[ident] = workers[ident]["mac"]
        print(f"[fake] worker {ident} MAC={macs[ident]}", flush=True)

    ip = local_ip()
    print(f"[fake] local ip on LINKNET: {ip}", flush=True)

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", args.port))
    s.settimeout(0.25)

    end = time.time() + args.timeout if args.timeout > 0 else None
    next_beacon = 0
    uptime = 0
    try:
        while True:
            now = time.time()
            if end and now >= end:
                break
            # heartbeat: send one BEACON for every fake worker
            if now >= next_beacon:
                for ident, w in workers.items():
                    payload = "BEACON|%s|%s|%s|%d|%d" % (
                        ident, ip or "0.0.0.0", w["mac"],
                        int(uptime * 1000), 0)
                    s.sendto(payload.encode(), ("255.255.255.255", args.port))
                uptime = (time.time() - now) + uptime
                next_beacon = now + args.beacon_ms / 1000.0
            # receive: JOB -> compute -> RESULT back to sender
            try:
                data, addr = s.recvfrom(65535)
            except socket.timeout:
                continue
            if data[:4] == b"JOB|":
                try:
                    fields = data.decode(errors="replace").split("|")
                    # JOB|<coord>|<jobid>|SUM16|<len>|payload(u16, LE)
                    f_coord, f_jobid, f_op, f_len = fields[1], fields[2], fields[3], fields[4]
                    n = int(f_len)
                    if f_op != "SUM16" or n <= 0:
                        continue
                    payload = data.split(b"|", 5)[5]
                    if len(payload) < n * 2:
                        print(f"[fake] short payload {len(payload)} for {n} u16", flush=True)
                        continue
                    vals = struct.unpack("<%dH" % n, payload[: n * 2])
                    total = sum(vals)
                    # one worker *per job*: pick the id whose turn (round-robin)
                    ident = None
                    for k in workers:
                        ident = k
                        break
                    workers[ident]["jobs"] += 1
                    workers[ident]["sum"] = total
                    reply = "RESULT|%s|%s|SUM16|%d|%d" % (ident, f_jobid, total, n)
                    s.sendto(reply.encode(), (addr[0], args.port))
                    print(f"[fake:{ident}] job {f_jobid} len={n} sum={total} "
                          f"to={f_coord}@({addr[0]})", flush=True)
                except Exception as e:
                    print(f"[fake] parse error: {e}", flush=True)
    finally:
        s.close()
    print(f"[fake] done; {sum(w['jobs'] for w in workers.values())} jobs handled", flush=True)


if __name__ == "__main__":
    sys.exit(main())
