#!/usr/bin/env python3
"""Computer-side WiFi master for the esp32-linkbench `link-station` firmware.

Topology:  computer (master) --WiFi/UDP--> ESP32 workers (slaves).  This is the
direct-WiFi alternative to the USB-serial->ESP-NOW bridge documented in
docs/PC_MASTER_WIFI_BRIDGE.md; both put the computer in the master role but
this one talks IP instead of USB.

Wire protocol (matches src/station_comm.cpp, UDP port 42100):
  PING|<seq>                          ->  PONG|<seq>
  JOB|<coord>|<jobid>|SUM16|<len>|<uint16 payload, host byte order>
                                      ->  RESULT|<worker>|<jobid>|SUM16|<sum>|<count>
  BEACON|<id>|<ip>|<mac>|<uptime>|<heap>   (workers broadcast every 2 s)

Usage:
  python3 wifi_master.py discover [--port 42100] [--timeout 5]
  python3 wifi_master.py ping --ip 192.168.4.2 [--port 42100]
  python3 wifi_master.py sum --ip 192.168.4.2 --len 64 [--port 42100]
  python3 wifi_master.py bench --ip 192.168.4.2 --len 650 --n 50 [--port 42100]

Run this from a computer that is on the same L2 network as the boards (e.g.
joined the LINKNET AP hosted by one board). Standard library only.
"""
import argparse, socket, struct, sys, time


def send_recv(ip, port, datagram, wait=1.0):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(wait)
    try:
        s.sendto(datagram, (ip, port))
        t0 = time.perf_counter()
        data, _addr = s.recvfrom(2048)
        return data, time.perf_counter() - t0
    finally:
        s.close()


def discover(port=42100, timeout=5.0):
    """Passively collect BEACON frames broadcast by workers."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    s.bind(("", port))
    s.settimeout(timeout)
    found = {}
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            data, addr = s.recvfrom(2048)
        except socket.timeout:
            break
        fields = data.decode(errors="replace").split("|")
        if fields and fields[0] == "BEACON" and len(fields) >= 5:
            dedup = fields[2] if len(fields) > 2 else addr[0]
            found[dedup] = {"id": fields[1], "ip": fields[2], "mac": fields[3]}
    s.close()
    return list(found.values())


def ping(ip, port=42100):
    seq = int(time.time() * 1000) & 0xFFFF
    data, rtt = send_recv(ip, port, f"PING|{seq}".encode())
    fields = data.decode(errors="replace").split("|")
    if fields and fields[0] == "PONG":
        return rtt, fields[1] if len(fields) > 1 else "?"
    raise RuntimeError(f"unexpected reply: {data!r}")


def sum16(bytes_data):
    # firmware SUM16 adds uint16 words in host byte order; replicate it
    acc = 0
    for i in range(0, len(bytes_data) - 1, 2):
        acc += struct.unpack("<H", bytes_data[i:i + 2])[0]
    return acc


def sumjob(ip, port, length, jobid=None, seed=0x5A5A):
    if not (1 <= length <= 650):
        raise ValueError("length must be 1..650 (one datagram)")
    jobid = jobid if jobid is not None else int(time.time() * 1000) & 0xFFFFFFFF
    payload = bytes([(seed + i * 31) & 0xFF for i in range(length * 2)])
    datagram = b"JOB|pc|%d|SUM16|%d|" % (jobid, length) + payload
    data, rtt = send_recv(ip, port, datagram)
    fields = data.decode(errors="replace").split("|")
    if not (fields and fields[0] == "RESULT" and len(fields) >= 6):
        raise RuntimeError(f"unexpected reply: {data!r}")
    _, wid, rjobid, op, rsum, rcount = fields
    expected = sum16(payload)
    ok = int(rsum) == expected and int(rcount) == length
    return {"worker": wid, "jobid": int(rjobid), "op": op,
            "sum": int(rsum), "expected": expected, "count": int(rcount),
            "rtt_s": rtt, "ok": ok}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["discover", "ping", "sum", "bench"])
    ap.add_argument("--ip", default=None)
    ap.add_argument("--port", type=int, default=42100)
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--len", type=int, default=64)
    ap.add_argument("--n", type=int, default=50)
    a = ap.parse_args(argv)

    if a.cmd == "discover":
        for w in discover(a.port, a.timeout):
            print(f"{w['id']:>16}  {w['ip']:>15}  {w['mac']}")
        return 0

    if not a.ip:
        print("--ip is required for this command", file=sys.stderr)
        return 2

    if a.cmd == "ping":
        rtt, seq = ping(a.ip, a.port)
        print(f"PONG seq={seq} rtt={rtt*1e6:.0f} us")
        return 0

    if a.cmd == "sum":
        r = sumjob(a.ip, a.port, a.len)
        print(f"RESULT worker={r['worker']} jobid={r['jobid']} sum={r['sum']} "
              f"count={r['count']} ok={int(r['ok'])} rtt={r['rtt_s']*1e3:.3f} ms")
        return 0 if r["ok"] else 1

    # bench: N SUM16 jobs, median RTT + datagram throughput (payload only)
    rtts, oks = [], 0
    for _ in range(a.n):
        r = sumjob(a.ip, a.port, a.len)
        rtts.append(r["rtt_s"])
        oks += r["ok"]
    rtts.sort()
    med = rtts[len(rtts) // 2]
    payload_bps = a.len * 2 * a.n / sum(rtts)
    print(f"bench n={a.n} len={a.len} ok={oks}/{a.n} rtt_med={med*1e6:.0f} us "
          f"payload_thr={payload_bps/1024:.1f} KiB/s")
    return 0 if oks == a.n else 1


if __name__ == "__main__":
    raise SystemExit(main())
