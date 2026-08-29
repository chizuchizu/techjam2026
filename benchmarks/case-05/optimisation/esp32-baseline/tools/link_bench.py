#!/usr/bin/env python3
"""link_bench.py - run the ESP32-C3 ESP-NOW 2-node link bandwidth benchmark.

Two XIAO ESP32-C3 boards (server + client, both flashed with esp32-linkbench)
are connected over USB CDC. This script reads BOTH serial ports, waits for the
client to run its 3 payload-size rounds, and reports the achieved
node-to-node bandwidth, packet loss, reliability and RTT.

Usage:
  python3 link_bench.py PORT_SERVER PORT_CLIENT [--dur 60] [--json out.json]

Protocol (printed by the firmware, parsed here):
  LINKFW|role=SERVER|mac=...                        boot line
  LINKFW|role=CLIENT|mac=...                        boot line
  SERVER|alive|rx_pkts=N|rx_bytes=B                 liveness (server)
  CLIENT|P=<P>|N=<N>|sent=<S>|fail=<F>|err=<E>|acked=<A>|bytes=<B>|us=<U>|thr=<T>|rtt_us_med=<R>
  CLIENT|DONE
"""
import argparse, json, re, sys, time
try:
    import serial
except ImportError:
    sys.exit("need pyserial: pip install pyserial")

CLIENT_RE = re.compile(
    r"CLIENT\|P=(\d+)\|N=(\d+)\|sent=(\d+)\|fail=(\d+)\|err=(\d+)\|acked=(\d+)\|"
    r"bytes=(\d+)\|us=(\d+)\|thr=(\d+)(?:\|rtt_us_med=(\d+))?")
SERVER_ALIVE_RE = re.compile(r"SERVER\|alive\|rx_pkts=(\d+)\|rx_bytes=(\d+)")
SERVER_RX_RE = re.compile(r"SERVER\|rx\|pkts=(\d+)\|bytes=(\d+)")
BOOT_RE = re.compile(r"LINKFW\|role=(SERVER|CLIENT)\|mac=([0-9A-F:]+)")


def parse_feed(chunks):
    """Parse raw serial bytes; return (summary, parsed lines). Pure/importable."""
    text = "".join(chunks) if isinstance(chunks, (list, tuple)) else chunks
    summary = {"boot": {}, "rounds": [], "server_last": None, "server_rx": None, "done": False}
    for raw in text.splitlines():
        line = raw.strip()
        m = BOOT_RE.search(line)
        if m:
            summary["boot"][m.group(1)] = m.group(2); continue
        m = SERVER_RX_RE.search(line)
        if m:
            summary["server_rx"] = {"pkts": int(m.group(1)), "bytes": int(m.group(2))}
            continue
        m = SERVER_ALIVE_RE.search(line)
        if m:
            summary["server_last"] = {"rx_pkts": int(m.group(1)), "rx_bytes": int(m.group(2))}
            continue
        m = CLIENT_RE.search(line)
        if m:
            g = [int(x) if x is not None else None for x in m.groups()]
            summary["rounds"].append({
                "P": g[0], "N": g[1], "sent": g[2], "fail": g[3], "err": g[4],
                "acked": g[5], "bytes": g[6], "us": g[7], "thr": g[8], "rtt_us_med": g[9]})
            continue
        if line == "CLIENT|DONE":
            summary["done"] = True
    return summary


def summarize(parsed):
    lines = []
    lines.append("role boot: server=%s client=%s (done=%s)" % (
        parsed["boot"].get("SERVER"), parsed["boot"].get("CLIENT"), parsed["done"]))
    if not parsed["rounds"]:
        lines.append("  no CLIENT round lines captured")
        return "\n".join(lines)
    tot = 0.0
    sr = parsed.get("server_rx") or parsed.get("server_last")
    if sr:
        issued = sum(r["bytes"] for r in parsed["rounds"])
        eff = 100.0 * sr["bytes"] / issued if issued else 0.0
        lines.append("  server ground-truth: rx_pkts=%d rx_bytes=%d (%.1f KB) vs issued %.1f KB -> %.1f%% delivered" % (
            sr["pkts"], sr["bytes"], sr["bytes"] / 1024, issued / 1024, eff))
    lines.append("  payload  sent  fail  err  acked  ack%  rtt_us_med  (KB/s  MB/s)")
    for r in parsed["rounds"]:
        loss = 100.0 * (r["fail"] + r["err"]) / r["N"] if r["N"] else 0.0
        ap = 100.0 * r["acked"] / r["N"] if r["N"] else 0.0
        rtt = r["rtt_us_med"] if r["rtt_us_med"] is not None else -1
        lines.append("  %6d  %4d  %4d  %3d  %5d  %5.1f  %9s   %6.1f %.3f" % (
            r["P"], r["sent"], r["fail"], r["err"], r["acked"], ap,
            ("%.0f" % rtt) if rtt >= 0 else "n/a",
            r["thr"] / 1024.0, r["thr"] / 1e6))
        tot += r["thr"]
    n = len(parsed["rounds"])
    lines.append("  mean useful node-to-node bandwidth: %.0f B/s (%.1f KB/s, %.3f MB/s)" % (
        tot / n, tot / n / 1024, tot / n / 1e6))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("port_server")
    ap.add_argument("port_client")
    ap.add_argument("--dur", type=float, default=60.0, help="max wait seconds")
    ap.add_argument("--reset-client", action="store_true",
                    help="hard-reset the client board (DTR/RTS) before capturing")
    ap.add_argument("--json", default=None, help="write parsed result JSON")
    a = ap.parse_args()

    if a.reset_client:
        rc = serial.Serial(a.port_client, 115200, timeout=1.0)
        time.sleep(0.2)
        rc.setDTR(False); rc.setRTS(True); time.sleep(0.15)
        rc.setRTS(False); rc.setDTR(True); time.sleep(0.15)
        rc.setDTR(False); rc.close()
        time.sleep(1.0)
    ss = serial.Serial(a.port_server, 115200, timeout=0.5)
    sc = serial.Serial(a.port_client, 115200, timeout=0.5)
    time.sleep(0.5)
    ss.reset_input_buffer(); sc.reset_input_buffer()

    def drain(ser):
        n = ser.in_waiting
        return ser.read(n).decode(errors="replace") if n else ""

    buf_s, buf_c = "", ""
    t0, done = time.time(), False
    print("waiting for client rounds (up to %.0f s)..." % a.dur, flush=True)
    while not done and (time.time() - t0) < a.dur:
        buf_c += drain(sc); buf_s += drain(ss)
        p_c = parse_feed(buf_c)
        if "CLIENT|DONE" in buf_c or p_c["done"]:
            done = True
            time.sleep(1.0)          # collect any tail lines
            buf_c += drain(sc); buf_s += drain(ss)
        else:
            time.sleep(0.25)

    parsed = parse_feed(buf_c)
    ps = parse_feed(buf_s)
    # server-side lines may land on the server serial; merge what the client
    # serial did not capture
    parsed["boot"]["SERVER"] = parsed["boot"].get("SERVER") or ps["boot"].get("SERVER")
    parsed["boot"]["CLIENT"] = parsed["boot"].get("CLIENT") or ps["boot"].get("CLIENT")
    parsed["server_last"] = parsed["server_last"] or ps["server_last"]
    parsed["server_rx"] = parsed["server_rx"] or ps["server_rx"]
    ss.close(); sc.close()

    out = {"client_serial": buf_c, "server_serial": buf_s,
           "rounds": parsed["rounds"],
           "server_last": parsed["server_last"],
           "server_rx": parsed["server_rx"],
           "done": parsed["done"]}
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
    print(summarize(parsed))
    if parsed["server_last"]:
        print("server rx: pkts=%d bytes=%d (%.1f KB)" % (
            parsed["server_last"]["rx_pkts"], parsed["server_last"]["rx_bytes"],
            parsed["server_last"]["rx_bytes"] / 1024))


if __name__ == "__main__":
    main()
