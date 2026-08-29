#!/usr/bin/env python3
"""run_linkfast.py - drive/parse the esp32-linkfast 2-node ESP-NOW benchmark.

Usage:
  python3 run_linkfast.py PORT_SERVER PORT_CLIENT [-t DURATION] [--mode NAME] [--logdir DIR]
Optional: --reset-client / --reset-server hard-reset that board first.

Parses:
  PING|s|P=..|r=..|rtt=..|t0=..|sr=..|cs=..|sc=..|snr=..
  PING|done|P=..|n=..|lost=..|rtt_med|rtt_min|rtt_max|rtt_p95|jit|srv_med|cs_med|sc_med
  STREAM/CLIENT lines (linkbench-compatible)
  SERVER|rx ground truth
"""
import argparse, json, os, re, sys, time
import serial

PING_S_RE = re.compile(
    r"PING\|s\|P=(\d+)\|r=(\d+)\|rtt=(\d+)\|t0=(\d+)\|sr=(\d+)\|cs=(\d+)\|sc=(\d+)\|snr=(-?\d+)")
PING_DONE_RE = re.compile(
    r"PING\|done\|P=(\d+)\|n=(\d+)\|lost=(\d+)\|rtt_med=(\d+)\|rtt_min=(\d+)\|rtt_max=(\d+)\|rtt_p95=(\d+)\|jit=(\d+)\|srv_med=(\d+)\|cs_med=(\d+)\|sc_med=(\d+)")
CLIENT_RE = re.compile(
    r"CLIENT\|P=(\d+)\|N=(\d+)\|sent=(\d+)\|fail=(\d+)\|err=(\d+)\|acked=(\d+)\|bytes=(\d+)\|us=(\d+)\|thr=(\d+)\|rtt_us_med=(\d+)")
BOOT_RE = re.compile(r"LINKFW\|role=(SERVER|CLIENT)\|mac=([0-9A-F:]+)\|ch=(\d+)\|mode=(\w+)")
ALIVE_RE = re.compile(r"SERVER\|alive\|rx_pkts=(\d+)\|rx_bytes=(\d+)")
RX_RE = re.compile(r"SERVER\|rx\|pkts=(\d+)\|bytes=(\d+)")


def parse(text):
    out = {"boot": {}, "ping_done": [], "ping_samples": 0, "stream": [], "server_last": None, "server_rx": None}
    for raw in text.splitlines():
        line = raw.strip()
        m = BOOT_RE.search(line)
        if m:
            out["boot"][m.group(1)] = {"mac": m.group(2), "ch": int(m.group(3)), "mode": m.group(4)}
            continue
        m = PING_DONE_RE.search(line)
        if m:
            g = [int(x) for x in m.groups()]
            out["ping_done"].append(dict(zip(
                ["P","n","lost","rtt_med","rtt_min","rtt_max","rtt_p95","jit","srv_med","cs_med","sc_med"], g)))
            continue
        m = PING_S_RE.search(line)
        if m:
            out["ping_samples"] += 1
            continue
        m = CLIENT_RE.search(line)
        if m:
            g = [int(x) for x in m.groups()]
            out["stream"].append(dict(zip(["P","N","sent","fail","err","acked","bytes","us","thr","rtt_us_med"], g)))
            continue
        m = RX_RE.search(line)
        if m:
            out["server_rx"] = {"pkts": int(m.group(1)), "bytes": int(m.group(2))}
            continue
        m = ALIVE_RE.search(line)
        if m:
            out["server_last"] = {"pkts": int(m.group(1)), "bytes": int(m.group(2))}
    return out


def reset_board(port):
    s = serial.Serial(port, 115200, timeout=1.0); time.sleep(0.2)
    s.setDTR(False); s.setRTS(True); time.sleep(0.15)
    s.setRTS(False); s.setDTR(True); time.sleep(0.15)
    s.close(); time.sleep(1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("port_server"); ap.add_argument("port_client")
    ap.add_argument("-t", "--timeout", type=float, default=120.0)
    ap.add_argument("--mode", default="bench")
    ap.add_argument("--reset-server", action="store_true")
    ap.add_argument("--reset-client", action="store_true")
    ap.add_argument("--raw-dir", default=".logs")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    os.makedirs(a.raw_dir, exist_ok=True)
    if a.reset_client: reset_board(a.port_client)
    if a.reset_server: reset_board(a.port_server)
    time.sleep(0.5)
    ss = serial.Serial(a.port_server, 115200, timeout=0.5)
    sc = serial.Serial(a.port_client, 115200, timeout=0.5)
    time.sleep(0.5)
    ss.reset_input_buffer(); sc.reset_input_buffer()

    def drain(ser):
        n = ser.in_waiting
        return ser.read(n).decode(errors="replace") if n else ""

    buf_s, buf_c, t0 = "", "", time.time()
    done = False
    while not done and (time.time() - t0) < a.timeout:
        buf_c += drain(sc); buf_s += drain(ss)
        if "CLIENT|DONE" in buf_c:
            done = True; time.sleep(1.0)
            buf_c += drain(sc); buf_s += drain(ss)
        else:
            time.sleep(0.2)
    ss.close(); sc.close()
    os.makedirs(a.raw_dir, exist_ok=True)
    with open(os.path.join(a.raw_dir, f"client_{a.mode}.log"), "w") as f: f.write(buf_c)
    with open(os.path.join(a.raw_dir, f"server_{a.mode}.log"), "w") as f: f.write(buf_s)
    rid = f"{a.mode}_{int(time.time())}"
    p = parse(buf_c)
    ps = parse(buf_s)
    p["boot"].setdefault("SERVER", ps["boot"].get("SERVER"))
    p["server_last"] = p["server_last"] or ps["server_last"]
    p["server_rx"] = p["server_rx"] or ps["server_rx"]

    print(f"== {a.mode} | server={p['boot'].get('SERVER')} client={p['boot'].get('CLIENT')} ==")
    print("  PING latency (us):  P     n  lost  med  min  max  p95   jit  srv_med  cs_med  sc_med")
    for d in p["ping_done"]:
        print("  %5d  %6d %5d %5d %5d %5d %5d %5d %8d %7d %7d" % (
            d["P"], d["n"], d["lost"], d["rtt_med"], d["rtt_min"], d["rtt_max"],
            d["rtt_p95"], d["jit"], d["srv_med"], d["cs_med"], d["sc_med"]))
    if p["stream"]:
        print("  STREAM:  P   N   sent fail err acked   thr(KB/s)  rtt_med(us)")
        for d in p["stream"]:
            print("  %5d %4d %5d %4d %3d %5d %9.1f %10d" % (
                d["P"], d["N"], d["sent"], d["fail"], d["err"], d["acked"],
                d["thr"]/1024.0, d["rtt_us_med"]))
    if p["server_rx"]:
        sr = p["server_rx"]
        print(f"  server ground truth rx: pkts={sr['pkts']} bytes={sr['bytes']}")
    if a.json:
        json.dump(p, open(a.json, "w"), indent=1)
    return p


if __name__ == "__main__":
    main()
