#!/usr/bin/env python3
"""
time_cluster.py - measure the two-board distributed forward with minimal USB.

The accuracy path (run_cluster_e2e.py) has to move a 32 KB input and a 32 KB
output frame per board per seed. On a host whose usbip bridge stalls under
load that is exactly what gets lost. Timing does not depend on the input
values, so this driver skips the frames entirely: it assigns roles, then runs
one device-side forward at a time ('T' 1) and reads the board's own record of
it back with a status probe. Every exchange is a few bytes.

Usage: python3 tools/time_cluster.py [--reps 5]
"""
import argparse
import glob
import re
import sys
import time

import serial


def probe(ser, cmd=b"M", wait=1.2):
    """Ask a board something over an already-open handle.

    The handle is kept open for the whole run: reopening a C3's USB-CDC port
    toggles DTR/RTS, which is the board's reset line."""
    ser.reset_input_buffer()
    ser.write(cmd)
    time.sleep(wait)
    return ser.read(ser.in_waiting or 1).decode(errors="replace")


def fields(text):
    return {k: int(v) for k, v in re.findall(r"(\w+)=(-?\d+)", text)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--baud", type=int, default=115200)
    args = ap.parse_args()

    ports = sorted(glob.glob("/dev/ttyACM*"))
    if len(ports) < 2:
        print(f"need two boards, found {ports}")
        return 2

    sers = [serial.Serial(p, args.baud, timeout=0.4) for p in ports]
    time.sleep(0.6)
    st = [fields(probe(s, b"M")) for s in sers]
    print(f"[ports] {ports[0]} node={st[0].get('node')} | {ports[1]} node={st[1].get('node')}")

    if st[0].get("link") != 1 or st[1].get("link") != 1:
        print("[link] assigning roles ...")
        sers[0].write(b"N0")
        time.sleep(1.5)
        sers[1].write(b"N1")
        time.sleep(16)
        st = [fields(probe(s, b"M")) for s in sers]
    if st[0].get("link") != 1 or st[1].get("link") != 1:
        print(f"peer link not up: {st}")
        return 2
    # follow the roles, not the port order
    if st[0].get("node") == 1:
        ports.reverse(); st.reverse(); sers.reverse()
    print(f"[link] up, node0={ports[0]} node1={ports[1]}")

    walls, comps, links = [], [], []
    for r in range(args.reps):
        seq0 = [fields(probe(s, b"M")).get("seq", 0) for s in sers]
        for s in sers:                        # 'T' 1: one device-side forward
            s.write(b"T" + bytes([1]))
        time.sleep(3.0)

        got = []
        for s, s0 in zip(sers, seq0):
            for _ in range(12):
                f = fields(probe(s, b"M"))
                if f.get("seq", 0) > s0 and f.get("last_us"):
                    got.append(f)
                    break
                time.sleep(1.0)
            else:
                got.append(None)
        if any(g is None for g in got):
            print(f"rep {r}: measurement not retrievable, skipping")
            continue
        wall = max(g["last_us"] for g in got) / 1e6
        comp = max(g["last_comp"] for g in got) / 1e6
        link = max(g["last_link"] for g in got) / 1e6
        walls.append(wall); comps.append(comp); links.append(link)
        print(f"rep {r}: wall={wall:.3f}s comp={comp:.3f}s link={link:.3f}s")

    if not walls:
        print("no measurements captured")
        return 1
    walls.sort()
    med = walls[len(walls) // 2]
    print(f"\nmedian distributed forward: {med:.3f} s over {len(walls)} reps "
          f"(min {min(walls):.3f}, max {max(walls):.3f})")
    print(f"compute {max(comps):.3f} s max, link wait {max(links):.3f} s max")
    return 0


if __name__ == "__main__":
    sys.exit(main())
