#!/usr/bin/env python3
"""
device_test.py - drive the ESP32-C3 firmware over serial and verify outputs.

Protocol (src/main.cpp): 'M' -> mode line; 'R' -> read 65536 B input,
one forward, stream 65536 B output + "END ..."; 'T <n>' -> timed forwards.

Usage:
  python3 tools/device_test.py /dev/cu.usbmodem2101 [--seeds 0 1 2 3 4] \\
      [--baud 115200] [--reps 3]

For each seed: send input_<s>.bin, capture output, check the exact torch gate
with ref_<s>.bin (atol=0.002, rtol=0.02). Prints per-seed PASS/FAIL.
"""
import argparse
import pathlib
import sys
import time

import numpy as np
import serial  # pyserial

N = 128 * 128
ATOL, RTOL = 0.002, 0.02


def read_until(ser, token: bytes, timeout: float = 30.0) -> bytes:
    buf = b""
    t0 = time.time()
    while token not in buf:
        if time.time() - t0 > timeout:
            raise TimeoutError(f"no {token!r} within {timeout}s; got {buf[-80:]!r}")
        chunk = ser.read(ser.in_waiting or 1)
        if chunk:
            buf += chunk
    return buf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("port")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--root", default=".", help="project root (testdata lives here)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    ser = serial.Serial(args.port, args.baud, timeout=1.0)
    time.sleep(0.5)
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    ser.write(b"M")
    mode_line = read_until(ser, b"\n", timeout=5.0)
    print(f"[device] {mode_line.strip().decode()}")
    ser.reset_input_buffer()

    total_fail = 0
    for s in args.seeds:
        inp = (root / "testdata" / f"input_{s}.bin").read_bytes()
        assert len(inp) == N * 4
        ref = np.fromfile(root / "testdata" / f"ref_{s}.bin", dtype="<f4")
        ref = ref.astype(np.float64)

        ser.reset_input_buffer()
        ser.write(b"R")
        ser.write(inp)
        out = b""
        t0 = time.time()
        while len(out) < N * 4:
            if time.time() - t0 > 120:
                print(f"seed {s}: TIMEOUT waiting for output")
                return 2
            chunk = ser.read(N * 4 - len(out))
            if chunk:
                out += chunk
        tail = read_until(ser, b"\n", timeout=20.0)
        got = np.frombuffer(out, dtype="<f4").astype(np.float64)

        tinfo = ""
        if b"us=" in tail:
            try:
                us = int(tail.split(b"us=", 1)[1].split(b"\n", 1)[0])
                tinfo = f" {us/1e6:.3f}s fwd"
            except Exception:
                pass

        d = np.abs(got - ref)
        ok = (d <= ATOL) | (d <= RTOL * np.abs(ref))
        nfail = int((~ok).sum())
        total_fail += nfail
        print(f"seed {s}: fails={nfail:5d} max_abs={d.max():.4e}{tinfo} "
              f"{'PASS' if nfail == 0 else 'FAIL'}")

    try:
        inp = (root / "testdata" / "input_0.bin").read_bytes()
        ser.write(b"T" + bytes([args.reps]))
        ser.write(inp)
        tline = read_until(ser, b"\n", timeout=120.0)
        print(f"[device] {tline.strip().decode()}")
    except Exception as e:
        print(f"[device] timing sweep skipped: {e}")

    ser.close()
    print("ALL PASS" if total_fail == 0 else f"{total_fail} FAILURES")
    return 1 if total_fail else 0


if __name__ == "__main__":
    sys.exit(main())
