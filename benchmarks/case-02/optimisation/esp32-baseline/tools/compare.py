#!/usr/bin/env python3
"""
compare.py - verify a raw output dump from the ESP32 (or host) against the
torch references using the benchmark's real gate (atol=0.002, rtol=0.02).

Usage:
  python3 tools/compare.py <got.bin> <seed> [--exact] [--atol X] [--rtol Y]
  cat /dev/cu.usbmodemX | python3 tools/compare.py - <seed>

A dump is the 65536-byte float32 LE output produced by the device's 'R'
command. Prints per-element gate result (+max abs/rel error).
"""
import argparse
import pathlib
import struct
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", help="path to raw float32 output dump, or '-' for stdin")
    ap.add_argument("seed", type=int)
    ap.add_argument("--root", default=".", help="project root (testdata lives here)")
    ap.add_argument("--atol", type=float, default=0.002)
    ap.add_argument("--rtol", type=float, default=0.02)
    args = ap.parse_args()

    if args.dump == "-":
        raw = sys.stdin.buffer.read()
    else:
        raw = pathlib.Path(args.dump).read_bytes()
    got = np.frombuffer(raw, dtype="<f4").reshape(128, 128)
    assert got.size == 128 * 128, f"expected 16384 floats, got {got.size}"

    ref = np.fromfile(pathlib.Path(args.root) / "testdata" / f"ref_{args.seed}.bin",
                      dtype="<f4").reshape(128, 128).astype(np.float64)
    ref[~np.isfinite(ref)] = 0.0

    a = got.astype(np.float64)
    b = ref
    d = np.abs(a - b)
    pass_mask = (d <= args.atol) | (d <= args.rtol * np.abs(b))
    nfail = int((~pass_mask).sum())
    print(f"seed {args.seed}: elements={d.size} failed={nfail} "
          f"max_abs={d.max():.4e} max_rel={(d/np.maximum(np.abs(b),1e-30)).max():.4e}")
    print("PASS" if nfail == 0 else "FAIL")
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
