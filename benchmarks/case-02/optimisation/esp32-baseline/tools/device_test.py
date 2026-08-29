#!/usr/bin/env python3
"""
device_test.py - drive the ESP32-C3 firmware over serial and verify outputs.

Protocol (src/main.cpp): 'M' -> "TM <mode> <S> <D>"; 'S' -> "TM OK mode=<m>";
'R' <65536 B float32 LE input> -> one forward, streams 65536 B output + "END ...";
'T <n>' -> warmup 1 + n timed forwards, prints "TM <mode> <us> ...".

Firmware handling notes this driver compensates for:
  * native-USB CDC RX on the C3 drops host->device bursts above ~1 KB unless
    the host paces; send_input() writes 1 KB chunks with a 20 ms gap.
  * if a previous interrupted run left the firmware stuck in read_input(),
    a startup "kick" sends one full zero frame so it returns to idle.

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
CHUNK, GAP = 1024, 0.02  # paced host->device delivery (see module docstring)


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


def send_input(ser, data: bytes, chunk: int = CHUNK, gap: float = GAP) -> None:
    """Deliver a full S*D-byte input frame without tripping the CDC RX
    drop bug: 1 KB chunks, 20 ms apart (safe margin below the >4 KB/10 ms
    threshold; 3/3 runs lossless at 1 KB/20 ms)."""
    off = 0
    while off < len(data):
        ser.write(data[off:off + chunk])
        off += chunk
        if gap:
            time.sleep(gap)


def drain_quiet(ser, quiet_s: float = 2.0, max_wait: float = 120.0) -> None:
    """Drain RX until no bytes arrive for quiet_s. After a kick the C3 emits
    one 'TM unknown cmd' line per leftover byte of the interrupted frame at a
    slow USB-CDC rate, so this needs a generous window before returning."""
    t0 = time.time()
    last = time.time()
    while time.time() - t0 < max_wait:
        if ser.in_waiting:
            ser.read(ser.in_waiting)
            last = time.time()
        elif time.time() - last >= quiet_s:
            break
        else:
            time.sleep(0.05)
    ser.reset_input_buffer()


def _kick(ser, timeout: float = 150.0) -> None:
    """Complete a pending 'R' frame when the firmware is stuck in
    read_input(): send zeros, then drain the 64 KB output frame and the
    END line so the command loop returns to idle."""
    ser.write(b"R")
    time.sleep(0.2)
    send_input(ser, b"\0" * (N * 4))
    out = b""
    t0 = time.time()
    while len(out) < N * 4:
        if time.time() - t0 > timeout:
            raise RuntimeError("kick: no output frame within timeout")
        chunk = ser.read(N * 4 - len(out))
        if chunk:
            out += chunk
    tail = read_until(ser, b"\n", timeout=30.0)
    if b"END" not in tail:
        raise RuntimeError(f"kick did not recover firmware: {tail[-80:]!r}")
    print(f"[device] recovered (stale frame drained): {tail.strip().decode()}")
    drain_quiet(ser)  # clear trailing bytes left from the interrupted frame


def wait_idle(ser, timeout: float = 12.0) -> bytes:
    """Return the first 'TM ' console line. If the firmware is stuck in a
    pending read_input() (interrupted previous run), feed it one full zero
    frame so it returns to the command loop, then retry."""
    ser.reset_input_buffer()
    ser.write(b"M")
    try:
        return read_until(ser, b"\n", timeout=max(3.0, timeout - 3.0))
    except TimeoutError:
        # likely stuck in read_input(): complete the pending frame (zeros),
        # let the forward finish, then ask again.
        print("[device] no reply to M; kicking pending input frame ...")
        _kick(ser)
        # after the kick the firmware is idle again; a few M retries cover
        # any last straggler bytes still draining from the interrupted frame
        for _ in range(20):
            ser.reset_input_buffer()
            ser.write(b"M")
            try:
                line = read_until(ser, b"\n", timeout=5.0)
                if line.strip():
                    return line
            except TimeoutError:
                time.sleep(0.5)
        raise RuntimeError("firmware unresponsive after recovery kick")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("port")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--root", default=".", help="project root (testdata lives here)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    global N  # frame floats = S*D per case; inferred from the first input
    root = pathlib.Path(args.root)
    probe = root / "testdata" / "input_0.bin"
    if probe.exists():
        N = len(probe.read_bytes()) // 4
    print(f"[device] frame = {N} floats ({N * 4} bytes) -> TM_S*TM_D from input file")
    if N <= 0:
        raise SystemExit(f"cannot determine frame size from {probe}")
    ser = serial.Serial(args.port, args.baud, timeout=1.0)
    time.sleep(0.5)
    # opportunistically capture the boot banner if the port open put the
    # firmware in a freshly-booted state (usually missed on CDC re-enum).
    banner = b""
    t0 = time.time()
    while time.time() - t0 < 1.0:
        chunk = ser.read(ser.in_waiting or 1)
        if chunk:
            banner += chunk

    mode_line = wait_idle(ser)
    print(f"[device] boot: {banner.decode(errors='replace').strip()!r}")
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
        time.sleep(0.2)  # let the firmware enter read_input() first
        send_input(ser, inp)
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
        ser.reset_input_buffer()
        ser.write(b"T" + bytes([args.reps]))
        time.sleep(0.2)
        send_input(ser, inp)
        # 1 + reps forwards: at ~42 s/forward on this snapshot, reps=3 -> ~170 s
        tline = read_until(ser, b"\n", timeout=300.0)
        print(f"[device] {tline.strip().decode()}")
    except Exception as e:
        print(f"[device] timing sweep skipped: {e}")

    ser.close()
    print("ALL PASS" if total_fail == 0 else f"{total_fail} FAILURES")
    return 1 if total_fail else 0


if __name__ == "__main__":
    sys.exit(main())
