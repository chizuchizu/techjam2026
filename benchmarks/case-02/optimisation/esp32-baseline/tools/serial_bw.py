#!/usr/bin/env python3
"""serial_bw.py - measure host<->board USB-CDC serial bandwidth for one ESP32-C3.

Uses ONE full 'R' round trip of the case-2 baseline protocol (64 KB float32 in,
one forward, 64 KB out + END). Only the transfer phases are timed; the ~42 s
forward is NOT part of the bandwidth numbers.

  host->board : push of the 64 KB input (configurable chunk/pause) wall-time.
  board->host : drain of the 64 KB output from first byte to END (no pacing).

Usage: python3 serial_bw.py PORT [--chunk 4096] [--pause 0.015]
"""
import argparse, sys, time
try:
    import serial
except ImportError:
    sys.exit("need pyserial: pip install pyserial")

N = 128 * 128 * 4  # 65536


def wait_idle(ser):
    ser.reset_input_buffer()
    ser.write(b"M"); time.sleep(0.8)
    b = ser.read(ser.in_waiting or 1)
    if not b:
        raise RuntimeError("board did not answer M")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("port")
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--pause", type=float, default=0.015,
                    help="seconds between chunks for host->board phase (0=xfail later)")
    ap.add_argument("--drain-timeout", type=float, default=95.0)
    a = ap.parse_args()

    ser = serial.Serial(a.port, 115200, timeout=2.0)
    time.sleep(0.3)
    # hard DTR/RTS reset: always starts from a clean slate
    ser.setDTR(False); ser.setRTS(True); time.sleep(0.15)
    ser.setRTS(False); ser.setDTR(True); time.sleep(0.15)
    ser.setDTR(False); time.sleep(1.5); ser.reset_input_buffer()
    wait_idle(ser)

    inp = bytes(N)
    ser.reset_input_buffer()
    ser.write(b"R"); time.sleep(0.2); ser.write(inp[:a.chunk]); time.sleep(0.2)
    t_push = time.time()
    for off in range(a.chunk, N, a.chunk):
        ser.write(inp[off:off + a.chunk]); time.sleep(a.pause)
    push_wall = time.time() - t_push
    print("host->board: %d B pushed in %.3f s -> %.1f KB/s (chunk=%d pause=%.0f ms)"
          % (N, push_wall, N / push_wall / 1024, a.chunk, a.pause * 1e3), flush=True)

    out, t_out, t0 = b"", None, time.time()
    while len(out) < N and (time.time() - t0) < a.drain_timeout:
        c = ser.read(8192)
        if c:
            if t_out is None:
                t_out = time.time()
            out += c
        else:
            time.sleep(0.05)
    if t_out is None:
        print("board->host: NO output within %.0f s (input was likely too fast)"
              % a.drain_timeout); return 2
    drain = time.time() - t_out
    # wait for END to confirm frame integrity
    end_seen = False
    tail = b""
    t1 = time.time()
    while (time.time() - t1) < 6.0:
        c = ser.read(1024)
        if c:
            tail += c
            if b"END" in tail:
                end_seen = True; break
        else:
            time.sleep(0.05)
    print("board->host: %d B drained in %.3f s -> %.1f KB/s (END frame: %s)"
          % (len(out), drain, len(out) / drain / 1024, end_seen), flush=True)
    ser.close()
    return 0 if (end_seen and len(out) == N) else 1


if __name__ == "__main__":
    sys.exit(main())
