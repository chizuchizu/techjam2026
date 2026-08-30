
#!/usr/bin/env python3
"""profile_04.py - run n forwards on the board then dump per-phase profile."""
import sys, time, pathlib
import numpy as np
import serial

PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/cu.usbmodem1101"
N = 128 * 128
root = pathlib.Path(__file__).resolve().parent.parent  # project root
inp = (root / "testdata" / "input_0.bin").read_bytes()
CHUNK, GAP = 1024, 0.02

def read_until(ser, token, timeout=30.0):
    buf = b""
    t0 = time.time()
    while token not in buf:
        if time.time() - t0 > timeout:
            raise TimeoutError(f"no {token!r}; got {buf[-80:]!r}")
        chunk = ser.read(ser.in_waiting or 1)
        if chunk:
            buf += chunk
    return buf

def send_input(ser, data):
    off = 0
    while off < len(data):
        ser.write(data[off:off+CHUNK]); off += CHUNK
        time.sleep(GAP)

ser = serial.Serial(PORT, 115200, timeout=1.0)
time.sleep(0.5)
banner = b""
t0=time.time()
while time.time()-t0<1.0:
    c=ser.read(ser.in_waiting or 1)
    if c: banner+=c
# wait idle
ser.write(b"M")
line = read_until(ser, b"\n", timeout=5.0)
print("idle:", line.strip().decode(errors='replace'))
n_fwd = int(sys.argv[2]) if len(sys.argv) > 2 else 3
for i in range(n_fwd):
    ser.reset_input_buffer()
    ser.write(b"R"); time.sleep(0.2)
    send_input(ser, inp)
    out = b""
    t0 = time.time()
    while len(out) < N*4:
        if time.time()-t0 > 120:
            raise TimeoutError("output timeout")
        c = ser.read(N*4-len(out))
        if c: out += c
    tail = read_until(ser, b"\n", timeout=20.0)
    print(f"fwd {i}: {tail.strip().decode(errors='replace')}")
# dump profile
ser.reset_input_buffer()
ser.write(b"P")
buf = read_until(ser, b"total_wall", timeout=10.0)
for ln in buf.decode(errors='replace').splitlines():
    print("PROF:", ln)
ser.reset_input_buffer()
ser.write(b"X")
read_until(ser, b"\n", timeout=5.0)
ser.close()
