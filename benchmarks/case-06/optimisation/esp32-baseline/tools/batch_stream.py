#!/usr/bin/env python3
"""
batch_stream.py - stream a B-frame batch through the case-06 ESP32 firmware.

The board runs ONE forward per input frame (it never holds the whole B=10000
batch in SRAM). This driver is the host side of that stream: it walks the
per-frame input bins (or synthesizes deterministic frames), sends each frame
over the same serial protocol as tools/device_test.py, and (optionally) checks
every frame against its stored torch reference (atol=0.002, rtol=0.02).

Serial protocol (see src/main.cpp):
    'M'         -> "TM <mode> <S> <D>"
    'R' <frame bytes float32 LE> -> one forward, streams the frame back + "END ..."
    'T' <n>     -> warmup 1 + n timed forwards

The driver deliberately does NOT invent a per-frame compute time. All timing
reported below comes from ONE of:
  * a real device run  (the firmware's `esp_timer_get_time()` delta printed
    after each 'R' as `us=<n>`), or
  * a `--fwd-us <n>` assumption YOU supply (used only to show how the batch
    estimate is computed).

For the default (no --port) run we print the transport/estimation structure
and the byte-volume math for B frames but mark the finish time as UNMEASURED.

A full B=10000 batch is a long-running execution: at the ~1.996 s/forward the
case-02 optimisation snapshot measured this would be ~5.5 h of pure compute
plus transfer, so it is intentionally not run here; the throughput structure
is reported instead of a fabricated number.

Usage:
  python3 tools/batch_stream.py --help
  python3 tools/batch_stream.py --dry-run --frames 10000
  python3 tools/batch_stream.py /dev/cu.usbmodem2101 --frames 10 [--check]
  python3 tools/batch_stream.py --port /dev/cu.usbmodem2101 --synthesize --frames 3
"""
import argparse
import pathlib
import sys
import time

N_DEFAULT = 128 * 128          # S*D for case 6 (=16384 floats, 65536 bytes)
ATOL, RTOL = 0.002, 0.02
CHUNK, GAP = 1024, 0.02        # paced host->device delivery (see device_test.py)
BATCH_B = 10000                # case-06 batch size (informational only)


def _lazy_serial():
    try:
        import serial  # pyserial
    except ImportError as e:  # pragma: no cover
        raise SystemExit("live device mode needs pyserial: pip install pyserial") from e
    return serial


def _lazy_np():
    try:
        import numpy as np
    except ImportError as e:
        raise SystemExit("synthesize/check modes need numpy in SYSTEM python3: "
                         "python3 -m pip install numpy") from e
    return np


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
    """Deliver one S*D-byte frame without tripping the C3 native-USB CDC RX
    drop bug: 1 KB chunks, 20 ms apart (same pace as tools/device_test.py)."""
    off = 0
    while off < len(data):
        ser.write(data[off:off + chunk])
        off += chunk
        if gap:
            time.sleep(gap)


def wait_idle(ser, frame_bytes: int, timeout: float = 12.0) -> bytes:
    """Return the first 'TM ' console line, recovering a stuck 'R' if needed."""
    ser.reset_input_buffer()
    ser.write(b"M")
    try:
        return read_until(ser, b"\n", timeout=max(3.0, timeout - 3.0))
    except TimeoutError:
        print("[stream] no reply to M; kicking a pending input frame ...", file=sys.stderr)
        ser.write(b"R")
        time.sleep(0.2)
        send_input(ser, b"\0" * frame_bytes)
        _ = ser.read(frame_bytes)                     # drain stale output frame
        read_until(ser, b"\n", timeout=30.0)
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


def frame_from_bin(root: pathlib.Path, seed: int, n: int) -> tuple:
    """Return (input_bytes, ref_np_or_None) for testdata seed file."""
    src = (root / "testdata" / f"input_{seed}.bin").read_bytes()
    if len(src) != n * 4:
        raise SystemExit(f"testdata/input_{seed}.bin is {len(src)} B, expected {n*4} B")
    ref_path = root / "testdata" / f"ref_{seed}.bin"
    ref = None
    if ref_path.exists():
        np = _lazy_np()
        ref = np.fromfile(ref_path, dtype="<f4").astype(np.float64)
        if ref.size != n:
            ref = None
    return src, ref


def synth_frame(seed: int, n: int) -> bytes:
    """Deterministic synthetic input frame (same scale as the benchmark's
    generator; there is NO stored reference for these frames)."""
    np = _lazy_np()
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n).astype(np.float32)
    return x.tobytes()


def summarize_gate(fails, max_abs):
    return (f"fails={fails:5d} max_abs={max_abs:.4e} "
            f"{'PASS' if fails == 0 else 'FAIL'}")


def run_live(args, n: int) -> int:
    """Stream `--frames` input frames through the firmware over serial."""
    serial_mod = _lazy_serial()
    ser = serial_mod.Serial(args.port, args.baud, timeout=1.0)
    time.sleep(0.5)
    try:
        mode_line = wait_idle(ser, n * 4)
        print(f"[stream] firmware: {mode_line.strip().decode(errors='replace')}")
    except Exception as e:
        ser.close()
        raise RuntimeError(f"cannot reach firmware on {args.port}: {e}") from e

    frame_bytes = n * 4
    fails_total = 0
    sent_bytes = 0
    recv_bytes = 0
    fwd_us = []
    t0 = time.monotonic()

    for i in range(args.frames):
        seed = args.seed + i
        if args.synthesize:
            src = synth_frame(seed, n)
            ref = None
        else:
            # cycle over the available bins when frames > stored seeds
            src, ref = frame_from_bin(pathlib.Path(args.root), seed, n)

        ser.reset_input_buffer()
        ser.write(b"R")
        time.sleep(0.2)                         # let firmware enter read_input()
        send_input(ser, src, args.chunk, args.gap)
        sent_bytes += len(src)

        out = b""
        tfr = time.time()
        while len(out) < frame_bytes:
            if time.time() - tfr > 120:
                print(f"frame {i}: TIMEOUT waiting for output")
                return 2
            chunk = ser.read(frame_bytes - len(out))
            if chunk:
                out += chunk
        recv_bytes += frame_bytes
        tail = read_until(ser, b"\n", timeout=20.0)
        recv_bytes += len(tail)

        us = None
        if b"us=" in tail:
            try:
                us = int(tail.split(b"us=", 1)[1].split(b"\n", 1)[0])
                fwd_us.append(us)
            except Exception:
                pass

        stat = f"frame {i:>6} seed={seed}"
        if ref is not None and args.check:
            np = _lazy_np()
            got = np.frombuffer(out, dtype="<f4").astype(np.float64)
            d = np.abs(got - ref)
            ok = (d <= ATOL) | (d <= RTOL * np.abs(ref))
            fails = int((~ok).sum())
            fails_total += fails
            stat += " " + summarize_gate(fails, d.max())
        elif args.synthesize:
            stat += " synthesized (no stored ref; not checked)"
        else:
            stat += " (not checked)"
        if us is not None:
            stat += f" fwd={us/1e6:.3f}s"
        print(stat)

    wall = time.monotonic() - t0
    ser.close()
    print(f"[stream] {args.frames} frames wall-clock {wall:.3f}s "
          f"(host pacing + firmware compute, measured for THIS run)")
    if fwd_us:
        mean_us = sum(fwd_us) / len(fwd_us)
        print(f"[stream] firmware-reported per-forward: "
              f"min {min(fwd_us)/1e6:.3f}s mean {mean_us/1e6:.3f}s "
              f"max {max(fwd_us)/1e6:.3f}s ({len(fwd_us)} samples)")
    print(f"[stream] host->device {sent_bytes} B, device->host {recv_bytes} B")
    return 0 if fails_total == 0 else 1


def run_dry(args, n: int) -> int:
    frame_bytes = n * 4
    fwd_us = args.fwd_us
    tx = args.frames * frame_bytes
    rx = args.frames * frame_bytes
    print("batch stream plan (no device; nothing measured)")
    print(f"  geometry: B={BATCH_B} (informational), S*D = {n} floats/frame")
    print(f"  frames to stream: {args.frames} "
          f"({'synthesized' if args.synthesize else 'testdata bins'})")
    print("transport structure (per frame):")
    print(f"  input  = {frame_bytes} B ({n} floats, float32 LE)")
    print(f"  output = {frame_bytes} B")
    print("transport rates for the estimate (reference values from the case-02 "
          "port, NOT re-measured in this case-06 run; override with --tx-bps/"
          "--rx-bps):")
    print(f"  host->device = {args.tx_bps/1e3:.1f} KB/s "
          f"(paced {args.chunk} B / {args.gap}s gap, device_test.py method)")
    print(f"  device->host = {args.rx_bps/1e3:.1f} KB/s "
          f"(native USB-CDC reference)")
    print(f"  (a true hardware-UART bridge at {args.baud} 8N1 would cap at "
          f"{args.baud/10/1e3:.1f} KB/s raw-wire; native USB-CDC does not use "
          f"this baud limit)")
    print(f"  {args.frames} frames host->device volume = {tx} B = {tx/1e6:.3f} MB")
    print(f"  {args.frames} frames device->host volume = {rx} B = {rx/1e6:.3f} MB "
          f"(plus END lines)")
    if fwd_us is not None:
        fwd_s = fwd_us / 1e6
        in_time = frame_bytes / args.tx_bps
        out_time = frame_bytes / args.rx_bps
        per_frame = in_time + fwd_s + out_time
        print("batch estimate (--fwd-us is YOUR assumption, not a measurement):")
        print(f"  per-forward compute assumption = {fwd_s:.3f}s")
        print(f"  per-frame transfer = host->device {in_time:.3f}s "
              f"+ device->host {out_time:.3f}s = {in_time + out_time:.3f}s")
        print(f"  per-frame schedule             = {per_frame:.3f}s")
        print(f"  {args.frames} frames ESTIMATED    = {args.frames*per_frame:.3f}s "
              f"= {args.frames*per_frame/3600:.3f}h")
        print(f"  effective frames/s (estimated) = {1.0/per_frame:.3f}")
    else:
        print("batch estimate: needs a measured per-forward time (--fwd-us) to "
              "complete; NOT estimated because no physical timing was measured.")
        print("example: --fwd-us 1996000  (1.996s case-02 snapshot) to see HOW "
              "the estimate is built.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Stream per-frame inputs through the case-06 ESP32 firmware "
                    "and print the batch throughput/estimation structure without "
                    "fabricating physical timing.")
    ap.add_argument("port", nargs="?", default=None,
                    help="serial port (/dev/cu.usbmodem2101). Omit for a dry-run plan.")
    ap.add_argument("--port", dest="port_opt", default=None,
                    help="same as the positional port (either may be used)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--root", default="..", help="project root (testdata lives here)")
    ap.add_argument("--frames", type=int, default=10000,
                    help="number of frames to stream (B=10000 for case 6)")
    ap.add_argument("--seed", type=int, default=0,
                    help="first input seed to use; seed+i for the i-th frame")
    ap.add_argument("--synthesize", action="store_true",
                    help="synthesize deterministic frames instead of reading bins")
    ap.add_argument("--check", action="store_true",
                    help="validate each frame against its stored torch ref")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and exit even if a port was given")
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--gap", type=float, default=GAP)
    ap.add_argument("--fwd-us", type=float, default=None,
                    help="USER-SUPPLIED per-forward time in us, used only to build "
                         "the example batch estimate (not a measurement)")
    ap.add_argument("--tx-bps", type=float, default=51200.0,
                    help="host->device transport rate used in the estimate "
                         "(default 51200 = device_test.py 1KB/20ms pacing)")
    ap.add_argument("--rx-bps", type=float, default=286000.0,
                    help="device->host transport rate used in the estimate "
                         "(default 286000 = native USB-CDC reference from the "
                         "case-02 port; NOT re-measured here)")
    args = ap.parse_args()

    port = args.port or args.port_opt

    probe = pathlib.Path(args.root) / "testdata" / "input_0.bin"
    n = N_DEFAULT
    if probe.exists():
        n = len(probe.read_bytes()) // 4
    if n <= 0:
        raise SystemExit(f"cannot determine frame size (S*D) from {probe}")

    if args.dry_run or port is None:
        return run_dry(args, n)
    return run_live(args, n)


if __name__ == "__main__":
    sys.exit(main())
