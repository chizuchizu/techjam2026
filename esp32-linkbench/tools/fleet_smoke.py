#!/usr/bin/env python3
"""fleet_smoke.py - drive the ESP32 FLEET SUM16 demo from the host.

Opens the USB serial of a connected board (as many as you pass), resets it so
the app boots and prints, waits for READY, lets the fleet rediscover peers,
then sends 'B' and asserts the coordinator's merged sum matches its locally
computed expected value (LINKFW-S|job|done|...|match=1).

The firmware also auto-runs the demo every 6 s (LF_DEMO_AUTO=1), so if the 'B'
collides with an auto-job the tool simply waits for the next done line.

Usage:
  python3 tools/fleet_smoke.py --port /dev/cu.usbmodem101
  python3 tools/fleet_smoke.py --port /dev/cu.usbmodem101 --port /dev/cu.usbmodem1101
  python3 tools/fleet_smoke.py --port /dev/cu.usbmodem101 -n 5     # run demo 5x
"""
import argparse
import re
import sys
import time

try:
    import serial
except ImportError:
    sys.exit("pip install pyserial")

BOOT_TIMEOUT = 45.0      # sec to wait for "READY" after resetting the board
DEMO_TIMEOUT = 14.0      # sec to wait for a job outcome

OUTCOME = re.compile(r"LINKFW-S\|job\|done\|.*match=(\d)")
NOT_COORD = re.compile(r"LINKFW-S\|job\|(?:not_coord|no_peers)")
BUSY = re.compile(r"LINKFW-S\|job\|busy")
BOOT_LANDMARK = re.compile(r"LINKFW-S\|(?:READY|info\|ip|connected\|ip|ap\|ssid|role\|coord)")


def open_port(port, baud):
    """Open the native-USB CDC and pulse EN so the app boots and prints.

    Opening a native-USB ESP32-C3 asserts DTR, dropping the chip into ROM
    download mode. We hold IO9 high while pulsing EN low-high to boot the app.
    The USB device re-enumerates on reset; callers must tolerate the read fd
    dying ("device not configured") and reconnect.
    """
    s = serial.Serial(port, baud, timeout=0.2)
    try:
        s.setDTR(False); s.setRTS(True); time.sleep(0.15)   # EN low (reset)
        s.setRTS(False); time.sleep(0.30)                   # EN high: boot app
    except Exception:
        pass
    return s


def read_lines(s, seconds):
    """Read all complete lines arriving on `s` over `seconds`.

    Returns a list of text lines. Tries to survive the USB re-enumeration:
    if the read fd throws, it re-opens the port (no reset pulse) and resumes.
    """
    out = []
    end = time.time() + seconds
    buf = b""
    while time.time() < end:
        try:
            c = s.read(4096)
        except Exception:
            c = b""
            try:
                s.close()
            except Exception:
                pass
            time.sleep(0.6)
            try:
                s = serial.Serial(s.port, s.baudrate, timeout=0.2)
                # reopening re-asserts DTR (ROM download mode): re-pulse EN so
                # the app boots again, exactly as open_port does.
                s.setDTR(False); s.setRTS(True); time.sleep(0.15)
                s.setRTS(False); time.sleep(0.30)
            except Exception:
                continue
        if c:
            buf += c
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                line = raw.decode(errors="replace").strip("\r")
                if line:
                    out.append(line)
                if len(out) >= 4000:
                    return s, out
        else:
            time.sleep(0.05)
    if buf.strip():
        line = buf.decode(errors="replace").strip("\r")
        if line:
            out.append(line)
    return s, out


def run_once(port, baud):
    s = open_port(port, baud)
    try:
        # Phase 1: boot the app and wait until the fleet layer is up.
        booted, lines = False, []
        seen = {}

        def printable(s):
            # keep text STATUS lines, drop binary payload garbage
            return all(32 <= ord(c) < 127 or c in "\t" for c in s)

        def ingest(lines):
            for line in lines:
                if line.startswith("LINKFW-S") and line not in seen:
                    seen[line] = True
                if printable(line):
                    print("  | " + line)

        while True:
            s, lines = read_lines(s, BOOT_TIMEOUT)
            ingest(lines)
            if any(BOOT_LANDMARK.search(l) for l in lines):
                booted = True
                break
            if any("LINKFW-S" in l for l in lines):
                # boards keep printing beacons/jobs; new lines may stop, so if
                # no new output arrived for a full window, that is "quiet" and
                # we re-open once to catch a fresh burst rather than hang.
                pass
            break  # one long read window is enough to see a full boot

        if not booted:
            print(f"[{port}] board never reached a boot landmark")
            return None

        time.sleep(1.5)               # let the fleet rediscover peers
        try:
            s.write(b"B")              # ask the coordinator for a demo run
        except Exception:
            pass
        end = time.time() + DEMO_TIMEOUT
        while time.time() < end:
            remain = end - time.time()
            if remain <= 0:
                break
            s, lines = read_lines(s, min(remain, 6.0))
            for line in lines:
                if line.startswith("LINKFW-S") and printable(line):
                    print("  | " + line)
                m = OUTCOME.search(line)
                if m:
                    print(f"[{port}] OUTCOME: {line}")
                    return int(m.group(1)) == 1
                if NOT_COORD.search(line):
                    print(f"[{port}] OUTCOME: {line.strip()} (not coordinator here)")
                    return "worker"
                # BUSY: an auto-run demo is in flight; keep waiting for done.
        print(f"[{port}] no demo outcome within {DEMO_TIMEOUT:.0f} s")
        return None
    finally:
        try:
            s.close()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", action="append", required=True,
                    help="USB serial port of an ESP32 (repeat for more boards)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("-n", "--repeat", type=int, default=1)
    args = ap.parse_args()

    results = []
    for it in range(args.repeat):
        for p in args.port:
            print(f"--- demo run {it + 1} on {p} ---")
            r = run_once(p, args.baud)
            results.append((p, r))
            time.sleep(0.3)

    ok = all(r in (True, "worker") for _, r in results) and any(
        r is True for _, r in results)
    print("RESULT:", "PASS" if ok else "FAIL", results)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
