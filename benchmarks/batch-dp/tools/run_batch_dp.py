#!/usr/bin/env python3
"""
run_batch_dp.py - data-parallel batch inference across N ESP32-C3 boards.

Cases 1, 3, 4 and 5 are the case-2 geometry at batch B = 64, 4, 16 and 128.
Every one of the B inputs is an independent forward over the same weights, so
the boards need to exchange nothing at all: input i runs on board i % N. That
is the whole design - no firmware changes, no inter-board link. Each board runs
the maintained single-board firmware from
../../case-02/optimisation/esp32-baseline and holds the full weight blobs,
which are identical for every one of these cases.

Reported per run:
  compute wall   max over boards of the summed device forward times - the
                 batch time the cluster achieves, excluding host transfer,
                 which is the same convention the single-board number uses
  serial wall    measured end to end including the 64 KB in / 64 KB out per
                 input over USB, for context
  speedup        (sum of every forward's device time) / compute wall, i.e.
                 against one board doing all B inputs

Usage:
  python3 tools/run_batch_dp.py --batch 4 16 64 128 [--boards /dev/ttyACM0 ...]
"""
import argparse
import glob
import json
import pathlib
import re
import statistics
import subprocess
import sys
import threading
import time

import numpy as np
import serial

N_ELEMS = 128 * 128
FRAME = N_ELEMS * 4
ATOL, RTOL = 0.002, 0.02
CHUNK, GAP = 1024, 0.02          # paced host->device delivery (CDC RX limit)
USBIPD = "/mnt/c/Program Files/usbipd-win/usbipd.exe"
ROOT = pathlib.Path(__file__).resolve().parents[1]


def usbipd_reattach():
    """Re-attach the boards after the WSL usbip bridge drops one."""
    import os
    if not os.path.exists(USBIPD):
        return
    try:
        out = subprocess.run([USBIPD, "list"], capture_output=True,
                             text=True, timeout=20).stdout
    except Exception:
        return
    ids, conn = [], False
    for line in out.splitlines():
        if line.startswith("Connected:"):
            conn = True
            continue
        if line.startswith("Persisted:"):
            conn = False
        if conn and "JTAG/serial debug unit" in line:
            ids.append(line.split()[0])
    for b in ids:
        subprocess.run([USBIPD, "attach", "--wsl", "--busid", b],
                       capture_output=True)
    time.sleep(2.0)


class Board:
    """One board, with its port drained by a background thread.

    Leaving the port idle through a multi-second forward wedges the handle on
    this bench's usbip bridge, so it is read continuously."""

    def __init__(self, port, baud=115200, name=""):
        self.port, self.baud, self.name = port, baud, name or port
        self.buf = bytearray()
        self.lock = threading.Lock()
        self.stop = False
        self._open()

    def _open(self):
        self.ser = serial.Serial(self.port, self.baud, timeout=0.05)
        self.rx = threading.Thread(target=self._pump, daemon=True)
        self.rx.start()

    def _pump(self):
        while not self.stop:
            try:
                c = self.ser.read(4096)
            except Exception:
                return
            if c:
                with self.lock:
                    self.buf.extend(c)

    def close(self):
        self.stop = True
        time.sleep(0.15)
        try:
            self.ser.close()
        except Exception:
            pass

    def clear(self):
        with self.lock:
            self.buf.clear()

    def recycle(self):
        self.stop = True
        try:
            self.ser.close()
        except Exception:
            pass
        time.sleep(0.4)
        usbipd_reattach()
        self.stop = False
        for _ in range(20):
            try:
                self._open()
                return
            except Exception:
                time.sleep(0.5)
        raise RuntimeError(f"{self.name}: port did not come back")

    def read_exact(self, n, timeout):
        t0 = time.time()
        while True:
            with self.lock:
                if len(self.buf) >= n:
                    d = bytes(self.buf[:n])
                    del self.buf[:n]
                    return d
                have = len(self.buf)
            if time.time() - t0 > timeout:
                raise TimeoutError(f"{self.name}: {have}/{n} bytes")
            time.sleep(0.02)

    def read_line(self, timeout):
        t0 = time.time()
        while True:
            with self.lock:
                i = self.buf.find(b"\n")
                if i >= 0:
                    line = bytes(self.buf[:i + 1])
                    del self.buf[:i + 1]
                    return line
            if time.time() - t0 > timeout:
                raise TimeoutError(f"{self.name}: no line in {timeout}s")
            time.sleep(0.02)

    def send_paced(self, data):
        for off in range(0, len(data), CHUNK):
            self.ser.write(data[off:off + CHUNK])
            time.sleep(GAP)

    def drain_quiet(self, quiet=1.5, cap=90.0):
        """Discard bytes until the board has been silent for `quiet` seconds."""
        t0 = last = time.time()
        while time.time() - t0 < cap:
            with self.lock:
                if self.buf:
                    self.buf.clear()
                    last = time.time()
            if time.time() - last >= quiet:
                return True
            time.sleep(0.05)
        return False

    def resync(self):
        """Get the board back to a known-idle state before reusing it.

        A forward that fails part-way through its 64 KB output leaves the rest
        of that frame queued on the device. Re-issuing 'R' without clearing it
        prefixes the next output with the tail of the old one, and the frame
        that comes back is misaligned - which reads as a catastrophic accuracy
        failure rather than as the transport fault it is. So: drain to silence,
        then require a clean reply before continuing."""
        self.drain_quiet()
        for attempt in range(4):
            try:
                self.clear()
                self.ser.write(b"M")
                line = self.read_line(8).decode(errors="replace")
                if "TM " in line:
                    self.drain_quiet(quiet=0.6, cap=10.0)
                    return True
            except TimeoutError:
                pass
            if attempt == 0:
                kick(self)          # blocked inside read_input()
            else:
                self.recycle()
            self.drain_quiet()
        return False

    def forward(self, frame, tries=5):
        """Run one input, return (output bytes, device microseconds)."""
        for attempt in range(tries):
            try:
                self.clear()
                self.ser.write(b"R")
                time.sleep(0.2)
                self.send_paced(frame)
                out = self.read_exact(FRAME, 120)
                tail = self.read_line(30).decode(errors="replace")
                m = re.search(r"us=(\d+)", tail)
                return out, (int(m.group(1)) if m else 0)
            except Exception as exc:
                print(f"[{self.name}] forward retry {attempt + 1}: {exc}")
                if not self.resync():
                    self.recycle()
        raise RuntimeError(f"{self.name}: forward failed after {tries} tries")


def kick(b):
    """Complete a half-delivered input frame.

    An interrupted run can leave the firmware blocked inside read_input()
    waiting for the rest of a 64 KB frame; it answers nothing until it gets
    them. Feeding it zeros lets that forward finish and returns the command
    loop to idle."""
    b.clear()
    b.ser.write(b"R")
    time.sleep(0.2)
    b.send_paced(b"\0" * FRAME)
    try:
        b.read_exact(FRAME, 180)
        b.read_line(30)
    except TimeoutError:
        pass
    # the leftover bytes of the interrupted frame come back as command echoes
    t0 = last = time.time()
    while time.time() - t0 < 30:
        with b.lock:
            if b.buf:
                b.buf.clear()
                last = time.time()
        if time.time() - last > 2.0:
            break
        time.sleep(0.1)


def wait_idle(b):
    for attempt in range(4):
        try:
            b.clear()
            b.ser.write(b"M")
            return b.read_line(8).strip().decode(errors="replace")
        except TimeoutError:
            if attempt == 0:
                print(f"[{b.name}] no reply; completing a pending input frame")
                kick(b)
            else:
                b.recycle()
    raise RuntimeError(f"{b.name}: unresponsive")


def run_batch(boards, batch, root, atol=ATOL, rtol=RTOL):
    d = root / "testdata" / f"B{batch}"
    inputs = [(d / f"input_{i}.bin").read_bytes() for i in range(batch)]
    refs = [np.fromfile(d / f"ref_{i}.bin", dtype="<f4").astype(np.float64)
            for i in range(batch)]

    n = len(boards)
    assign = [[i for i in range(batch) if i % n == k] for k in range(n)]
    results = [dict(us=[], fails=0, max_abs=0.0, wall=0.0, missing=[])
               for _ in range(n)]

    def worker(k):
        b, r = boards[k], results[k]
        t0 = time.time()
        for i in assign[k]:
            try:
                out, us = b.forward(inputs[i])
            except Exception as exc:
                # Record it and carry on. A dropped input must never quietly
                # shrink the batch: the summary below refuses to report a
                # speedup unless every assigned input actually ran.
                print(f"[{b.name}] input {i} lost: {exc}")
                r["missing"].append(i)
                continue
            got = np.frombuffer(out, dtype="<f4").astype(np.float64)
            dd = np.abs(got - refs[i])
            ok = (dd <= atol) | (dd <= rtol * np.abs(refs[i]))
            r["us"].append(us)
            r["fails"] += int((~ok).sum())
            r["max_abs"] = max(r["max_abs"], float(dd.max()))
        r["wall"] = time.time() - t0

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(n)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    serial_wall = time.time() - t0

    per_board = [sum(r["us"]) / 1e6 for r in results]
    all_us = [u for r in results for u in r["us"]]
    missing = sorted(i for r in results for i in r["missing"])
    # Scale each board's measured compute to the inputs it was assigned, so a
    # transport loss cannot masquerade as a shorter batch. Only meaningful when
    # nothing is missing; when something is, the row is reported INCOMPLETE.
    total_compute = sum(all_us) / 1e6          # one board doing every input
    compute_wall = max(per_board)              # the cluster's batch time
    fails = sum(r["fails"] for r in results)
    return dict(
        batch=batch, boards=n, missing=missing, complete=not missing,
        checked=len(all_us),
        per_board_inputs=[len(a) for a in assign],
        per_board_compute_s=[round(p, 3) for p in per_board],
        compute_wall_s=round(compute_wall, 3),
        one_board_compute_s=round(total_compute, 3),
        speedup=(round(total_compute / compute_wall, 3)
                 if compute_wall and not missing else None),
        serial_wall_s=round(serial_wall, 1),
        per_forward_s=round(statistics.median(all_us) / 1e6, 4) if all_us else 0,
        fails=fails,
        max_abs=max(r["max_abs"] for r in results),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", nargs="+", type=int, default=[4, 16, 64, 128])
    ap.add_argument("--boards", nargs="*", default=None)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    ports = args.boards or sorted(glob.glob("/dev/ttyACM*"))
    if not ports:
        print("no boards found; run ../case-02/multiboard/esp32-cluster-full/"
              "tools/attach_boards.sh")
        return 2
    print(f"[boards] {', '.join(ports)}")

    boards = [Board(p, args.baud, f"b{k}") for k, p in enumerate(ports)]
    time.sleep(0.8)
    for b in boards:
        print(f"[{b.name}] {wait_idle(b)}")

    root = pathlib.Path(args.root)
    rows = []
    for B in args.batch:
        if B < len(boards):
            print(f"B={B}: fewer inputs than boards, skipping")
            continue
        print(f"\n=== B={B} on {len(boards)} boards ===")
        r = run_batch(boards, B, root)
        rows.append(r)
        head = (f"B={B:4d} boards={r['boards']} inputs/board={r['per_board_inputs']}\n"
                f"      per-forward {r['per_forward_s']:.3f}s | ")
        if r["complete"]:
            head += (f"one board {r['one_board_compute_s']:.1f}s -> "
                     f"cluster {r['compute_wall_s']:.1f}s = {r['speedup']:.2f}x")
        else:
            head += (f"INCOMPLETE: {len(r['missing'])} of {B} inputs lost to the "
                     f"USB bridge {r['missing']}; no speedup reported")
        print(head + f"\n      end-to-end incl. USB {r['serial_wall_s']:.1f}s | "
                     f"fails={r['fails']} max_abs={r['max_abs']:.3e} "
                     f"{'PASS' if r['fails'] == 0 else 'FAIL'} "
                     f"({r['checked']}/{B} forwards checked)")

    for b in boards:
        b.close()

    if rows:
        print("\n| B | boards | per-forward | 1 board | cluster | speedup | gate |")
        print("|---:|---:|---:|---:|---:|---:|---|")
        for r in rows:
            sp = f"{r['speedup']:.2f}x" if r["complete"] else "incomplete"
            gate = ("PASS" if r["fails"] == 0 else f"{r['fails']} fails")
            if not r["complete"]:
                gate += f" ({len(r['missing'])} inputs lost)"
            print(f"| {r['batch']} | {r['boards']} | {r['per_forward_s']:.3f} s | "
                  f"{r['one_board_compute_s']:.1f} s | {r['compute_wall_s']:.1f} s | "
                  f"{sp} | {gate} |")
    if args.json and rows:
        pathlib.Path(args.json).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.json}")
    return 1 if any(r["fails"] or not r["complete"] for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
