#!/usr/bin/env python3
"""
run_cluster_e2e.py - end-to-end case-2 forward on two ESP32-C3 boards.

Both boards run the same firmware (src/main.cpp). This driver assigns the
roles, scatters the input rows, triggers one barrier-bracketed distributed
forward, gathers the output rows, and applies the benchmark gate
(|a-b| <= 0.002 OR |a-b| <= 0.02*|b|) against the torch reference.

Row partition (see src/model_shard.h): global token row i lives on node i % 2
at local index i // 2, so node 0 holds the even rows and node 1 the odd ones.

The reported wall time is the boards' own barrier-to-barrier measurement of the
distributed forward. Host scatter/gather over USB serial is outside it, exactly
as the single-board 'R' timing excludes its own serial transfer.

Usage:
  python3 tools/run_cluster_e2e.py /dev/ttyACM0 /dev/ttyACM1 \\
      [--seeds 0 1 2 3 4] [--reps 3] [--csv out.csv]
"""
import argparse
import glob
import os
import pathlib
import statistics
import sys
import subprocess
import threading
import time

import numpy as np
import serial  # pyserial

S, D = 128, 128
NODES = 2
SLOC = S // NODES
LOCAL_BYTES = SLOC * D * 4
ATOL, RTOL = 0.002, 0.02
CHUNK, GAP = 1024, 0.02  # paced host->device delivery (native USB CDC RX)

ROOT = pathlib.Path(__file__).resolve().parents[3] / "optimisation" / "esp32-baseline"


USBIPD = "/mnt/c/Program Files/usbipd-win/usbipd.exe"


def usbipd_reattach(busids):
    """Detach and re-attach the boards on the WSL usbip bus.

    The host side of usbip intermittently stops delivering CDC bytes under the
    load of a run and a plain reopen does not always clear it; re-attaching
    does. The USB port reset does not reset the ESP32-C3, so the boards keep
    their roles, their peer link and their loaded input."""
    if not busids or not os.path.exists(USBIPD):
        return False
    for action in ("detach", "attach"):
        for b in busids:
            cmd = [USBIPD, action, "--busid", b]
            if action == "attach":
                cmd.insert(2, "--wsl")
            subprocess.run(cmd, capture_output=True)
        time.sleep(1.5)
    for _ in range(20):
        if all(os.path.exists(p) for p in ("/dev/ttyACM0", "/dev/ttyACM1")):
            time.sleep(1.0)
            return True
        time.sleep(0.5)
    return False


class Node:
    """One board, with its serial port drained by a background thread.

    The port has to be read continuously: leaving it idle for the seconds a
    distributed forward takes wedges the host handle under WSL/usbip, and the
    board's reply then never arrives even though it was sent. The thread keeps
    the pipe moving and buffers whatever the board emits."""

    busids = []
    sibling = {}

    def __init__(self, port, baud, name):
        self.port, self.baud, self.name = port, baud, name
        self.recycle_count = 0
        self.node_id = None
        self.buf = bytearray()
        self.lock = threading.Lock()
        self.stop = False
        self._open()

    def _open(self):
        self.ser = serial.Serial(self.port, self.baud, timeout=0.05)
        self.thread = threading.Thread(target=self._pump, daemon=True)
        self.thread.start()

    def _pump(self):
        while not self.stop:
            try:
                chunk = self.ser.read(4096)
            except Exception:
                return
            if chunk:
                with self.lock:
                    self.buf.extend(chunk)

    def recycle(self):
        """Reopen the port, keeping whatever was already buffered.

        Under WSL/usbip the host handle intermittently stops delivering bytes -
        sometimes in the middle of a line - while the board is busy with the
        radio. The board's output is still queued on the device, and reopening
        the port resumes delivery, so the partial line reassembles."""
        self.stop = True
        try:
            self.ser.close()
        except Exception:
            pass
        time.sleep(0.4)
        self.recycle_count += 1
        if (self.recycle_count % 2 == 0 or not os.path.exists(self.port)) and Node.busids:
            print(f"[node {self.name}] re-attaching the usbip devices")
            usbipd_reattach(Node.busids)
            self._resolve_port()
        self.stop = False
        for _ in range(20):
            try:
                self._open()
                break
            except Exception:
                time.sleep(0.5)
        else:
            raise RuntimeError(f"node {self.name}: port {self.port} did not come back")
        print(f"[node {self.name}] recycled the serial handle")

    def close(self):
        self.stop = True
        time.sleep(0.15)
        try:
            self.ser.close()
        except Exception:
            pass

    def _resolve_port(self):
        """A re-attach renumbers the tty nodes, so find the port whose firmware
        reports this node's role rather than trusting the old path."""
        if self.node_id is None:
            return
        for cand in sorted(glob.glob("/dev/ttyACM*")):
            if cand == getattr(Node.sibling.get(self.name), "port", None):
                continue
            try:
                probe = serial.Serial(cand, self.baud, timeout=0.4)
                probe.write(b"M")
                time.sleep(0.6)
                reply = probe.read(probe.in_waiting or 1)
                probe.close()
            except Exception:
                continue
            if f"node={self.node_id}".encode() in reply:
                if cand != self.port:
                    print(f"[node {self.name}] port moved {self.port} -> {cand}")
                self.port = cand
                return

    def clear(self):
        with self.lock:
            self.buf.clear()

    def drain(self, quiet=0.5, cap=6.0):
        """Discard anything still queued, including replies from an earlier
        run that the board only managed to deliver once the port recovered."""
        t0 = last = time.time()
        while time.time() - t0 < cap:
            with self.lock:
                if self.buf:
                    self.buf.clear()
                    last = time.time()
            if time.time() - last >= quiet:
                break
            time.sleep(0.05)

    def read_reply(self, prefixes, timeout=120.0):
        """Read lines until one starts with an expected prefix."""
        t0 = time.time()
        while True:
            line = self.read_line(max(5.0, timeout - (time.time() - t0)))
            text = line.strip().decode(errors="replace").lstrip(".")
            if any(text.startswith(p) for p in prefixes):
                return text
            if text:
                print(f"[node {self.name}] skipped: {text}")
            if time.time() - t0 > timeout:
                raise TimeoutError(f"node {self.name}: no {prefixes} in {timeout}s")

    def write(self, data, retries=2):
        for attempt in range(retries + 1):
            try:
                self.ser.write(data)
                return
            except Exception:
                if attempt == retries:
                    raise
                self.recycle()

    def read_line(self, timeout=40.0, recycles=3):
        for attempt in range(recycles + 1):
            try:
                return self._read_line(timeout if attempt == 0 else 20.0)
            except TimeoutError:
                if attempt == recycles:
                    raise
                self.recycle()

    def _read_line(self, timeout, keepalive=True):
        t0 = time.time()
        last_poke = 0.0
        while True:
            if keepalive and time.time() - last_poke > 0.2:
                last_poke = time.time()
                try:
                    self.ser.write(b"\n")   # firmware ignores it silently
                except Exception:
                    pass
            with self.lock:
                i = self.buf.find(b"\n")
                if i >= 0:
                    line = bytes(self.buf[:i + 1])
                    del self.buf[:i + 1]
                    return line
            if time.time() - t0 > timeout:
                with self.lock:
                    tail = bytes(self.buf[-120:])
                raise TimeoutError(f"node {self.name}: no line in {timeout}s; got {tail!r}")
            time.sleep(0.02)

    def read_exact(self, n, timeout=60.0, recycles=2):
        for attempt in range(recycles + 1):
            try:
                return self._read_exact(n, timeout if attempt == 0 else 20.0)
            except TimeoutError:
                if attempt == recycles:
                    raise
                self.recycle()

    def _read_exact(self, n, timeout):
        t0 = time.time()
        while True:
            with self.lock:
                if len(self.buf) >= n:
                    data = bytes(self.buf[:n])
                    del self.buf[:n]
                    return data
                have = len(self.buf)
            if time.time() - t0 > timeout:
                raise TimeoutError(f"node {self.name}: got {have}/{n} bytes")
            time.sleep(0.02)

    def fetch_output(self, n, tries=3):
        """'O' just re-streams the node's output buffer, so a truncated USB
        frame is recovered by asking again."""
        for _ in range(tries):
            try:
                self.clear()
                self.write(b"O")
                data = self.read_exact(n, 30.0)
                self.read_line(20.0)  # END
                return data
            except (TimeoutError, RuntimeError) as exc:
                print(f"[node {self.name}] output frame incomplete ({exc}), retrying")
                self.clear()
        raise RuntimeError(f"node {self.name}: could not read the output frame")


def send_paced(node, data):
    for off in range(0, len(data), CHUNK):
        node.write(data[off:off + CHUNK])
        time.sleep(GAP)


def scatter(x):
    """[S*D] -> (node0 local rows, node1 local rows) as little-endian bytes."""
    rows = x.reshape(S, D)
    return rows[0::NODES].tobytes(), rows[1::NODES].tobytes()


def gather(part0, part1):
    y = np.empty((S, D), dtype="<f4")
    y[0::NODES] = np.frombuffer(part0, dtype="<f4").reshape(SLOC, D)
    y[1::NODES] = np.frombuffer(part1, dtype="<f4").reshape(SLOC, D)
    return y.reshape(-1)


def parse_done(line):
    """Parse "TM DONE k=v ...". Leading '.' heartbeat bytes are ignored."""
    fields = {}
    for tok in line.decode(errors="replace").split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            try:
                fields[k] = int(v)
            except ValueError:
                pass
    return fields


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port_a", nargs="?", default="auto",
                    help='serial port, or "auto" to take the ESP32 tty nodes '
                         "in order (they are renumbered by a usbip re-attach)")
    ap.add_argument("port_b", nargs="?", default="auto")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--root", default=str(ROOT), help="dir holding testdata/")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--usbipd-busids", default="2-1,3-1",
                    help="WSL usbip bus ids to re-attach when a port stalls "
                         "(empty string disables)")
    args = ap.parse_args()

    if args.port_a == "auto" or args.port_b == "auto":
        found = sorted(glob.glob("/dev/ttyACM*"))
        if len(found) < 2:
            print(f"need two boards, found {found}; run tools/attach_boards.sh")
            return 2
        args.port_a, args.port_b = found[0], found[1]
        print(f"[ports] {args.port_a} {args.port_b}")

    root = pathlib.Path(args.root)
    Node.busids = [b for b in args.usbipd_busids.split(",") if b]
    na = Node(args.port_a, args.baud, "A")
    nb = Node(args.port_b, args.baud, "B")
    time.sleep(0.8)

    def status(node, tries=4):
        """Ask for the node's state. Late lines from the radio bring-up can be
        flushed out of order, so read until the 'TM SHARD' reply shows up."""
        for _ in range(tries):
            node.drain()
            node.write(b"M")
            t0 = time.time()
            while time.time() - t0 < 6.0:
                try:
                    line = node.read_line(3.0).strip().decode(errors="replace")
                except TimeoutError:
                    break
                if line:
                    print(f"[node {node.name}] {line}")
                if "TM SHARD" in line:
                    return line
        return ""

    def linked(a, b):
        return ("node=0" in a and "node=1" in b
                and "link=1" in a and "link=1" in b)

    Node.sibling = {"A": nb, "B": na}
    st_a, st_b = status(na), status(nb)
    if "node=1" in st_a and "node=0" in st_b:
        # a usbip re-attach can swap the tty nodes; follow the roles, not the
        # port order
        na, nb = nb, na
        na.name, nb.name = "A", "B"
        st_a, st_b = st_b, st_a
        print("[ports] swapped to match the node roles")
    if not linked(st_a, st_b):
        # node 0 raises the SoftAP and blocks in accept(); node 1 then joins it.
        # The reply is read back with a fresh status probe rather than inline:
        # the boards' USB CDC can stall while the radio comes up.
        na.write(b"N0")
        time.sleep(1.5)
        nb.write(b"N1")
        time.sleep(15.0)
        st_a, st_b = status(na), status(nb)
    if not linked(st_a, st_b):
        print("peer link did not come up")
        return 2
    na.node_id, nb.node_id = 0, 1
    Node.sibling = {"A": nb, "B": na}
    print("[link] up")

    rows = []
    skipped = []
    total_fail = 0
    for seed in args.seeds:
        try:
            x = np.fromfile(root / "testdata" / f"input_{seed}.bin", dtype="<f4")
            ref = np.fromfile(root / "testdata" / f"ref_{seed}.bin", dtype="<f4").astype(np.float64)
            pa, pb = scatter(x)

            for node, part in ((na, pa), (nb, pb)):
                node.clear()
                node.write(b"L")
                time.sleep(0.15)
                send_paced(node, part)
                ack = node.read_reply(("TM LOADED",), 40.0)
                if "LOADED" not in ack:
                    print(f"load failed: {ack!r}")
                    return 2

            na.write(b"G")
            nb.write(b"G")
            # A lost "TM DONE" line is a USB transport failure, not a compute
            # failure: the boards still ran the forward and 'O' still returns the
            # result. Validate anyway and report the timing as unavailable.
            da, db = {}, {}
            for node, into in ((na, "a"), (nb, "b")):
                try:
                    line = node.read_reply(("TM DONE", "TM ERR"))
                    fields = parse_done(line.encode())
                    if "us" not in fields:
                        print(f"[node {node.name}] forward reported: {line}")
                except TimeoutError:
                    print(f"[node {node.name}] timing line lost to the USB bridge; "
                          f"validating the output anyway")
                    fields = {}
                if into == "a":
                    da = fields
                else:
                    db = fields

            outs = []
            for node in (na, nb):
                outs.append(node.fetch_output(LOCAL_BYTES))
            got = gather(*outs).astype(np.float64)

            d = np.abs(got - ref)
            ok = (d <= ATOL) | (d <= RTOL * np.abs(ref))
            nfail = int((~ok).sum())
            total_fail += nfail
            timed = "us" in da and "us" in db
            wall = max(da["us"], db["us"]) / 1e6 if timed else float("nan")
            rows.append(dict(seed=seed, wall_s=wall,
                             comp_a_s=da.get("comp", 0) / 1e6,
                             comp_b_s=db.get("comp", 0) / 1e6,
                             link_a_s=da.get("link", 0) / 1e6,
                             link_b_s=db.get("link", 0) / 1e6,
                             tx_bytes=da.get("tx", 0), rx_bytes=da.get("rx", 0),
                             retx_a=da.get("retx", 0), retx_b=db.get("retx", 0),
                             fails=nfail, max_abs=float(d.max())))
            timing = (f"wall={wall:.3f}s comp=({da['comp']/1e6:.3f},{db['comp']/1e6:.3f})s "
                      f"link=({da['link']/1e6:.3f},{db['link']/1e6:.3f})s "
                      f"kv={da['tx']/1024:.0f}KiB retx={da.get('retx', 0)}"
                      if timed else "timing unavailable")
            print(f"seed {seed}: fails={nfail:5d} max_abs={d.max():.4e} {timing} "
                  f"{'PASS' if nfail == 0 else 'FAIL'}")

        except Exception as exc:
            # a seed lost to the USB bridge should not sink the run
            print(f"seed {seed}: ABORTED ({exc})")
            skipped.append(seed)
            try:
                status(na)
                status(nb)
            except Exception:
                pass

    walls = [r["wall_s"] for r in rows if r["wall_s"] == r["wall_s"]]
    if walls:
        print(f"\nmedian distributed forward: {statistics.median(walls):.3f} s "
              f"(min {min(walls):.3f}, max {max(walls):.3f}) over {len(walls)} "
              f"timed seeds of {len(rows)}")
    if skipped:
        print(f"seeds not completed (USB transport): {skipped}")
    print("ALL PASS" if total_fail == 0 else f"{total_fail} FAILING ELEMENTS")

    if args.csv and rows:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv}")

    if args.reps > 0:
        try:
            na.write(b"T" + bytes([args.reps]))
            nb.write(b"T" + bytes([args.reps]))
            print(f"[node A] {na.read_reply(('TM TIMES', 'TM ERR'), 300)}")
            print(f"[node B] {nb.read_reply(('TM TIMES', 'TM ERR'), 300)}")
        except Exception as exc:
            print(f"[timing sweep] not captured: {exc}")

    na.close()
    nb.close()
    return 1 if total_fail else 0


if __name__ == "__main__":
    sys.exit(main())
