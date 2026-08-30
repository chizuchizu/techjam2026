#!/usr/bin/env python3
"""tp_collect.py - drive an instrumented board and write a raw tinyprof capture.

Protocol is the firmware's existing one (src/main.cpp), plus two commands the
tinyprof build adds:

    M            -> "TM <mode> <S> <D>"          identify
    S            -> "TM OK mode=<m>"             ensure Q12 weights are scanned
    R <64 KB in> -> 64 KB out + "END forward=..  us=.."   one gated forward
    T <n>        -> warm-up + n timed forwards
    Z            -> "TPROF|reset|ok=1"           zero the counters
    G            -> the full TPROF| record set

Three device-handling behaviours are inherited deliberately, each for a reason
already documented in this repo rather than rediscovered here:

  * host->device writes are paced at 1 KB / 20 ms. The C3's native-USB CDC drops
    bursts above roughly 1 KB (see device_test.py's module docstring).
  * a startup "kick" sends one full zero frame, because read_input() has no
    timeout and an interrupted previous run leaves the firmware inside it.
  * an empty capture never overwrites a good artifact, mirroring the guard in
    esp32-linkbench/tools/run_bench.py. A wedged port must not silently replace
    a 42-second measurement with an empty file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import time

try:
    import serial  # pyserial
except ImportError:
    sys.exit("need pyserial: pip install pyserial")

import tp_parse

CHUNK, GAP = 1024, 0.02        # documented CDC pacing; do not raise blindly
ATOL, RTOL = 0.002, 0.02       # the benchmark's own gate


def _read_until(ser, token: bytes, timeout: float) -> bytes:
    buf, t0 = b"", time.time()
    while token not in buf:
        if time.time() - t0 > timeout:
            raise TimeoutError(f"no {token!r} within {timeout}s; tail={buf[-120:]!r}")
        chunk = ser.read(ser.in_waiting or 1)
        if chunk:
            buf += chunk
    return buf


def _read_exact(ser, n: int, timeout: float) -> bytes:
    buf, t0 = b"", time.time()
    while len(buf) < n:
        if time.time() - t0 > timeout:
            raise TimeoutError(f"got {len(buf)} of {n} bytes")
        c = ser.read(min(4096, n - len(buf)))
        if c:
            buf += c
            t0 = time.time()
    return buf


def _send_paced(ser, data: bytes) -> None:
    for off in range(0, len(data), CHUNK):
        ser.write(data[off:off + CHUNK])
        time.sleep(GAP)


def _drain_quiet(ser, quiet_s=2.0, max_wait=180.0) -> None:
    t0, last = time.time(), time.time()
    while time.time() - t0 < max_wait:
        if ser.read(ser.in_waiting or 1):
            last = time.time()
        elif time.time() - last > quiet_s:
            return
        time.sleep(0.01)


def _sha(path: pathlib.Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return ""


def collect(port: str, project: pathlib.Path, tag: str, seeds: list[int],
            reps: int, baud: int, forward_timeout: float,
            n_out: int) -> tuple[list[str], dict]:
    """Return (captured lines, gate results). Leaves the board idle on any exit."""
    lines: list[str] = []
    gates: list[dict] = []
    ser = serial.Serial(port, baud, timeout=1.0)
    try:
        time.sleep(0.4)
        _drain_quiet(ser, 1.0, 20.0)

        # Kick: a zero frame returns the firmware to idle if a previous run died
        # inside read_input(). Cheap insurance against a 42 s hang.
        ser.write(b"R")
        _send_paced(ser, bytes(n_out * 4))
        _read_exact(ser, n_out * 4, forward_timeout)
        _read_until(ser, b"\n", 20.0)

        ser.write(b"S")
        lines.append(_read_until(ser, b"\n", 30.0).decode("utf-8", "replace").strip())
        ser.write(b"Z")
        lines.append(_read_until(ser, b"\n", 20.0).decode("utf-8", "replace").strip())

        import numpy as np
        for s in seeds:
            ref = np.fromfile(project / "testdata" / f"ref_{s}.bin", dtype=np.float32)
            inp = (project / "testdata" / f"input_{s}.bin").read_bytes()
            ser.write(b"R")
            _send_paced(ser, inp)
            out = np.frombuffer(_read_exact(ser, n_out * 4, forward_timeout),
                                dtype=np.float32)
            tail = _read_until(ser, b"\n", 30.0).decode("utf-8", "replace")
            d = np.abs(out - ref)
            rel = np.divide(d, np.abs(ref), out=np.zeros_like(d), where=np.abs(ref) > 0)
            fails = int(np.count_nonzero(~((d <= ATOL) | (d <= RTOL * np.abs(ref)))))
            gates.append({"seed": s, "fails": fails, "max_abs": float(d.max()),
                          "max_rel": float(rel.max()), "atol": ATOL, "rtol": RTOL})
            us = int(tail.split("us=", 1)[1].split()[0]) if "us=" in tail else None
            lines.append(f"TPROF|gate|seed={s}|fails={fails}|max_abs={d.max():.6e}"
                         f"|max_rel={rel.max():.6e}|atol={ATOL}|rtol={RTOL}")
            if us:
                lines.append(f"TPROF|fwd|us={us}|reps=1|per_forward_us={us}")
            print(f"  seed {s}: fails={fails} max_abs={d.max():.4e} "
                  f"{'PASS' if fails == 0 else 'FAIL'} ({us} us)")

        if reps > 0:
            ser.write(b"T" + bytes([min(9, max(1, reps))]))
            lines.append(_read_until(ser, b"\n", forward_timeout * (reps + 2))
                         .decode("utf-8", "replace").strip())

        ser.write(b"G")
        dump = _read_until(ser, b"TPROF|end", 120.0).decode("utf-8", "replace")
        dump += _read_until(ser, b"\n", 10.0).decode("utf-8", "replace")
        lines.extend(dump.splitlines())
    finally:
        try:
            _drain_quiet(ser, 0.5, 5.0)
            ser.close()
        except Exception:
            pass
    return lines, {"seeds": gates}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", required=True)
    ap.add_argument("--project", required=True, help="firmware project directory")
    ap.add_argument("--env", help="PlatformIO env to flash first (skipped if omitted)")
    ap.add_argument("--tag", default="capture")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--forward-timeout", type=float, default=180.0,
                    help="per-forward ceiling; the un-optimised build needs >60 s")
    ap.add_argument("-o", "--output", required=True)
    a = ap.parse_args()

    project = pathlib.Path(a.project).resolve()
    if a.env:
        pio = pathlib.Path.home() / ".platformio/penv/bin/pio"
        print(f"tinyprof: flashing {a.env} to {a.port}")
        subprocess.run([str(pio), "run", "-e", a.env, "-t", "upload",
                        "--upload-port", a.port], cwd=project, check=True)
        time.sleep(2.0)

    # Output element count from the firmware's own shape, not an assumption.
    import re
    cfg = (project / "src" / "tm_config.h").read_text()
    S = int(re.search(r"#define\s+TM_S\s+(\d+)", cfg).group(1))
    D = int(re.search(r"#define\s+TM_D\s+(\d+)", cfg).group(1))

    print(f"tinyprof: capturing {a.tag} from {a.port}")
    lines, _ = collect(a.port, project, a.tag, a.seeds, a.reps, a.baud,
                       a.forward_timeout, S * D)
    raw = tp_parse.parse_lines(lines)

    out = pathlib.Path(a.output)
    if not tp_parse.is_complete(raw):
        # Never let a wedged port replace a good measurement with an empty file.
        print("tinyprof: capture produced no complete profile "
              f"({len(tp_parse.every(raw, 'op'))} op records). "
              f"NOT writing {out}.", file=sys.stderr)
        (out.parent / (out.stem + ".failed.log")).write_text("\n".join(lines) + "\n")
        return 1

    elf = project / ".pio" / "build" / (a.env or "") / "firmware.elf"
    doc = {
        "capture": raw,
        "provenance": {
            "tag": a.tag,
            "port": a.port,
            "baud": a.baud,
            "pacing": {"chunk_bytes": CHUNK, "gap_s": GAP},
            "project": str(project),
            "env": a.env,
            "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "weights_sha256": _sha(project / "weights_q12.bin"),
            "elf_sha256": _sha(elf) if elf.exists() else None,
            "captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"tinyprof: wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
