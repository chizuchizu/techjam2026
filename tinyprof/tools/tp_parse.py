#!/usr/bin/env python3
"""tp_parse.py - pure parser for the tinyprof wire format.

Deliberately free of serial, filesystem and network dependencies so it can be
tested against a fixture without a board, the way
`benchmarks/case-02/optimisation/esp32-baseline/tools/link_bench.py` keeps
`parse_feed()` importable.

Wire format is the repo's established pipe convention (see
`esp32-linkbench/src/main.cpp`): one record per line, `TPROF|<kind>|k=v|k=v`.
Unknown kinds and unknown keys are preserved rather than dropped, so a firmware
that emits more than this host build knows about does not lose data.
"""
from __future__ import annotations

import re

LINE = re.compile(r"^\s*TPROF\|([a-z_]+)\|?(.*?)\s*$")
KV = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^|]*)")

# Legacy kernels.c microbench lines, kept in their original format so existing
# readers (FLASH_TEST.md, teammates' scripts) keep working. Parsed here too so
# the artifact carries them rather than forcing a second capture.
KBENCH = re.compile(r"^KB(\d+)\s+n=(\d+)\s+avg_us=([\d.]+)\s+tot_ms=([\d.]+)")
CYCBENCH = re.compile(r"^C(\d)CYC\s+n=(\d+)\s+avg_cyc=([\d.]+)\s+avg_us=([\d.]+)")


def _coerce(v: str):
    """Numbers become numbers; everything else stays a string.

    Kept strict on purpose: a value like `8.81e-04` must parse as a float, but
    a tag like `opt23` must not become one, and a firmware version like `1.4.0`
    must survive intact.
    """
    try:
        return int(v)
    except ValueError:
        pass
    try:
        f = float(v)
    except ValueError:
        return v
    return f


def parse_lines(lines) -> dict:
    """Parse an iterable of text lines into a raw capture dict.

    Returns {"records": [...], "kbench": [...], "cycbench": [...],
             "unparsed": [...]}. No derivation happens here - that is
    tp_analyze's job, and keeping the split means a capture can be re-analyzed
    later with a corrected model without re-running the hardware.
    """
    records, kbench, cycbench, unparsed = [], [], [], []
    for raw in lines:
        line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
        line = line.rstrip("\r\n")
        if not line.strip():
            continue
        m = LINE.match(line)
        if m:
            kind, rest = m.group(1), m.group(2)
            rec = {"kind": kind}
            for k, v in KV.findall(rest):
                # A payload key must never overwrite the record type. The arena
                # record used to carry its own `kind=`, which silently turned
                # every arena line into a record of type "fp32_activation" and
                # made the census vanish. Colliding keys are kept, suffixed.
                rec[k + "_" if k == "kind" else k] = _coerce(v)
            records.append(rec)
            continue
        m = KBENCH.match(line)
        if m:
            kbench.append({"slot": int(m.group(1)), "n": int(m.group(2)),
                           "avg_us": float(m.group(3)), "tot_ms": float(m.group(4))})
            continue
        m = CYCBENCH.match(line)
        if m:
            cycbench.append({"core": int(m.group(1)), "n": int(m.group(2)),
                             "avg_cyc": float(m.group(3)), "avg_us": float(m.group(4))})
            continue
        unparsed.append(line)
    return {"records": records, "kbench": kbench,
            "cycbench": cycbench, "unparsed": unparsed}


def first(raw: dict, kind: str) -> dict | None:
    for r in raw["records"]:
        if r["kind"] == kind:
            return r
    return None


def every(raw: dict, kind: str) -> list:
    return [r for r in raw["records"] if r["kind"] == kind]


def is_complete(raw: dict) -> bool:
    """A capture is usable only if the firmware said it finished and at least
    one op record arrived.

    This is the guard that stops a wedged serial port from silently replacing a
    good artifact with an empty one - the same failure
    `esp32-linkbench/tools/run_bench.py` protects against with its
    'keeping previous result, NOT overwriting' branch.
    """
    end = first(raw, "end")
    return bool(every(raw, "op")) and end is not None and end.get("ok") == 1


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(parse_lines(sys.stdin), indent=1))
