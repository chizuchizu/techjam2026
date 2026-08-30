#!/usr/bin/env python3
"""tp_compare.py - join two tinyprof artifacts into one comparison.

The comparison is the deliverable, and it is also where a profiler is easiest
to mislead with. Three things this file refuses to do:

  * Compare captures of different shapes, different weights, or one device and
    one host capture, without --force. A speedup across a changed input is not
    a speedup, and CONTRIBUTING.md says so explicitly.
  * Report a missing op as zero. An op absent on one side is marked `absent`,
    because "attention got 100x faster" and "attention is no longer a zone"
    look identical once you write a 0 in the table.
  * Rank on inclusive time. Rankings use exclusive time, so a parent and its
    children can sit in the same table without double counting.

The headline it produces is the share shift, not just the ratio: attention was
71% of the baseline forward, and what it became is the actual story of the
optimisation work.
"""
from __future__ import annotations

import json
import pathlib


def _ops(art: dict) -> dict[str, dict]:
    return {o["name"]: o for o in art.get("ops", [])}


def _checks(base: dict, opt: dict) -> list[str]:
    problems = []
    if base.get("shape") != opt.get("shape"):
        problems.append(f"shape differs: {base.get('shape')} vs {opt.get('shape')}")
    if base.get("device") != opt.get("device"):
        problems.append(
            f"device differs: {base.get('device')} vs {opt.get('device')} - "
            "a host time and a device time are not comparable")
    bw = (base.get("provenance") or {}).get("weights_sha256")
    ow = (opt.get("provenance") or {}).get("weights_sha256")
    if bw and ow and bw != ow:
        problems.append("weights differ between the two captures")
    for side, art in (("baseline", base), ("optimised", opt)):
        acc = art.get("accuracy") or {}
        if acc.get("all_pass") is False:
            problems.append(f"{side} capture did not pass the accuracy gate")
    return problems


def compare(base: dict, opt: dict, force: bool = False) -> dict:
    problems = _checks(base, opt)
    if problems and not force:
        raise SystemExit("tinyprof: refusing to compare:\n  - " + "\n  - ".join(problems))

    bo, oo = _ops(base), _ops(opt)
    b_wall = base["wall"]["us_per_forward"]
    o_wall = opt["wall"]["us_per_forward"]

    rows = []
    for name in sorted(set(bo) | set(oo)):
        b, o = bo.get(name), oo.get(name)
        row = {"op": name, "absent": None}
        if b is None or o is None:
            row["absent"] = "optimised" if o is None else "baseline"
        bt = b["exclusive_us_per_forward"] if b else None
        ot = o["exclusive_us_per_forward"] if o else None
        row.update({
            "baseline_us": bt,
            "optimised_us": ot,
            "baseline_pct_of_wall": b["share_of_forward_pct"] if b else None,
            "optimised_pct_of_wall": o["share_of_forward_pct"] if o else None,
            "speedup": round(bt / ot, 2) if bt and ot else None,
            "baseline_calls": b["calls_per_forward"] if b else None,
            "optimised_calls": o["calls_per_forward"] if o else None,
            "calls_comparable": (b["calls_per_forward"] == o["calls_per_forward"])
                                if b and o else None,
            "us_saved": round(bt - ot, 2) if bt is not None and ot is not None else None,
        })
        if b and o:
            row["share_shift_pct_points"] = round(
                (o["share_of_forward_pct"] or 0) - (b["share_of_forward_pct"] or 0), 2)
        rows.append(row)

    # What actually bought the speedup, in absolute microseconds removed from
    # the forward. Ratios flatter small ops; this ranks by time removed.
    by_saved = sorted([r for r in rows if r["us_saved"] is not None],
                      key=lambda r: -r["us_saved"])

    def _mem(art):
        m = art.get("memory") or {}
        st = m.get("static") or {}
        return {
            "arena_census_total": m.get("arena_census_total_bytes"),
            "dram_bss": (st.get("dram") or {}).get("bss"),
            "dram_used": (st.get("dram") or {}).get("used"),
            "dram_free": (st.get("dram") or {}).get("free"),
            "dram_capacity": (st.get("dram") or {}).get("capacity"),
            "flash_used": (st.get("flash") or {}).get("used"),
            "heap_min_free": (m.get("runtime") or {}).get("heap_min_free_since_boot"),
            "stack_hwm_bytes": (m.get("runtime") or {}).get("stack_hwm_bytes"),
        }

    def _traffic(art):
        t = art.get("traffic") or {}
        return {
            "flash_xip_bytes_per_forward": t.get("flash_xip_bytes_per_forward"),
            "sram_bytes_per_forward": t.get("sram_bytes_per_forward"),
            "validated": t.get("validated"),
        }

    return {
        "tinyprof_comparison_version": 1,
        "baseline": {"tag": base.get("tag"), "device": base.get("device"),
                     "s_per_forward": base["wall"]["s_per_forward"]},
        "optimised": {"tag": opt.get("tag"), "device": opt.get("device"),
                      "s_per_forward": opt["wall"]["s_per_forward"]},
        "shape": opt.get("shape"),
        "speedup": round(b_wall / o_wall, 3) if o_wall else None,
        "us_removed_per_forward": round(b_wall - o_wall, 1),
        "ops": rows,
        "ranked_by_time_removed": [r["op"] for r in by_saved][:10],
        "accuracy": {"baseline": base.get("accuracy"), "optimised": opt.get("accuracy")},
        "memory": {"baseline": _mem(base), "optimised": _mem(opt)},
        "traffic": {"baseline": _traffic(base), "optimised": _traffic(opt)},
        "overhead": {"baseline": base.get("overhead"), "optimised": opt.get("overhead")},
        "warnings": problems,
        "note": ("Exclusive time throughout: a zone's own time with nested zones "
                 "subtracted. Call counts are compared but flagged when the two "
                 "builds bracket differently, which is a structural difference, not "
                 "a regression."),
    }


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("baseline")
    ap.add_argument("optimised")
    ap.add_argument("-o", "--output", default="-")
    ap.add_argument("--force", action="store_true",
                    help="compare anyway despite provenance mismatches (they stay in `warnings`)")
    a = ap.parse_args()
    base = json.loads(pathlib.Path(a.baseline).read_text())
    opt = json.loads(pathlib.Path(a.optimised).read_text())
    cmp = compare(base, opt, a.force)
    text = json.dumps(cmp, indent=1)
    if a.output == "-":
        print(text)
    else:
        pathlib.Path(a.output).write_text(text + "\n")
        print(f"tinyprof: wrote {a.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
