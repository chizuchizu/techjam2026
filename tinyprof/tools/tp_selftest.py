#!/usr/bin/env python3
"""tp_selftest.py - run the whole pipeline against committed fixtures.

No board, no weights, no build: two recorded TPROF captures are replayed through
parse -> analyze -> compare -> report and the invariants are asserted. This is
what `make check` runs, and it is the reason a change to the traffic model or the
nesting table cannot land silently.

The assertions are chosen to be the ones that would actually be wrong if the tool
regressed, not merely the ones that are easy to write.
"""
from __future__ import annotations

import json
import pathlib
import sys

import tp_analyze
import tp_compare
import tp_parse
import tp_report

HERE = pathlib.Path(__file__).resolve().parent
FIX = HERE / "testdata"


def check(cond, msg):
    if not cond:
        print(f"FAIL: {msg}")
        return 1
    print(f"  ok: {msg}")
    return 0


def main() -> int:
    bad = 0
    spec = json.loads((HERE / "spec" / "case02.json").read_text())

    arts = {}
    for side in ("baseline", "optimised"):
        raw = tp_parse.parse_lines((FIX / f"fixture_{side}.tprof").read_text().splitlines())
        bad += check(tp_parse.is_complete(raw), f"{side} fixture is a complete capture")
        arts[side] = tp_analyze.analyze(raw, spec)

    b, o = arts["baseline"], arts["optimised"]

    # Nesting: exclusive times must account for the forward without exceeding it.
    for side, a in arts.items():
        tot = sum(x["exclusive_us_per_forward"] for x in a["ops"])
        wall = a["wall"]["us_per_forward"]
        bad += check(0.90 <= tot / wall <= 1.02,
                     f"{side}: exclusive times sum to {100 * tot / wall:.1f}% of the "
                     f"forward (nesting table is consistent)")

    # Every zone reports, including the sub-microsecond ones. This is the bug the
    # tick clock fixed: with a 1 us timer and an `if (d > 0)` guard, res1 vanished.
    names = {x["name"] for x in o["ops"]}
    bad += check({"res1", "res2"} <= names,
                 "cheap residual zones are present, not dropped by clock resolution")
    for nm in ("res1", "res2"):
        rec = next(x for x in o["ops"] if x["name"] == nm)
        bad += check(rec["calls_per_forward"] == 4,
                     f"{nm} reports all 4 calls/forward, not a truncated count")

    # Attention phase counts are deterministic: L*H*S per forward, exactly.
    qk = next(x for x in o["ops"] if x["name"] == "attn_qk")
    S, H, L = o["shape"]["S"], o["shape"]["H"], o["shape"]["L"]
    bad += check(qk["calls_per_forward"] == S * H * L,
                 f"attn_qk = S*H*L = {S * H * L} calls/forward")

    # The traffic model reproduces the published flash figure from measured counts.
    t = o["traffic"]
    bad += check(t["validated"], "traffic model agrees with measured call counts")
    weights_only = sum(
        spec["ops"][k]["flash_xip_bytes_per_call"] * spec["ops"][k]["n_per_forward"]
        for k in ("quant", "oproj", "f1", "f2"))
    bad += check(weights_only == 786432,
                 f"weight-matrix flash traffic = {weights_only:,} B/forward "
                 f"(= the 768 KiB in optimisations/20_profiling_memory_compute.md)")

    # The baseline capture must still reproduce the historical accuracy result.
    g = b["accuracy"]["seeds"][0]
    bad += check(g["fails"] == 0 and abs(g["max_abs"] - 8.1241e-4) < 1e-7,
                 f"baseline seed 0 reproduces max_abs={g['max_abs']:.4e}, matching "
                 f"baseline/results/teammate_esp32_baseline_seed0_v1.log")

    # Comparison and report must build, and the report must be self-contained.
    cmp = tp_compare.compare(b, o, force=True)
    bad += check(cmp["speedup"] is not None, "comparison produces a speedup")
    absent = [r["op"] for r in cmp["ops"] if r["absent"]]
    bad += check(all(r["baseline_us"] is None or r["optimised_us"] is None
                     for r in cmp["ops"] if r["absent"]),
                 f"ops present on one side only are marked absent, not zero ({absent})")

    html = tp_report.render(cmp, b, o)
    for token in ("http://", "https://", "<img"):
        bad += check(token not in html, f"report contains no {token!r}")
    bad += check(tp_report.render(cmp, b, o) == html,
                 "report render is deterministic (diffable in git)")

    print("\n" + ("tinyprof selftest: ALL PASS" if not bad
                  else f"tinyprof selftest: {bad} FAILED"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
