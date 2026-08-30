#!/usr/bin/env python3
"""tp_analyze.py - turn a raw tinyprof capture into the canonical artifact.

Three rules shape this file:

1. Nothing is asserted without a method. Every derived value carries either
   `"measured": true` or a `"method"` string, so CONTRIBUTING.md's
   measured-versus-projected requirement is enforced by the schema rather than
   by remembering to write it in prose.

2. Nesting is made explicit. `quant` is measured inside `qkv`, `gelu` inside
   `f2`, and the three attention phases inside `attn`. Summing inclusive times
   would exceed the forward wall and a top-10 ranking would double-count. Every
   ranking here uses exclusive time.

3. Traffic is the measured call count times a declared per-call model, and the
   model is checked against the measured count before it is trusted.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import statistics

import tp_parse

HERE = pathlib.Path(__file__).resolve().parent


def _load_score_module(project: pathlib.Path):
    """Import the existing score.py by path rather than copying its constants.

    score.py already encodes the FLOP count, the int-MAC ceiling and the
    soft-float peak with their primary sources. Re-deriving them here would
    create a second set of numbers that can disagree with `make score`.
    """
    p = project / "tools" / "score.py"
    if not p.exists():
        return None
    spec = importlib.util.spec_from_file_location("tm_score", p)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        return None
    return mod


def _ticks_to_us(ticks: float, tick_hz: int) -> float:
    return ticks * 1e6 / tick_hz if tick_hz else 0.0


def analyze(raw: dict, spec: dict | None = None, elf_info: dict | None = None,
            project: pathlib.Path | None = None, provenance: dict | None = None) -> dict:
    hdr = tp_parse.first(raw, "hdr") or {}
    env = tp_parse.first(raw, "env") or {}
    wall = tp_parse.first(raw, "wall") or {}
    ovh = tp_parse.first(raw, "ovh") or {}
    fwd = tp_parse.first(raw, "fwd") or {}
    ops_raw = tp_parse.every(raw, "op")

    tick_hz = int(hdr.get("tick_hz") or 0)
    ovh_ticks_per_probe = (ovh.get("milliticks_per_probe") or 0) / 1000.0
    reps = int(env.get("reps") or fwd.get("reps") or 1) or 1
    shape = {k: int(hdr.get(k, 0)) for k in ("S", "D", "H", "F", "L")}

    # Wall time per forward. The device 'T' command and host driver both report
    # a whole-run figure; `wall` is the last forward's PBT/PET bracket. Prefer
    # the explicit per-forward number when the driver supplied one.
    if fwd.get("per_forward_us"):
        per_forward_us = float(fwd["per_forward_us"])
        wall_source = "driver total / reps"
    elif wall.get("us"):
        per_forward_us = float(wall["us"])
        wall_source = "firmware PBT/PET bracket, last forward"
    else:
        per_forward_us = 0.0
        wall_source = "unavailable"

    by_index = {int(o["i"]): o for o in ops_raw}
    child_ticks: dict[int, int] = {}
    for o in ops_raw:
        parent = int(o.get("parent", -1))
        if parent >= 0:
            child_ticks[parent] = child_ticks.get(parent, 0) + int(o.get("ticks", 0))

    ops = []
    for o in sorted(ops_raw, key=lambda r: int(r["i"])):
        idx = int(o["i"])
        name = str(o["name"])
        incl = int(o.get("ticks", 0))
        n = int(o.get("n", 0))
        parent = int(o.get("parent", -1))
        excl = incl - child_ticks.get(idx, 0)

        # First-order overhead correction. Each zone costs two clock reads; the
        # closing read's latency lands inside the measured interval, the opening
        # one does not. A child's probes sit inside its parent's inclusive time
        # too, but they cancel when children are subtracted, so correcting each
        # zone by n * 1 probe and re-deriving exclusive time is consistent.
        # Reported alongside the raw figure, never instead of it.
        own_probe_ticks = n * ovh_ticks_per_probe
        child_probe_ticks = sum(
            int(c.get("n", 0)) * ovh_ticks_per_probe
            for c in ops_raw if int(c.get("parent", -1)) == idx)
        incl_corr = max(0.0, incl - own_probe_ticks - child_probe_ticks)
        excl_corr = max(0.0, excl - own_probe_ticks)

        incl_us = _ticks_to_us(incl, tick_hz) / reps
        excl_us = _ticks_to_us(excl, tick_hz) / reps
        n_fwd = n / reps
        avg_us = _ticks_to_us(incl, tick_hz) / n if n else 0.0

        # A zone whose average is near the tick period is measuring its own
        # probe more than the work. Flagged rather than quietly reported: it is
        # the honest reading of `res1` at ~30 ns of real work per call.
        tick_us = 1e6 / tick_hz if tick_hz else 0.0
        rec = {
            "index": idx,
            "name": name,
            "parent": by_index[parent]["name"] if parent >= 0 and parent in by_index else None,
            "calls_per_forward": n_fwd,
            "inclusive_us_per_forward": round(incl_us, 3),
            "exclusive_us_per_forward": round(excl_us, 3),
            "avg_us_per_call": round(avg_us, 4),
            "share_of_forward_pct": round(100.0 * excl_us / per_forward_us, 3) if per_forward_us else None,
            "exclusive_us_per_forward_overhead_corrected": round(
                _ticks_to_us(excl_corr, tick_hz) / reps, 3),
            "inclusive_us_per_forward_overhead_corrected": round(
                _ticks_to_us(incl_corr, tick_hz) / reps, 3),
            "probe_overhead_pct_of_zone": (
                round(100.0 * own_probe_ticks / incl, 2) if incl > 0 else None),
            "resolution_limited": bool(tick_us and avg_us < 20 * tick_us),
            "measured": True,
        }
        if excl < 0:
            # Children out-summing the parent means the parent bracket does not
            # actually enclose them. Surfaced loudly instead of clamped to zero.
            rec["warning"] = ("children exceed parent: the declared nesting does not "
                              "match where the zones were placed")
        ops.append(rec)

    by_name = {o["name"]: o for o in ops}

    # ---- traffic -------------------------------------------------------
    traffic = None
    if spec:
        rows, flash_total, sram_total, unvalidated = [], 0, 0, []
        for name, s in spec.get("ops", {}).items():
            op = by_name.get(name)
            if not op:
                continue
            n_fwd = op["calls_per_forward"]
            expected = s.get("n_per_forward")
            ok = expected is None or abs(n_fwd - expected) < 1e-6
            if not ok:
                unvalidated.append(
                    f"{name}: measured {n_fwd:g} calls/forward, spec expects {expected}")
            fb = s.get("flash_xip_bytes_per_call", 0) * n_fwd
            sb = s.get("sram_bytes_per_call", 0) * n_fwd
            flash_total += fb
            sram_total += sb
            us = op["exclusive_us_per_forward"]
            rows.append({
                "op": name,
                "calls_per_forward": n_fwd,
                "flash_xip_bytes_per_forward": int(fb),
                "sram_bytes_per_forward": int(sb),
                "achieved_mb_s": round((fb + sb) / us, 2) if us > 0 else None,
                "confidence": s.get("confidence"),
                "why": s.get("why"),
                "closed_form": s.get("closed_form"),
                "call_count_validated": ok,
            })
        traffic = {
            "flash_xip_bytes_per_forward": int(flash_total),
            "sram_bytes_per_forward": int(sram_total),
            "by_op": sorted(rows, key=lambda r: -r["flash_xip_bytes_per_forward"]),
            "validated": not unvalidated,
            "validation_notes": unvalidated,
            "measured": False,
            "method": ("declared bytes-per-call from spec/case02.json multiplied by the "
                       "MEASURED call count; each entry carries its derivation, and any "
                       "op whose measured call count disagrees with the spec is listed "
                       "in validation_notes and must not be trusted"),
        }

    # ---- roofline / MFU -------------------------------------------------
    roofline = None
    if project:
        sc = _load_score_module(pathlib.Path(project))
        if sc and per_forward_us > 0:
            t_s = per_forward_us / 1e6
            try:
                fl = sc.flops_forward()
                lo = sc.score_case(fl, t_s, bw=sc.BW_LO)
                hi = sc.score_case(fl, t_s, bw=sc.BW_HI)
                roofline = {
                    "seconds_per_forward": round(t_s, 6),
                    "flops": {k: v for k, v in fl.items()},
                    "achieved_mflop_s": round(lo["achieved_mflop_s"], 3),
                    "mfu_mix": round(lo["mfu_mix"], 4),
                    "mfu_int": round(lo["mfu_int"], 6),
                    "arithmetic_intensity_flop_per_byte": {
                        "total": round(lo["ai_flop_b"], 4),
                        "gemm": round(lo["ai_gemm"], 4),
                        "float_ops": round(lo["ai_fp"], 4),
                        "source": "score.py operand-traffic model",
                    },
                    "exscore": {
                        "at_bw_320MB_s": round(lo["exscore_lo"], 4) if lo["exscore_lo"] else None,
                        "at_bw_640MB_s": round(hi["exscore_lo"], 4) if hi["exscore_lo"] else None,
                        "roofline_seconds": round(lo["roof_t"], 6),
                    },
                    "peaks": {
                        "int_mac_flop_s": sc.P_INT,
                        "softfloat_flop_s": sc.P_SOFTFP,
                        "softfloat_range": [sc.P_SOFTFP_LO, sc.P_SOFTFP_HI],
                        "sram_bw_range_B_s": [sc.BW_LO, sc.BW_HI],
                    },
                    "measured": True,
                    "method": ("FLOP count, peak constants and the roofline are imported "
                               "from the project's own tools/score.py, so this cannot "
                               "disagree with `make score`. Only t_s comes from tinyprof."),
                }
                # The peaks in score.py are ESP32-C3 constants: 160 MMAC/s and a
                # ~2 MFLOP/s soft-float ceiling. Dividing a host time by them
                # yields an MFU above 100%, which is not a good score - it is a
                # category error. Flag it rather than print it.
                if str(env.get("device", "")).startswith("host"):
                    roofline["applicable"] = False
                    roofline["not_applicable_reason"] = (
                        "the peak constants are ESP32-C3 figures (160 MMAC/s int, "
                        "~2 MFLOP/s soft-float) and this capture came from a host "
                        "with a hardware FPU. MFU and ExScore are meaningless here; "
                        "the FLOP count and the arithmetic intensities still hold, "
                        "because those depend on the model shape, not the machine.")
                else:
                    roofline["applicable"] = True
                # Second AI, from tinyprof's own measured-call-count traffic rather
                # than score.py's 4 B/MAC estimate. Two independent models of the
                # same quantity, reported side by side: where they disagree, the
                # gap is the uncertainty, and hiding one of them hides that.
                if traffic:
                    tot_b = (traffic["flash_xip_bytes_per_forward"]
                             + traffic["sram_bytes_per_forward"])
                    if tot_b:
                        roofline["arithmetic_intensity_measured_traffic"] = {
                            "flop_per_byte": round(fl["total"] / tot_b, 4),
                            "bytes_per_forward": tot_b,
                            "method": ("total FLOPs / (flash + SRAM bytes from the "
                                       "tinyprof traffic model x measured call counts)"),
                        }
            except Exception as exc:
                roofline = {"error": f"score.py present but unusable: {exc}"}

    # ---- memory --------------------------------------------------------
    census = [{"name": r.get("name"), "bytes": int(r.get("bytes", 0)),
               "role": r.get("role", r.get("kind_"))}
              for r in tp_parse.every(raw, "arena")]
    census_total = sum(c["bytes"] for c in census)
    mem_rec = tp_parse.first(raw, "mem") or {}
    stack_rec = tp_parse.first(raw, "stack") or {}

    memory = {
        "arena_census": census,
        "arena_census_total_bytes": census_total,
        "runtime": {
            "heap_free": mem_rec.get("heap_free"),
            "heap_min_free_since_boot": mem_rec.get("heap_min_free"),
            "heap_largest_free_block": mem_rec.get("heap_largest"),
            "heap_fragmentation_bytes": (
                mem_rec["heap_free"] - mem_rec["heap_largest"]
                if mem_rec.get("heap_free") is not None
                and mem_rec.get("heap_largest") is not None else None),
            "stack_hwm_bytes": stack_rec.get("hwm_bytes"),
            "stack_task": stack_rec.get("task"),
            "measured": bool(mem_rec or stack_rec),
            "note": ("absent on a host capture: there is no FreeRTOS task or "
                     "ESP heap to interrogate"),
        },
        "static": elf_info,
    }
    if elf_info:
        bss = elf_info["dram"]["bss"]
        memory["census_vs_elf"] = {
            "census_total": census_total,
            "elf_dram_bss": bss,
            "unattributed_bytes": bss - census_total,
            "note": ("the remainder is framework/driver static state, not model "
                     "workspace. A sudden jump here means a buffer was added "
                     "without being declared in the census."),
        }

    # ---- overhead -------------------------------------------------------
    mt = ovh.get("milliticks_per_probe")
    probes = ovh.get("probes")
    overhead = None
    if mt is not None and probes is not None and tick_hz:
        ov_us = _ticks_to_us(probes * mt / 1000.0, tick_hz) / reps
        overhead = {
            "ns_per_probe": round(mt / 1000.0 * 1e9 / tick_hz, 2),
            "probes_per_forward": probes / reps,
            "estimated_us_per_forward": round(ov_us, 2),
            "estimated_pct_of_forward": round(100.0 * ov_us / per_forward_us, 3) if per_forward_us else None,
            "measured": True,
            "method": ("timed loop of 4096 back-to-back tick reads, executed on the "
                       "device at dump time; two reads per zone. Reported, not "
                       "subtracted - an overhead that differs between two builds is "
                       "precisely what would inflate a speedup if folded in silently. "
                       "A per-op first-order correction is provided alongside each raw "
                       "figure so the reader can see both."),
        }
        if overhead["estimated_pct_of_forward"] and overhead["estimated_pct_of_forward"] > 5:
            overhead["warning"] = (
                f"instrumentation accounts for {overhead['estimated_pct_of_forward']:.1f}% "
                "of the measured forward. The zone placement is too fine for this "
                "build: re-capture with coarser zones (for the baseline firmware, "
                "-DTINYPROF_ATTN_PHASES=0) before quoting these per-op times.")

    # Both rankings use exclusive time and the full op set. Exclusive is what
    # makes it safe to rank a parent alongside its children in one table.
    top_time = sorted(ops, key=lambda o: -o["exclusive_us_per_forward"])[:10]
    top_calls = sorted(ops, key=lambda o: -o["calls_per_forward"])[:10]

    gates = [{k: v for k, v in g.items() if k != "kind"} for g in tp_parse.every(raw, "gate")]

    return {
        "tinyprof_artifact_version": 1,
        "tag": hdr.get("tag", "unknown"),
        "device": env.get("device", "esp32c3"),
        "shape": shape,
        "cpu_mhz": hdr.get("mhz"),
        "tick_hz": tick_hz,
        "tick_resolution_ns": round(1e9 / tick_hz, 3) if tick_hz else None,
        "reps": reps,
        "mode": env.get("mode"),
        "wall": {
            "us_per_forward": round(per_forward_us, 3),
            "s_per_forward": round(per_forward_us / 1e6, 6),
            "source": wall_source,
            "measured": True,
        },
        "accuracy": {
            "gate": "abs_err <= 0.002 OR rel_err <= 0.02, per output element",
            "seeds": gates,
            "all_pass": all(g.get("fails") == 0 for g in gates) if gates else None,
        },
        "ops": ops,
        "top_by_exclusive_time": [o["name"] for o in top_time],
        "top_by_calls": [o["name"] for o in top_calls],
        "traffic": traffic,
        "memory": memory,
        "roofline": roofline,
        "overhead": overhead,
        "kbench": raw.get("kbench", []),
        "cycbench": raw.get("cycbench", []),
        "provenance": provenance or {},
        "unparsed_lines": raw.get("unparsed", []),
    }


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw", help="raw capture JSON from tp_collect, or '-' for TPROF text on stdin")
    ap.add_argument("--spec", default=str(HERE / "spec" / "case02.json"))
    ap.add_argument("--project", help="firmware project dir (for tools/score.py)")
    ap.add_argument("--elf")
    ap.add_argument("--map", dest="mapfile")
    ap.add_argument("--toolchain")
    ap.add_argument("-o", "--output", default="-")
    a = ap.parse_args()

    import sys
    if a.raw == "-":
        raw = tp_parse.parse_lines(sys.stdin)
        prov = {}
    else:
        doc = json.loads(pathlib.Path(a.raw).read_text())
        raw = doc.get("capture", doc)
        prov = doc.get("provenance", {})

    if not tp_parse.is_complete(raw):
        sys.exit("tinyprof: capture is incomplete (no op records, or no TPROF|end|ok=1). "
                 "Refusing to produce an artifact from a truncated run.")

    spec = json.loads(pathlib.Path(a.spec).read_text()) if a.spec and pathlib.Path(a.spec).exists() else None
    elf_info = None
    if a.elf:
        import tp_elf
        elf_info = tp_elf.analyze(a.elf, a.mapfile, a.toolchain)

    art = analyze(raw, spec, elf_info, a.project, prov)
    text = json.dumps(art, indent=1)
    if a.output == "-":
        print(text)
    else:
        pathlib.Path(a.output).write_text(text + "\n")
        print(f"tinyprof: wrote {a.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
