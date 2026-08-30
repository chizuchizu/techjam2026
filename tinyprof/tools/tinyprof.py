#!/usr/bin/env python3
"""tinyprof - an operator-level profiler for transformer inference on ESP32-C3.

Subcommands:

  collect    drive a board (or a host build) and write a raw capture
  analyze    raw capture -> canonical artifact (per-op time, memory, traffic)
  compare    two artifacts -> one comparison
  report     comparison -> one self-contained HTML file
  case2      the whole pipeline for case 2, baseline vs optimised

Each stage writes a file and the next stage reads it, on purpose: a 42-second
device capture is expensive, and a mistake in the traffic model or the report
should be fixable by re-running `analyze`, not by re-running the hardware.

See ../README.md for what it measures and ../PRIOR_ART.md for what it claims.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CASE2 = ROOT / "benchmarks" / "case-02"
OPT = CASE2 / "optimisation" / "esp32-baseline"
V0 = CASE2 / "baseline" / "esp32-baseline-v0"


def _run(args: list[str]) -> None:
    subprocess.run([sys.executable, *args], cwd=HERE, check=True)


def _analyze_args(raw: str, project: pathlib.Path, env: str | None, out: str) -> list[str]:
    args = ["tp_analyze.py", raw, "--project", str(project), "-o", out]
    if env:
        build = project / ".pio" / "build" / env
        if (build / "firmware.elf").exists():
            args += ["--elf", str(build / "firmware.elf")]
            if (build / "firmware.map").exists():
                args += ["--map", str(build / "firmware.map")]
    return args


def cmd_case2(a) -> int:
    """Run both sides and render the comparison.

    --port/--port-v0 capture from hardware; without them the host builds are
    used, and every artifact is stamped device=host so the report refuses to
    headline it as an ESP32 result.
    """
    outdir = pathlib.Path(a.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    for side, project, env, tag, port in (
            ("baseline", V0, "esp32-tinyprof-v0", "baseline-79f284a", a.port_v0),
            ("optimised", OPT, "esp32-tinyprof", "opt23", a.port)):
        raw = outdir / f"raw_{side}.json"
        art = outdir / f"artifact_{side}.json"
        if port:
            _run(["tp_collect.py", "--port", port, "--project", str(project),
                  "--env", env, "--tag", tag, "--seeds", *map(str, a.seeds),
                  "--reps", str(a.reps), "-o", str(raw)])
            _run(_analyze_args(str(raw), project, env, str(art)))
        else:
            binary = HERE / ("host_profile" if side == "optimised" else "host_profile_v0")
            if not binary.exists():
                sys.exit(f"tinyprof: {binary.name} not built. Run `make -C {HERE} "
                         f"{'host-profile' if side == 'optimised' else 'host_profile_v0'}` "
                         f"first, or pass --port to capture from hardware.")
            text = subprocess.run(
                [str(binary), "--root", str(project), "--seed", str(a.seeds[0]),
                 "--reps", str(a.reps), "--fast"],
                capture_output=True, text=True).stdout
            raw.write_text(text)
            proc = subprocess.run(
                [sys.executable, *_analyze_args("-", project, env, str(art))],
                cwd=HERE, input=text, text=True)
            if proc.returncode:
                return proc.returncode

    cmp_path = outdir / "comparison.json"
    _run(["tp_compare.py", str(outdir / "artifact_baseline.json"),
          str(outdir / "artifact_optimised.json"), "-o", str(cmp_path)]
         + (["--force"] if a.force else []))
    _run(["tp_report.py", str(cmp_path), str(outdir / "artifact_baseline.json"),
          str(outdir / "artifact_optimised.json"), "-o", str(outdir / "report.html")])

    c = json.loads(cmp_path.read_text())
    print(f"\ntinyprof: {c['baseline']['s_per_forward']:.4f} s -> "
          f"{c['optimised']['s_per_forward']:.4f} s = {c['speedup']:.2f}x "
          f"({c['baseline']['device']})")
    print(f"tinyprof: report at {outdir / 'report.html'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="tinyprof", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name, script in (("collect", "tp_collect.py"), ("analyze", "tp_analyze.py"),
                         ("compare", "tp_compare.py"), ("report", "tp_report.py")):
        p = sub.add_parser(name, add_help=False,
                           help=f"passthrough to {script}")
        p.set_defaults(_script=script)

    p = sub.add_parser("case2", help="baseline vs optimised, end to end")
    p.add_argument("--port", help="serial port for the optimised board")
    p.add_argument("--port-v0", help="serial port for the baseline board")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--outdir", default=str(CASE2 / "optimisation" / "results" / "tinyprof"))
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_case2)

    args, rest = ap.parse_known_args()
    if getattr(args, "_script", None):
        _run([args._script, *rest])
        return 0
    if rest:
        ap.error(f"unrecognised arguments: {' '.join(rest)}")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
