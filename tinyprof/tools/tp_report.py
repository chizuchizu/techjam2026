#!/usr/bin/env python3
"""tp_report.py - render a tinyprof comparison as one self-contained HTML file.

Self-contained is a requirement, not a preference: the file has to open from a
USB stick on a judge's laptop, survive being emailed, and stay readable in a
screen recording. So no CDN, no <img>, no external font - inline CSS and inline
SVG only, and the canonical comparison JSON is embedded in a <script
type="application/json"> block so the data is recoverable from the report alone.

Section order is deliberate. The accuracy gate comes before any speed chart,
because CONTRIBUTING.md is explicit that a speedup without a passing gate at the
same input shape is not a valid comparison - so the report is laid out to make
that impossible to skip past.
"""
from __future__ import annotations

import json
import pathlib

import tp_svg
from tp_svg import esc

CSS = """
/* Palette. Neutrals carry a slight cyan bias rather than pure grey - this is a
   bench-instrument readout, and the ground should read as chosen. The seven
   categorical series slots are the validated data-viz palette and are NOT
   adjusted here: they passed the CVD and contrast checks as a set, and
   re-picking one by eye would invalidate that. Semantic good/warn/bad are
   deliberately outside the series ramp so a status can never be mistaken for
   a data series. */
:root{
  color-scheme:light;
  --bg:#f6f8f8; --card:#ffffff; --card2:#eef2f2;
  --ink:#0e1417; --ink2:#4a5559; --ink3:#75817f;
  --rule:#dde4e4; --gridline:#e9eeee; --free:#e2e8e8;
  --accent:#2a78d6;
  --s0:#2a78d6; --s1:#eb6834; --s2:#1baf7a; --s3:#eda100;
  --s4:#e87ba4; --s5:#008300; --s6:#4a3aa7; --s7:#b4bcbb;
  --good:#0d7a4e; --warn:#8a5a00; --bad:#b3261e;
  --warn-bg:#fdf6e7; --bad-bg:#fdeeec;
}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){
  color-scheme:dark;
  --bg:#0f1415; --card:#171d1e; --card2:#1e2526;
  --ink:#f2f6f6; --ink2:#b3bfbe; --ink3:#828d8c;
  --rule:#2b3335; --gridline:#242b2c; --free:#272f30;
  --accent:#3987e5;
  --s0:#3987e5; --s1:#d95926; --s2:#199e70; --s3:#c98500;
  --s4:#d55181; --s5:#008300; --s6:#9085e9; --s7:#66706f;
  --good:#4ac088; --warn:#e0a83c; --bad:#f2685f;
  --warn-bg:#241d0e; --bad-bg:#26140f;
}}
:root[data-theme=dark]{
  color-scheme:dark;
  --bg:#0f1415; --card:#171d1e; --card2:#1e2526;
  --ink:#f2f6f6; --ink2:#b3bfbe; --ink3:#828d8c;
  --rule:#2b3335; --gridline:#242b2c; --free:#272f30;
  --accent:#3987e5;
  --s0:#3987e5; --s1:#d95926; --s2:#199e70; --s3:#c98500;
  --s4:#d55181; --s5:#008300; --s6:#9085e9; --s7:#66706f;
  --good:#4ac088; --warn:#e0a83c; --bad:#f2685f;
  --warn-bg:#241d0e; --bad-bg:#26140f;
}

/* Type. No web font: this file has to open from a USB stick with no network,
   and the selftest asserts it contains no external reference. So the stacks are
   chosen rather than defaulted - a humanist grotesque for prose, and a real
   monospace for every identifier and every number, because op names and byte
   counts are data and should look like it. */
:root{
  --sans:"IBM Plex Sans","Segoe UI",Inter,-apple-system,BlinkMacSystemFont,
         "Helvetica Neue",Arial,sans-serif;
  --mono:"IBM Plex Mono","SFMono-Regular",ui-monospace,Menlo,Consolas,
         "Liberation Mono",monospace;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);
  font:400 15px/1.6 var(--sans);
  font-variant-numeric:tabular-nums;
  text-rendering:optimizeLegibility}
.wrap{max-width:980px;margin:0 auto;padding:40px 22px 96px}

h1{font-size:31px;line-height:1.15;font-weight:600;letter-spacing:-.021em;
  margin:0 0 6px;text-wrap:balance}
h2{font-size:19px;font-weight:600;letter-spacing:-.012em;margin:44px 0 4px;
  padding-bottom:8px;border-bottom:2px solid var(--rule);scroll-margin-top:16px;
  text-wrap:balance}
h3{font-size:11.5px;font-weight:600;margin:26px 0 8px;color:var(--ink3);
  text-transform:uppercase;letter-spacing:.09em}
p{margin:10px 0;color:var(--ink2);max-width:74ch}
.sub{color:var(--ink3);font-size:13px;margin:0 0 4px;max-width:none}
.sub code,.toc code{font-size:12.5px}

/* Hero: the four numbers a reader wants before anything else. A grid rather
   than margins so the columns stay aligned when one value is much wider. */
.hero{display:grid;grid-template-columns:repeat(auto-fit,minmax(132px,1fr));
  gap:2px;margin:22px 0 4px;background:var(--rule);
  border:1px solid var(--rule);border-radius:6px;overflow:hidden}
.hero>div{background:var(--card);padding:14px 16px}
.hero .n{font:600 27px/1 var(--mono);letter-spacing:-.03em;color:var(--ink)}
.hero .k{font-size:10.5px;color:var(--ink3);text-transform:uppercase;
  letter-spacing:.09em;margin-bottom:8px}

.toc{display:flex;flex-wrap:wrap;gap:5px 16px;margin:20px 0 6px;
  padding:13px 15px;border:1px solid var(--rule);border-radius:6px;
  background:var(--card);font-size:12.5px}
.toc a{color:var(--ink2);text-decoration:none;white-space:nowrap;
  border-bottom:1px solid transparent}
.toc a:hover,.toc a:focus-visible{color:var(--accent);border-bottom-color:var(--accent)}

.card{background:var(--card);border:1px solid var(--rule);border-radius:6px;
  padding:16px 20px;margin:14px 0}
.card p:first-child{margin-top:0}
.card p:last-child{margin-bottom:0}

/* State is carried by a left stripe plus a word, never by colour alone. */
.banner{border-radius:6px;padding:13px 16px 13px 15px;margin:16px 0;
  font-size:14px;border:1px solid var(--rule);border-left:4px solid var(--ink3);
  color:var(--ink2);max-width:74ch}
.banner strong{color:var(--ink)}
.banner.warn{border-left-color:var(--warn);background:var(--warn-bg)}
.banner.bad{border-left-color:var(--bad);background:var(--bad-bg)}
.banner.good{border-left-color:var(--good)}

.scroll{overflow-x:auto;border:1px solid var(--rule);border-radius:6px;
  margin:10px 0;background:var(--card)}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:right;padding:7px 11px;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
thead th{position:sticky;top:0;z-index:1;background:var(--card2);
  font:600 10.5px/1.4 var(--sans);color:var(--ink3);
  text-transform:uppercase;letter-spacing:.07em;
  border-bottom:1px solid var(--rule);white-space:normal}
tbody td{border-bottom:1px solid var(--gridline);color:var(--ink2)}
tbody tr:last-child td{border-bottom:0}
tbody tr:hover td{background:var(--card2)}
td:first-child{color:var(--ink);font-family:var(--mono);font-size:12.5px}
td strong{font-weight:600;color:var(--ink)}

.legend{display:flex;flex-wrap:wrap;gap:8px 16px;margin:10px 0 2px;
  font-size:12.5px;color:var(--ink2)}
.legend .li{display:inline-flex;align-items:center}
.legend i{display:inline-block;width:9px;height:9px;border-radius:2px;
  margin-right:6px;flex:none}

.chart{display:block;margin:6px 0 2px}
.chart text{font-family:var(--sans);font-size:11px}
.chart .lbl{fill:var(--ink2);font-family:var(--mono);font-size:11px}
.chart .tick{fill:var(--ink3);font-size:10px}
.chart .val{fill:var(--ink3);font-size:10.5px;font-family:var(--mono)}
.chart .seg{fill:#fff;font-size:11px;font-weight:600;paint-order:stroke;
  stroke:rgba(0,0,0,.3);stroke-width:2.5px}
.chart .absent{fill:var(--ink3);font-size:10px;font-style:italic}
.chart .cap{fill:var(--ink3);font-size:11px}
.chart .grid{stroke:var(--gridline);stroke-width:1}
.chart .free{fill:var(--free)}

.note{font-size:12.5px;color:var(--ink3);margin:8px 0 0;max-width:76ch}
details{margin:12px 0;border:1px solid var(--rule);border-radius:6px;
  padding:11px 15px;background:var(--card)}
summary{cursor:pointer;color:var(--ink2);font-size:13.5px}
summary:focus-visible,a:focus-visible{outline:2px solid var(--accent);
  outline-offset:2px;border-radius:2px}
code{font-family:var(--mono);font-size:12.5px;color:var(--ink)}
@media (prefers-reduced-motion:reduce){*{animation:none!important;
  transition:none!important}}
@media (max-width:640px){.wrap{padding:26px 14px 64px}h1{font-size:25px}}
"""


def _fmt_b(n):
    if n is None:
        return "-"
    return f"{n:,} B" if n < 10240 else f"{n / 1024:,.1f} KiB"


def _table(headers, rows, cls=""):
    h = "".join(f"<th>{esc(x)}</th>" for x in headers)
    body = []
    for r in rows:
        tds = []
        for i, c in enumerate(r):
            k = ' class="op"' if i == 0 else ""
            tds.append(f"<td{k}>{c if isinstance(c, str) else esc(c)}</td>")
        body.append("<tr>" + "".join(tds) + "</tr>")
    return (f'<div class="scroll"><table class="{cls}"><thead><tr>{h}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def _share_rows(art, top=6):
    """Top-N ops by share plus an explicit Other, never a cycled 9th hue."""
    ops = [o for o in art["ops"] if (o.get("share_of_forward_pct") or 0) > 0]
    ops.sort(key=lambda o: -(o["share_of_forward_pct"] or 0))
    rows = [(o["name"], o["share_of_forward_pct"], i) for i, o in enumerate(ops[:top])]
    rest = sum(o["share_of_forward_pct"] or 0 for o in ops[top:])
    if rest > 0:
        rows.append((f"other ({len(ops) - top})", rest, tp_svg.OTHER_SLOT))
    return rows


def _pct(v, digits=2):
    return f"{v:.{digits}f}%" if v is not None else "-"


def _us(v):
    return tp_svg._fmt_us(v) if v is not None else "-"


def _zone_tree(art):
    """The nesting, drawn as an indented list.

    Included because the tree is the one thing a reader has to accept on trust
    to read every other table on the page, and it costs four lines to show it.
    """
    kids = {}
    for o in art["ops"]:
        kids.setdefault(o["parent"], []).append(o)
    rows = []

    def walk(parent, depth):
        for o in sorted(kids.get(parent, []),
                        key=lambda x: -x["exclusive_us_per_forward"]):
            rows.append([
                ("&nbsp;" * 4 * depth) + f"<code>{esc(o['name'])}</code>",
                f'{o["calls_per_forward"]:,.0f}',
                _us(o["inclusive_us_per_forward"]),
                _us(o["exclusive_us_per_forward"]),
                _pct(o["share_of_forward_pct"]),
            ])
            walk(o["name"], depth + 1)

    walk(None, 0)
    return _table(["zone (indented = measured inside its parent)", "calls/fwd",
                   "inclusive", "exclusive", "% of forward"], rows)


def render(cmp: dict, base: dict, opt: dict) -> str:
    S = cmp["shape"]
    shape = f"B=1 S={S.get('S')} D={S.get('D')} H={S.get('H')} F={S.get('F')} L={S.get('L')}"
    b_dev, o_dev = cmp["baseline"]["device"], cmp["optimised"]["device"]
    host = "host" in (str(b_dev) + str(o_dev))
    sides = (("baseline", base), ("optimised", opt))

    out = [f"<title>tinyprof Case 2 Profile</title><style>{CSS}</style>", '<div class="wrap">']
    out.append("<h1>Case 2 &mdash; baseline vs optimised</h1>")
    out.append(f'<p class="sub">{esc(shape)}, causal &middot; '
               f'<code>{esc(cmp["baseline"]["tag"])}</code> &rarr; '
               f'<code>{esc(cmp["optimised"]["tag"])}</code> &middot; '
               f'captured on {esc(b_dev)} &middot; profiled with tinyprof at '
               f'{esc(base.get("tick_resolution_ns"))}&nbsp;ns zone resolution</p>')

    titles = ["Accuracy", "Method &amp; overhead", "Zone tree", "Where the time goes",
              "Absolute time", "Every op", "Rankings", "Memory", "Traffic",
              "Roofline", "Kernel bench", "Provenance"]
    out.append('<nav class="toc">' + " ".join(
        f'<a href="#s{i}">{i}. {t}</a>' for i, t in enumerate(titles, start=1)) + "</nav>")

    if host:
        out.append('<div class="banner warn"><strong>Host capture, not an ESP32 '
                   'measurement.</strong> These timings come from an x86 host with a '
                   'hardware FPU and megabytes of cache. The optimisation being measured '
                   'targets a 160&nbsp;MHz RISC-V core with <em>no</em> FPU, so the host '
                   'ratios are not the device ratios &mdash; and can invert, since the '
                   'fixed-point path trades float work for integer work that a host does '
                   'not need. Structure, call counts, memory and traffic transfer; times '
                   'do not. Re-run against hardware for a quotable speedup.</div>')

    macs = (opt.get("roofline") or {}).get("flops", {}).get("total")
    out.append('<div class="hero">')
    hero = [("baseline", f'{cmp["baseline"]["s_per_forward"]:.4f} s'),
            ("optimised", f'{cmp["optimised"]["s_per_forward"]:.4f} s'),
            ("speedup", f'{cmp["speedup"]:.2f}x'),
            ("time removed", _us(cmp["us_removed_per_forward"]))]
    if macs:
        hero.append(("model work", f"{macs / 1e6:.1f} MFLOP"))
    for k, v in hero:
        out.append(f'<div><div class="k">{esc(k)}</div><div class="n">{esc(v)}</div></div>')
    out.append("</div>")

    for w in cmp.get("warnings", []):
        out.append(f'<div class="banner bad">{esc(w)}</div>')

    # ---- 1. accuracy ---------------------------------------------------
    out.append('<h2 id="s1">1. Accuracy gate</h2>')
    out.append('<p>Applied per output element: <code>abs_err &lt;= 0.002 OR '
               'rel_err &lt;= 0.02</code>, against the torch reference for the same seed. '
               'This section precedes the timings on purpose &mdash; a speedup measured at '
               'a different shape, or with a failing gate, is not a speedup.</p>')
    rows = []
    for side, art in sides:
        for g in (art.get("accuracy") or {}).get("seeds", []):
            ok = g.get("fails") == 0
            colour = "good" if ok else "bad"
            rows.append([side, g.get("seed"),
                         f'<span style="color:var(--{colour})">'
                         f'{"PASS" if ok else "FAIL"}</span>',
                         g.get("fails"), f'{g.get("max_abs", 0):.4e}',
                         f'{g.get("max_rel", 0):.4e}' if g.get("max_rel") else "-"])
    out.append(_table(["build", "seed", "result", "failed elements",
                       "max abs error", "max rel error"], rows))
    out.append('<p class="note">A large max relative error alongside zero failures is '
               'expected, not a defect: the gate is a disjunction, so an element whose '
               'reference value sits near zero passes on the absolute term regardless of '
               'its ratio. Reporting both is what makes that visible instead of alarming.</p>')

    # ---- 2. method -----------------------------------------------------
    ovh = cmp["overhead"] or {}
    out.append('<h2 id="s2">2. What is being timed, and what the timing costs</h2>')
    out.append('<div class="card"><p style="margin-top:0">One forward of the complete '
               'four-layer body. A warm-up forward runs first and is excluded. Host '
               'transfer of the 64&nbsp;KB input and output is excluded on both sides.</p>'
               f'<p>Zones are timed with the free-running cycle counter at '
               f'{esc(base.get("tick_resolution_ns"))}&nbsp;ns, not the microsecond timer. '
               'That is not a refinement: at 1&nbsp;us resolution the residual adds round '
               'to zero, and the implementation this replaced then dropped them with an '
               '<code>if (d &gt; 0)</code> guard &mdash; so <code>res1</code> was absent '
               'from the profile entirely and <code>res2</code> reported 2 calls out of 12. '
               'Zones are now counted whether or not they register time, and any zone '
               'averaging under 20 ticks is flagged <em>resolution-limited</em> rather than '
               'reported as though exact.</p>'
               '<p>All times below are <strong>exclusive</strong> unless a column says '
               "otherwise: a zone's own time with nested zones subtracted. The tree is in "
               'section&nbsp;3.</p></div>')
    rows = []
    for side, art in sides:
        o = ovh.get(side) or {}
        rows.append([side, f'{o.get("ns_per_probe", 0):.1f} ns',
                     f'{o.get("probes_per_forward", 0):,.0f}',
                     _us(o.get("estimated_us_per_forward")),
                     _pct(o.get("estimated_pct_of_forward"), 3)])
    out.append(_table(["build", "cost of one clock read", "clock reads/forward",
                       "overhead/forward", "% of forward"], rows))
    out.append('<p class="note">Measured on the device at dump time by timing 4,096 '
               'back-to-back clock reads &mdash; not assumed. It is reported rather than '
               'subtracted from the headline, because an overhead that differs between two '
               'builds is exactly what would inflate a speedup if it were folded in '
               'silently. A per-op first-order correction sits beside every raw figure in '
               'section&nbsp;6 and in the embedded JSON.</p>')
    for side, o in ovh.items():
        if (o or {}).get("warning"):
            out.append(f'<div class="banner warn"><strong>{esc(side)}:</strong> '
                       f'{esc(o["warning"])}</div>')

    # ---- 3. zone tree ---------------------------------------------------
    out.append('<h2 id="s3">3. Zone tree</h2>')
    out.append('<p>The nesting is declared by the firmware, not inferred host-side from '
               'the names. It matters because inclusive times sum past 100% of the '
               'forward: <code>quant</code> is measured inside <code>qkv</code>, '
               '<code>gelu</code> inside <code>f2</code>, and the attention phases inside '
               '<code>attn</code>. Ranking on inclusive time would double-count every '
               'parent.</p>')
    for side, art in sides:
        tot = sum(o["exclusive_us_per_forward"] for o in art["ops"])
        wall = art["wall"]["us_per_forward"]
        out.append(f'<h3>{esc(side)} &mdash; <code>{esc(art["tag"])}</code></h3>')
        out.append(_zone_tree(art))
        out.append(f'<p class="note">Exclusive times sum to {_us(tot)} against a measured '
                   f'forward of {_us(wall)} &mdash; {100.0 * tot / wall:.1f}%. That figure '
                   f'is the check that the tree above is correct; a wrong parent shows up '
                   f'here immediately as a total well over or under 100%.</p>')

    # ---- 4. share shift --------------------------------------------------
    out.append('<h2 id="s4">4. Where the time goes</h2>')
    out.append('<p>Each bar is one forward, normalised to 100%. The question an '
               'optimisation report has to answer is not which op is fastest but which op '
               '<em>is</em> the forward &mdash; and whether that changed.</p>')
    for title, art in sides:
        rws = _share_rows(art)
        out.append(f'<h3>{esc(title)} &mdash; <code>{esc(art["tag"])}</code></h3>')
        out.append(tp_svg.stacked_share(rws))
        out.append(tp_svg.legend([(r[0], r[2]) for r in rws]))
    shifts = sorted([r for r in cmp["ops"] if r.get("share_shift_pct_points") is not None],
                    key=lambda r: r["share_shift_pct_points"])
    if shifts:
        picked = shifts[:3] + [r for r in shifts[-3:] if r not in shifts[:3]]
        out.append('<p class="note">Largest share shifts: ' + ", ".join(
            f'<code>{esc(r["op"])}</code> {r["share_shift_pct_points"]:+.1f} pts'
            for r in picked) + '.</p>')

    # ---- 5. absolute ------------------------------------------------------
    ops = [r for r in cmp["ops"] if r["baseline_us"] or r["optimised_us"]]
    ops.sort(key=lambda r: -(r["baseline_us"] or 0))
    out.append('<h2 id="s5">5. Absolute time per op</h2>')
    out.append('<p>Log scale &mdash; the values span five orders of magnitude, and on a '
               'linear axis every op but the largest is an invisible sliver. An op present '
               'in only one build is marked <em>absent</em>, never drawn as zero: '
               '&ldquo;100x faster&rdquo; and &ldquo;no longer a zone&rdquo; must not look '
               'the same.</p>')
    out.append(tp_svg.legend([("baseline", 0), ("optimised", 1)]))
    out.append(tp_svg.grouped_log_bars(
        [r["op"] for r in ops],
        [("baseline", 0, [r["baseline_us"] for r in ops]),
         ("optimised", 1, [r["optimised_us"] for r in ops])]))

    # ---- 6. full per-op table ----------------------------------------------
    out.append('<h2 id="s6">6. Every op, both builds</h2>')
    out.append('<p>The complete table behind every chart above. Nothing is elided.</p>')
    for side, art in sides:
        out.append(f'<h3>{esc(side)} &mdash; <code>{esc(art["tag"])}</code></h3>')
        rows = []
        tr = {r["op"]: r for r in ((art.get("traffic") or {}).get("by_op") or [])}
        for o in sorted(art["ops"], key=lambda x: -x["exclusive_us_per_forward"]):
            t = tr.get(o["name"], {})
            rows.append([
                o["name"], o["parent"] or "&mdash;",
                f'{o["calls_per_forward"]:,.0f}',
                _us(o["inclusive_us_per_forward"]),
                _us(o["exclusive_us_per_forward"]),
                _us(o["exclusive_us_per_forward_overhead_corrected"]),
                f'{o["avg_us_per_call"]:.4f}',
                _pct(o["share_of_forward_pct"]),
                f'{t.get("flash_xip_bytes_per_forward", 0):,}' if t else "-",
                f'{t["achieved_mb_s"]:.1f}' if t.get("achieved_mb_s") else "-",
                "resolution-limited" if o["resolution_limited"] else "",
            ])
        out.append(_table(["op", "parent", "calls/fwd", "inclusive", "exclusive",
                           "excl. corrected", "avg us/call", "% of fwd",
                           "flash B/fwd", "MB/s", "note"], rows))

    # ---- 7. rankings --------------------------------------------------------
    out.append('<h2 id="s7">7. Rankings</h2>')
    out.append("<h3>Top 10 by time removed</h3>")
    out.append('<p class="note">Ranked by absolute microseconds taken out of the forward, '
               'not by ratio &mdash; a 100x speedup on a 0.3&nbsp;us op buys nothing.</p>')
    rows = []
    for r in sorted([x for x in cmp["ops"] if x["us_saved"] is not None],
                    key=lambda x: -x["us_saved"])[:10]:
        rows.append([r["op"], _us(r["baseline_us"]), _us(r["optimised_us"]),
                     f'{r["speedup"]:.1f}x' if r["speedup"] else "-",
                     _us(r["us_saved"]),
                     f'{r["share_shift_pct_points"]:+.1f}'
                     if r.get("share_shift_pct_points") is not None else "-"])
    out.append(_table(["op", "baseline", "optimised", "speedup", "time removed",
                       "share shift (pts)"], rows))

    for side, art in sides:
        out.append(f"<h3>Top 10 by call count &mdash; {esc(side)}</h3>")
        rows = []
        for o in sorted(art["ops"], key=lambda x: -x["calls_per_forward"])[:10]:
            rows.append([o["name"], f'{o["calls_per_forward"]:,.0f}',
                         f'{o["avg_us_per_call"]:.4f}',
                         _us(o["exclusive_us_per_forward"]),
                         _pct(o["share_of_forward_pct"]),
                         "resolution-limited" if o["resolution_limited"] else ""])
        out.append(_table(["op", "calls/forward", "avg us/call", "exclusive",
                           "% of forward", "note"], rows))
    incomparable = [r["op"] for r in cmp["ops"] if r.get("calls_comparable") is False]
    if incomparable:
        out.append('<p class="note">Call counts are <em>not</em> comparable for '
                   + ", ".join(f"<code>{esc(o)}</code>" for o in incomparable)
                   + ". That is a structural difference in where the two firmwares place "
                     "their zone brackets, not a regression: the baseline's attention is "
                     "one fused loop and cannot be split per query row without changing "
                     "the arithmetic being measured.</p>")

    # ---- 8. memory ------------------------------------------------------------
    out.append('<h2 id="s8">8. Memory</h2>')
    mb, mo = cmp["memory"]["baseline"], cmp["memory"]["optimised"]
    if mb.get("dram_capacity") or mo.get("dram_capacity"):
        out.append('<p>Static usage is read from the linked ELF with '
                   '<code>riscv32-esp-elf-size</code>, and the DRAM capacity from '
                   '<code>dram0_0_seg</code> in the link map rather than hard-coded &mdash; '
                   'on this chip the usable figure is not a round number and moves with '
                   'the IDF version. <code>.dram0.dummy</code> is excluded from '
                   '&ldquo;used&rdquo;: it mirrors the flash rodata reservation, and '
                   'counting it would overstate usage by tens of KB.</p>')
        out.append(tp_svg.legend([("model workspace", 0), ("other static", 2),
                                  ("initialised data", 3)]))
        for side, m in (("baseline", mb), ("optimised", mo)):
            if not m.get("dram_capacity"):
                continue
            out.append(f"<h3>{esc(side)} DRAM</h3>")
            out.append(tp_svg.capacity_bar(
                [("model workspace", m["arena_census_total"] or 0, 0),
                 ("other static",
                  max(0, (m["dram_bss"] or 0) - (m["arena_census_total"] or 0)), 2),
                 ("initialised data",
                  max(0, (m["dram_used"] or 0) - (m["dram_bss"] or 0)), 3)],
                m["dram_capacity"]))
        keys = ("dram_bss", "dram_used", "dram_free", "dram_capacity",
                "flash_used", "arena_census_total")
        rows = []
        for k in keys:
            bv, ov = mb.get(k), mo.get(k)
            rows.append([k,
                         f"{bv:,}" if bv is not None else "-",
                         f"{ov:,}" if ov is not None else "-",
                         f"{ov - bv:+,}" if (bv is not None and ov is not None) else "-"])
        out.append(_table(["section (bytes)", "baseline", "optimised", "delta"], rows))
    else:
        out.append('<p class="note">No ELF was supplied for either capture, so static '
                   'memory is unavailable. The arena census below still comes from the '
                   'firmware itself.</p>')

    out.append("<h3>Declared model workspace</h3>")
    out.append('<p class="note">Compiled into flash as a <code>static const</code> table, '
               'so the census costs no DRAM and cannot drift from the declarations it '
               'mirrors.</p>')
    cb = {c["name"]: c for c in base["memory"]["arena_census"]}
    co = {c["name"]: c for c in opt["memory"]["arena_census"]}
    rows = []
    for n in sorted(set(cb) | set(co)):
        rows.append([n, (cb.get(n) or co.get(n))["role"],
                     _fmt_b(cb[n]["bytes"]) if n in cb else "absent",
                     _fmt_b(co[n]["bytes"]) if n in co else "absent"])
    rows.append(["<strong>total</strong>", "",
                 f'<strong>{_fmt_b(mb["arena_census_total"])}</strong>',
                 f'<strong>{_fmt_b(mo["arena_census_total"])}</strong>'])
    out.append(_table(["buffer", "role", "baseline", "optimised"], rows))
    xc = (opt.get("memory") or {}).get("census_vs_elf")
    if xc:
        out.append(f'<p class="note">Cross-check: the firmware declares '
                   f'{xc["census_total"]:,}&nbsp;B of workspace, the ELF reports '
                   f'{xc["elf_dram_bss"]:,}&nbsp;B of <code>.bss</code>, and the '
                   f'{xc["unattributed_bytes"]:,}&nbsp;B difference is framework and '
                   f'driver static state. Two independent sources agreeing is what makes '
                   f'these numbers trustworthy &mdash; and this exact check is what found '
                   f'the 32&nbsp;KB <code>a16</code> scratch buffer missing from the '
                   f'census.</p>')

    st = (opt.get("memory") or {}).get("static") or {}
    if st.get("top_static"):
        out.append("<h3>Largest static objects, optimised build</h3>")
        out.append('<p class="note">From <code>nm --size-sort</code>, so a memory '
                   'regression arrives with a symbol name attached.</p>')
        out.append(_table(["symbol", "section", "bytes"],
                          [[r["symbol"], r["kind"], f'{r["bytes"]:,}']
                           for r in st["top_static"][:12]]))
    fsec = st.get("flash") or {}
    if fsec.get("embedded_weights"):
        out.append("<h3>Flash</h3>")
        frows = [["code (.flash.text)", f'{fsec["text"]:,}'],
                 ["read-only data (.flash.rodata)", f'{fsec["rodata"]:,}']]
        frows += [[f"embedded {k}", f"{v:,}"] for k, v in fsec["embedded_weights"].items()]
        frows += [["<strong>total used</strong>", f'<strong>{fsec["used"]:,}</strong>'],
                  ["code and data excluding weights",
                   f'{fsec["code_and_data_excl_weights"]:,}']]
        out.append(_table(["region", "bytes"], frows))
        out.append('<p class="note">Blob sizes come from the linker-generated '
                   '<code>_binary_*_start/_end</code> symbols, which is how 2.4&nbsp;MB of '
                   'flash gets attributed to weights rather than to code.</p>')

    any_runtime = False
    for side, art in sides:
        rt = (art.get("memory") or {}).get("runtime") or {}
        if not rt.get("measured"):
            continue
        any_runtime = True
        out.append(f"<h3>Runtime &mdash; {esc(side)}</h3>")
        out.append(_table(["metric", "value"], [
            ["free heap", _fmt_b(rt.get("heap_free"))],
            ["minimum free heap since boot", _fmt_b(rt.get("heap_min_free_since_boot"))],
            ["largest free block", _fmt_b(rt.get("heap_largest_free_block"))],
            ["heap unusable as one block (fragmentation)",
             _fmt_b(rt.get("heap_fragmentation_bytes"))],
            [f'stack high-water mark ({esc(rt.get("stack_task"))})',
             _fmt_b(rt.get("stack_hwm_bytes"))]]))
    if not any_runtime:
        out.append('<p class="note">Runtime heap and stack watermarks are device-only: a '
                   'host capture has no FreeRTOS task or ESP heap to interrogate. The '
                   'minimum-free-heap watermark is the useful one on device, because it '
                   'survives the fact that the dump happens after the forward has already '
                   'released whatever it used.</p>')

    # ---- 9. traffic ------------------------------------------------------------
    out.append('<h2 id="s9">9. Memory traffic</h2>')
    out.append('<p>Weights live in flash and stream through the 16&nbsp;KB XIP cache; '
               'activations live in SRAM. The two are reported separately because on this '
               'board they are different resources with different costs. Bytes are a '
               'declared per-call model multiplied by the <strong>measured</strong> call '
               'count &mdash; never an assumed one, which is the failure mode this '
               'replaces.</p>')
    rows = []
    for side, art in sides:
        t = art.get("traffic") or {}
        rows.append([side, f'{t.get("flash_xip_bytes_per_forward", 0):,}',
                     f'{t.get("sram_bytes_per_forward", 0):,}',
                     "yes" if t.get("validated")
                     else '<span style="color:var(--bad)">NO</span>'])
    out.append(_table(["build", "flash B/forward", "SRAM B/forward",
                       "call counts validated"], rows))
    tr = opt.get("traffic")
    if tr:
        out.append("<h3>By op, optimised build</h3>")
        rows = []
        for r in tr["by_op"]:
            rows.append([r["op"], f'{r["calls_per_forward"]:,.0f}',
                         f'{r["flash_xip_bytes_per_forward"]:,}',
                         f'{r["sram_bytes_per_forward"]:,}',
                         f'{r["achieved_mb_s"]:.1f}' if r["achieved_mb_s"] else "-",
                         r["confidence"],
                         "yes" if r["call_count_validated"]
                         else '<span style="color:var(--bad)">NO</span>'])
        out.append(_table(["op", "calls/fwd", "flash B/fwd", "SRAM B/fwd",
                           "achieved MB/s", "confidence", "count validated"], rows))
        out.append("<p class=\"note\"><code>analytic_exact</code> means every call moves "
                   "the same bytes. <code>analytic_averaged</code> means bytes vary with "
                   "the causal row index and only the per-forward total is exact &mdash; "
                   "the closed form is below. <code>parent_only</code> means the traffic "
                   "is attributed to the zone's children, so counting it here would "
                   "double it.</p>")
        out.append('<details open><summary><strong>How each figure is derived</strong> '
                   '&mdash; the point of keeping the model in the artifact rather than in '
                   "someone's notes</summary>")
        for r in tr["by_op"]:
            if r.get("why"):
                extra = (f' <em>Closed form: {esc(r["closed_form"])}.</em>'
                         if r.get("closed_form") else "")
                out.append(f'<p class="note"><code>{esc(r["op"])}</code> &mdash; '
                           f'{esc(r["why"])}{extra}</p>')
        out.append("</details>")

    # ---- 10. roofline -----------------------------------------------------------
    rf = opt.get("roofline") or {}
    if rf and not rf.get("error"):
        out.append('<h2 id="s10">10. Roofline and utilisation</h2>')
        if rf.get("applicable") is False:
            out.append(f'<div class="banner warn">{esc(rf["not_applicable_reason"])}</div>')
        fl = rf.get("flops", {})
        if fl:
            out.append("<h3>Model work per forward</h3>")
            frows = [[k, f"{fl[k]:,.0f}", _pct(100.0 * fl[k] / fl["total"])]
                     for k in ("gemm", "attn", "ln", "gelu", "res")]
            frows.append(["<strong>total</strong>",
                          f'<strong>{fl["total"]:,.0f}</strong>', "100.00%"])
            out.append(_table(["class", "FLOP", "share"], frows))
        ai = rf.get("arithmetic_intensity_flop_per_byte", {})
        aim = rf.get("arithmetic_intensity_measured_traffic")
        out.append("<h3>Arithmetic intensity &mdash; two independent models</h3>")
        rows = [["score.py operand model (total)", f'{ai.get("total", 0):.3f}',
                 "4 B/MAC, no operand reuse assumed"],
                ["score.py operand model (GEMM)", f'{ai.get("gemm", 0):.3f}',
                 "1 int16 A-read + 1 int16 W-read per MAC"],
                ["score.py operand model (float ops)", f'{ai.get("float_ops", 0):.3f}',
                 "attention + LayerNorm + GELU + residual operands"]]
        if aim:
            rows.append(["tinyprof measured-traffic model", f'{aim["flop_per_byte"]:.3f}',
                         f'{aim["bytes_per_forward"]:,} B/forward from measured call counts'])
        out.append(_table(["model", "FLOP/byte", "basis"], rows))
        if aim and ai.get("total"):
            ratio = aim["flop_per_byte"] / ai["total"]
            out.append(f'<p class="note">The two models disagree by about '
                       f'{ratio:.0f}x, and that gap is reported rather than resolved. '
                       f'score.py deliberately assumes an unblocked kernel with no operand '
                       f'reuse, which makes its intensity a lower bound; tinyprof counts '
                       f'the weight bytes the kernels actually stream, which credits the '
                       f'reuse the tiled cores get. The truth is between them, and showing '
                       f'either number alone would hide that.</p>')
        if rf.get("applicable"):
            out.append("<h3>Utilisation</h3>")
            ex = rf.get("exscore", {})
            out.append(_table(["metric", "value"], [
                ["achieved", f'{rf["achieved_mflop_s"]:.1f} MFLOP/s'],
                ["MFU against the realistic mixed peak", _pct(100.0 * rf["mfu_mix"])],
                ["MFU against the raw int-MAC ceiling", _pct(100.0 * rf["mfu_int"], 3)],
                ["ExScore at 320 MB/s SRAM", f'{ex.get("at_bw_320MB_s")}'],
                ["ExScore at 640 MB/s SRAM", f'{ex.get("at_bw_640MB_s")}'],
                ["roofline-bound time", f'{ex.get("roofline_seconds", 0):.4f} s']]))
        pk = rf.get("peaks", {})
        if pk:
            sf = pk.get("softfloat_range", [0, 0])
            bw = pk.get("sram_bw_range_B_s", [0, 0])
            out.append("<h3>Peak constants</h3>")
            out.append(_table(["constant", "value", "note"], [
                ["int16 MAC ceiling", f'{pk.get("int_mac_flop_s", 0) / 1e6:.0f} MFLOP/s',
                 "160 MMAC/s: one 32-bit multiply per cycle, no SIMD, no hardware MAC"],
                ["soft-float peak", f'{pk.get("softfloat_flop_s", 0) / 1e6:.1f} MFLOP/s',
                 f'range {sf[0] / 1e6:.1f}-{sf[1] / 1e6:.1f}; the C3 has no FPU'],
                ["sustained SRAM bandwidth",
                 f'{bw[0] / 1e6:.0f}-{bw[1] / 1e6:.0f} MB/s', "estimate"]]))
        out.append(f'<p class="note">{esc(rf.get("method", ""))}</p>')

    # ---- 11. kernel bench ---------------------------------------------------------
    kb = opt.get("kbench") or []
    cy = opt.get("cycbench") or []
    if kb or cy:
        out.append('<h2 id="s11">11. Kernel microbenchmark</h2>')
        out.append('<p>The pre-existing <code>kernels.c</code> instrumentation, carried '
                   'through unchanged so nothing that already reads its output breaks. '
                   'The cycles-per-call figures are an independent read on whether the '
                   'GEMM cores are compute-bound or stalled on the flash cache.</p>')
        if kb:
            out.append(_table(["slot", "calls", "avg us", "total ms"],
                              [[f'KB{r["slot"]}', f'{r["n"]:,}', f'{r["avg_us"]:.1f}',
                                f'{r["tot_ms"]:.1f}'] for r in kb]))
        if cy:
            out.append(_table(["core", "calls", "avg cycles", "avg us"],
                              [[f'core{r["core"]}', f'{r["n"]:,}', f'{r["avg_cyc"]:,.1f}',
                                f'{r["avg_us"]:.1f}'] for r in cy]))

    # ---- 12. provenance ------------------------------------------------------------
    out.append('<h2 id="s12">12. Provenance and method</h2>')
    rows = []
    for side, art in sides:
        p = art.get("provenance") or {}
        rows.append([side, art.get("tag"), art.get("device"),
                     f'{art["wall"]["s_per_forward"]:.6f} s', art["reps"],
                     art["wall"]["source"], (p.get("git_commit") or "-")[:10],
                     "dirty" if p.get("git_dirty") else ("clean" if p else "-"),
                     p.get("captured_utc", "-")])
    out.append(_table(["build", "tag", "device", "s/forward", "reps", "wall source",
                       "git", "tree", "captured (UTC)"], rows))
    rows = []
    for side, art in sides:
        p = art.get("provenance") or {}
        rows.append([side, (p.get("weights_sha256") or "-")[:16],
                     (p.get("elf_sha256") or "-")[:16], p.get("port", "-"),
                     p.get("env", "-")])
    out.append(_table(["build", "weights sha256", "ELF sha256", "port", "env"], rows))
    out.append(f'<p class="note">{esc(cmp.get("note", ""))}</p>')
    out.append('<p class="note">Every derived value in the embedded JSON carries either '
               '<code>"measured": true</code> or a <code>"method"</code> string, so the '
               'measured-versus-modelled split is part of the data rather than something '
               'this page asserts. The complete artifacts &mdash; both captures and the '
               'comparison &mdash; are embedded below and recoverable from this file '
               'alone: <code>JSON.parse(document.getElementById("tinyprof-data")'
               '.textContent)</code>.</p>')
    unparsed = [l for a in (base, opt) for l in a.get("unparsed_lines", [])]
    if unparsed:
        out.append('<details><summary>Lines the parser did not recognise</summary>'
                   + "".join(f'<p class="note"><code>{esc(l)}</code></p>' for l in unparsed)
                   + "</details>")

    payload = json.dumps({"comparison": cmp, "baseline": base, "optimised": opt})
    out.append('<script type="application/json" id="tinyprof-data">'
               + payload.replace("</", "<\\/") + "</script>")
    out.append("</div>")
    return "\n".join(out)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("comparison")
    ap.add_argument("baseline")
    ap.add_argument("optimised")
    ap.add_argument("-o", "--output", required=True)
    a = ap.parse_args()
    cmp = json.loads(pathlib.Path(a.comparison).read_text())
    base = json.loads(pathlib.Path(a.baseline).read_text())
    opt = json.loads(pathlib.Path(a.optimised).read_text())
    pathlib.Path(a.output).write_text(render(cmp, base, opt) + "\n")
    print(f"tinyprof: wrote {a.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
