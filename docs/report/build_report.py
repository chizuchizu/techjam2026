#!/usr/bin/env python3
"""build_report.py - generate docs/report/index.html.

The report is a single self-contained page: the CSS shell below, then a body
built section by section. Chart geometry is *computed* rather than hand-typed,
so correcting a measurement means editing one number in one list and re-running
this script - not nudging SVG coordinates until the picture looks right.

    python3 docs/report/build_report.py

Fonts come from Google Fonts (the only external reference on the page); the
charts are inline SVG with no library.
"""
import math
import pathlib

OUT = pathlib.Path(__file__).resolve().parent / "index.html"

SHELL = r"""<title>Transformers on a $5 Microcontroller</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
/* ─────────────────────────────────────────────────────────────
   Palette. The ground is a cool graphite-green paper — the colour
   of a datasheet page, not cream — and the single accent is a burnt
   signal orange drawn from the same family as the chart series, so
   page and charts read as one system. Chart series use the validated
   categorical palette and are not re-picked by eye.
   ───────────────────────────────────────────────────────────── */
:root{
  color-scheme:light;
  --paper:#f3f6f5; --panel:#ffffff; --panel-2:#e9efed;
  --ink:#0d1413; --ink-2:#4b5856; --ink-3:#788583;
  --rule:#dbe3e1; --grid:#e7eeec;
  --signal:#c2521f; --signal-soft:#fbeee7;
  --good:#0d7a4e; --warn:#8a5a00; --bad:#b3261e;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100;
  --s5:#e87ba4; --s6:#008300; --s7:#4a3aa7; --s8:#b4bcbb;
}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){
  color-scheme:dark;
  --paper:#0c1211; --panel:#141b1a; --panel-2:#1b2322;
  --ink:#eef3f2; --ink-2:#adbab8; --ink-3:#7e8b89;
  --rule:#28322f; --grid:#212a28;
  --signal:#e8763f; --signal-soft:#241611;
  --good:#4ac088; --warn:#e0a83c; --bad:#f2685f;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500;
  --s5:#d55181; --s6:#008300; --s7:#9085e9; --s8:#66706f;
}}
:root[data-theme=dark]{
  color-scheme:dark;
  --paper:#0c1211; --panel:#141b1a; --panel-2:#1b2322;
  --ink:#eef3f2; --ink-2:#adbab8; --ink-3:#7e8b89;
  --rule:#28322f; --grid:#212a28;
  --signal:#e8763f; --signal-soft:#241611;
  --good:#4ac088; --warn:#e0a83c; --bad:#f2685f;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500;
  --s5:#d55181; --s6:#008300; --s7:#9085e9; --s8:#66706f;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%;scroll-behavior:smooth}
@media (prefers-reduced-motion:reduce){html{scroll-behavior:auto}
  *{animation:none!important;transition:none!important}}
body{margin:0;background:var(--paper);color:var(--ink);
  font:400 17px/1.68 "Source Serif 4",Georgia,"Times New Roman",serif;
  font-variant-numeric:tabular-nums}

/* Layout: a narrow reading column, with evidence allowed to break wider. */
.wrap{max-width:1080px;margin:0 auto;padding:0 24px 120px}
.col{max-width:660px;margin:0 auto}
.wide{max-width:980px;margin:28px auto}
h1,h2,h3,h4,.mono,.k,th,.tag,.toc,.num{font-family:Archivo,"Helvetica Neue",Arial,sans-serif}
.mono,code,td.op,.num{font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace}

/* Scroll progress — the one flourish. */
#prog{position:fixed;top:0;left:0;height:2px;width:0;background:var(--signal);
  z-index:99}

header.hero{padding:88px 0 40px;border-bottom:1px solid var(--rule);margin-bottom:8px}
.eyebrow{font:600 11.5px/1 Archivo,sans-serif;letter-spacing:.16em;
  text-transform:uppercase;color:var(--signal);margin-bottom:22px}
h1{font:700 clamp(38px,6.2vw,62px)/1.02 Archivo,sans-serif;letter-spacing:-.032em;
  margin:0 0 20px;text-wrap:balance;max-width:16ch}
.standfirst{font-size:20.5px;line-height:1.55;color:var(--ink-2);max-width:60ch;margin:0}
.byline{margin-top:26px;font:500 12.5px/1.6 Archivo,sans-serif;color:var(--ink-3);
  letter-spacing:.02em}

h2{font:600 12px/1 Archivo,sans-serif;letter-spacing:.15em;text-transform:uppercase;
  color:var(--signal);margin:0 0 14px}
h3{font:600 clamp(24px,3.4vw,32px)/1.15 Archivo,sans-serif;letter-spacing:-.021em;
  margin:0 0 16px;color:var(--ink);text-wrap:balance}
h4{font:600 17px/1.3 Archivo,sans-serif;letter-spacing:-.01em;margin:30px 0 8px}
section{padding:64px 0;border-bottom:1px solid var(--rule)}
section:last-of-type{border-bottom:0}
.sec-no{font:600 11.5px/1 "IBM Plex Mono",monospace;color:var(--ink-3);
  letter-spacing:.1em;margin-bottom:10px}
p{margin:0 0 18px;color:var(--ink-2)}
p strong,li strong{color:var(--ink);font-weight:600}
.lede{font-size:19px;color:var(--ink);line-height:1.6}
ul,ol{margin:0 0 18px;padding-left:22px;color:var(--ink-2)}
li{margin:0 0 9px}
a{color:var(--signal)}
a:focus-visible,summary:focus-visible{outline:2px solid var(--signal);outline-offset:3px}

/* Evidence blocks */
.panel{background:var(--panel);border:1px solid var(--rule);border-radius:4px;
  padding:22px 24px;margin:26px 0}
.figure{margin:32px 0}
.figure figcaption{font:400 13.5px/1.55 "Source Serif 4",serif;color:var(--ink-3);
  margin-top:12px;max-width:70ch}
.figure figcaption b{color:var(--ink-2);font-weight:600}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
  gap:1px;background:var(--rule);border:1px solid var(--rule);border-radius:4px;
  overflow:hidden;margin:30px 0}
.stats>div{background:var(--panel);padding:18px 20px}
.stats .k{font:600 10.5px/1.3 Archivo,sans-serif;letter-spacing:.12em;
  text-transform:uppercase;color:var(--ink-3);margin-bottom:10px}
.stats .v{font:600 30px/1 "IBM Plex Mono",monospace;letter-spacing:-.035em;color:var(--ink)}
.stats .v small{font-size:14px;font-weight:500;color:var(--ink-3);letter-spacing:0}
.stats .n{font:400 12.5px/1.45 "Source Serif 4",serif;color:var(--ink-3);margin-top:8px}

.scroll{overflow-x:auto;border:1px solid var(--rule);border-radius:4px;
  background:var(--panel);margin:24px 0}
table{border-collapse:collapse;width:100%;
  font:400 13.5px/1.5 "IBM Plex Mono",monospace}
th,td{padding:8px 12px;text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
thead th{position:sticky;top:0;background:var(--panel-2);
  font:600 10.5px/1.4 Archivo,sans-serif;letter-spacing:.08em;text-transform:uppercase;
  color:var(--ink-3);border-bottom:1px solid var(--rule);white-space:normal}
tbody td{border-bottom:1px solid var(--grid);color:var(--ink-2)}
tbody tr:last-child td{border-bottom:0}
tbody tr:hover td{background:var(--panel-2)}
td.op,td:first-child{color:var(--ink)}
.tag{display:inline-block;font:600 10px/1.6 Archivo,sans-serif;letter-spacing:.07em;
  text-transform:uppercase;padding:1px 7px;border-radius:3px;border:1px solid currentColor}
.t-ok{color:var(--good)} .t-part{color:var(--warn)} .t-no{color:var(--bad)}
.t-est{color:var(--ink-3)}

blockquote{margin:30px 0;padding:2px 0 2px 22px;border-left:3px solid var(--signal);
  font-size:18.5px;line-height:1.55;color:var(--ink)}
blockquote cite{display:block;margin-top:10px;font:500 12.5px/1.5 Archivo,sans-serif;
  color:var(--ink-3);font-style:normal;letter-spacing:.02em}
.callout{border:1px solid var(--rule);border-left:3px solid var(--signal);
  background:var(--signal-soft);border-radius:3px;padding:18px 22px;margin:26px 0;
  font-size:15.5px;line-height:1.6}
.callout p:last-child{margin-bottom:0}
.callout .lbl{font:600 10.5px/1 Archivo,sans-serif;letter-spacing:.13em;
  text-transform:uppercase;color:var(--signal);display:block;margin-bottom:9px}

svg{display:block;max-width:100%;height:auto;overflow:visible}
svg text{font-family:"IBM Plex Mono",monospace;fill:var(--ink-2)}
svg .ax{font-family:Archivo,sans-serif;font-size:10.5px;fill:var(--ink-3);
  letter-spacing:.05em;text-transform:uppercase}
svg .lab{font-size:11.5px;fill:var(--ink-2)}
svg .big{font-family:Archivo,sans-serif;font-size:13px;font-weight:600;fill:var(--ink)}
svg .sm{font-size:10px;fill:var(--ink-3)}
svg .gl{stroke:var(--grid);stroke-width:1}
svg .rule{stroke:var(--rule);stroke-width:1}
.legend{display:flex;flex-wrap:wrap;gap:7px 18px;margin:14px 0 0;
  font:500 12px/1.5 Archivo,sans-serif;color:var(--ink-2)}
.legend span{display:inline-flex;align-items:center}
.legend i{width:10px;height:10px;border-radius:2px;margin-right:7px;flex:none}
details{border:1px solid var(--rule);border-radius:4px;padding:14px 18px;
  margin:22px 0;background:var(--panel)}
summary{cursor:pointer;font:600 14px/1.4 Archivo,sans-serif;color:var(--ink-2)}
details p:first-of-type{margin-top:14px}
footer{padding:56px 0 0;color:var(--ink-3);font-size:13.5px}
@media(max-width:700px){
  body{font-size:16px}
  header.hero{padding:56px 0 30px}
  section{padding:46px 0}
  .stats .v{font-size:25px}
}
</style>
<div id="prog"></div>
<div class="wrap">
"""

W = 900  # standard chart width


def stack(rows, width=W, h=30, gap=2, pct=True):
    """100%-stacked horizontal bar. rows = [(label, value, slotIndex)]."""
    tot = sum(r[1] for r in rows) or 1
    out, x = [f'<svg viewBox="0 0 {width} {h + 30}" width="100%">'], 0.0
    for lab, val, sl in rows:
        w = val / tot * width
        d = max(0, w - gap)
        txt = f"{val:.1f}%" if pct else lab
        out.append(f'<rect x="{x:.1f}" y="0" width="{d:.1f}" height="{h}" rx="2" '
                   f'fill="var(--s{sl})"><title>{lab}: {txt}</title></rect>')
        if d > 52:
            out.append(f'<text x="{x + d / 2:.1f}" y="{h / 2 + 4:.0f}" text-anchor="middle" '
                       f'style="fill:#fff;font-weight:600;font-size:11.5px;'
                       f'paint-order:stroke;stroke:rgba(0,0,0,.3);stroke-width:2.5px">'
                       f'{txt}</text>')
        x += w
    out.append("</svg>")
    return "".join(out)


def legend(items):
    return ('<div class="legend">' + "".join(
        f'<span><i style="background:var(--s{s})"></i>{l}</span>' for l, s in items)
        + "</div>")


def hbars(rows, width=W, maxv=None, label_w=210, bar_h=17, gap=11, fmt=None):
    """Horizontal bars, linear. rows = [(label, value, slot, annotation)]."""
    maxv = maxv or max(r[1] for r in rows)
    plot = width - label_w - 96
    h = len(rows) * (bar_h + gap) + 8
    out = [f'<svg viewBox="0 0 {width} {h}" width="100%">']
    for i, (lab, val, sl, ann) in enumerate(rows):
        y = i * (bar_h + gap)
        out.append(f'<text x="{label_w - 12}" y="{y + bar_h - 3}" text-anchor="end" '
                   f'class="lab">{lab}</text>')
        w = max(2.5, val / maxv * plot)
        out.append(f'<rect x="{label_w}" y="{y}" width="{w:.1f}" height="{bar_h}" rx="2" '
                   f'fill="var(--s{sl})"/>')
        t = fmt(val) if fmt else f"{val:g}"
        out.append(f'<text x="{label_w + w + 8:.1f}" y="{y + bar_h - 3}" class="big">{t}</text>')
        if ann:
            out.append(f'<text x="{label_w + w + 8 + 11 * len(t):.1f}" y="{y + bar_h - 3}" '
                       f'class="sm">{ann}</text>')
    out.append("</svg>")
    return "".join(out)


def capacity(rows, cap, width=W, h=34, note=""):
    """Stacked bar against a fixed capacity, with the free remainder drawn."""
    out, x, used = [f'<svg viewBox="0 0 {width} {h + 44}" width="100%">'], 0.0, 0
    for lab, val, sl in rows:
        w = val / cap * width
        used += val
        out.append(f'<rect x="{x:.1f}" y="0" width="{max(0, w - 2):.1f}" height="{h}" rx="2" '
                   f'fill="var(--s{sl})"><title>{lab}: {val:,} B</title></rect>')
        if w > 74:
            out.append(f'<text x="{x + w / 2:.1f}" y="{h / 2 + 4:.0f}" text-anchor="middle" '
                       f'style="fill:#fff;font-weight:600;font-size:11px;paint-order:stroke;'
                       f'stroke:rgba(0,0,0,.3);stroke-width:2.5px">{val // 1024} KB</text>')
        x += w
    free = max(0, cap - used)
    fw = free / cap * width
    out.append(f'<rect x="{x:.1f}" y="0" width="{max(0, fw - 2):.1f}" height="{h}" rx="2" '
               f'fill="var(--grid)" stroke="var(--rule)"><title>free: {free:,} B</title></rect>')
    if fw > 74:
        out.append(f'<text x="{x + fw / 2:.1f}" y="{h / 2 + 4:.0f}" text-anchor="middle" '
                   f'class="sm">{free // 1024} KB free</text>')
    out.append(f'<text x="0" y="{h + 20}" class="sm">{note}</text>')
    out.append("</svg>")
    return "".join(out)


def ladder(steps, width=W, height=280):
    """The optimisation ladder: log-y descending line with themed bands.
    steps = [(x_label, seconds, theme_slot, annotation_or_None)]"""
    pad_l, pad_r, pad_t, pad_b = 52, 16, 16, 46
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b
    lo, hi = 1.5, 50.0
    lg = lambda v: pad_t + ph * (math.log10(hi) - math.log10(v)) / (math.log10(hi) - math.log10(lo))
    n = len(steps)
    xs = [pad_l + pw * i / (n - 1) for i in range(n)]
    out = [f'<svg viewBox="0 0 {width} {height}" width="100%">']
    for gv in (2, 5, 10, 20, 50):
        y = lg(gv)
        out.append(f'<line class="gl" x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}"/>')
        out.append(f'<text x="{pad_l - 10}" y="{y + 4:.1f}" text-anchor="end" class="sm">{gv} s</text>')
    pts = " ".join(f"{x:.1f},{lg(v):.1f}" for x, (_, v, _, _) in zip(xs, steps))
    out.append(f'<polyline points="{pts}" fill="none" stroke="var(--s2)" stroke-width="2" '
               f'stroke-linejoin="round"/>')
    for x, (lab, v, sl, ann) in zip(xs, steps):
        y = lg(v)
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.4" fill="var(--s{sl})" '
                   f'stroke="var(--panel)" stroke-width="1.6">'
                   f'<title>{lab}: {v} s/forward</title></circle>')
        if ann:
            out.append(f'<text x="{x:.1f}" y="{y - 12:.1f}" text-anchor="middle" class="big">{ann}</text>')
    for i in (0, n - 1):
        out.append(f'<text x="{xs[i]:.1f}" y="{height - 26}" text-anchor="{"start" if i == 0 else "end"}" '
                   f'class="ax">{steps[i][0]}</text>')
    out.append(f'<text x="{width / 2}" y="{height - 26}" text-anchor="middle" class="ax">'
               f'24 optimisation steps</text>')
    out.append("</svg>")
    return "".join(out)


def scaling(width=W, height=300):
    """Measured speedup vs board count for the two parallel shapes."""
    pad_l, pad_b, pad_t, pad_r = 46, 46, 14, 120
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b
    xs = {1: 0, 2: 1, 4: 2, 8: 3}
    px = lambda n: pad_l + pw * xs[n] / 3
    py = lambda s: pad_t + ph * (1 - (s - 1) / 7)
    out = [f'<svg viewBox="0 0 {width} {height}" width="100%">']
    for s in (1, 2, 4, 6, 8):
        y = py(s)
        out.append(f'<line class="gl" x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + pw}" y2="{y:.1f}"/>')
        out.append(f'<text x="{pad_l - 10}" y="{y + 4:.1f}" text-anchor="end" class="sm">{s}x</text>')
    for n in (1, 2, 4, 8):
        out.append(f'<text x="{px(n):.1f}" y="{height - 26}" text-anchor="middle" class="sm">{n}</text>')
    out.append(f'<text x="{pad_l + pw / 2:.1f}" y="{height - 8}" text-anchor="middle" '
               f'class="ax">boards</text>')
    # ideal
    out.append(f'<line x1="{px(1):.1f}" y1="{py(1):.1f}" x2="{px(8):.1f}" y2="{py(8):.1f}" '
               f'stroke="var(--ink-3)" stroke-width="1.4" stroke-dasharray="4 4"/>')
    out.append(f'<text x="{px(8) + 8:.1f}" y="{py(8) + 4:.1f}" class="sm">ideal / measured DP</text>')
    series = [
        ("batch data-parallel (cases 1, 4, 5)", 2, [(1, 1.0), (2, 2.0), (4, 4.0), (8, 8.0)]),
        ("case 3 (B=4, saturates)", 4, [(1, 1.0), (2, 2.0), (4, 4.0), (8, 4.0)]),
        ("case 2 token-row split", 1, [(1, 1.0), (2, 1.56)]),
        ("case 2 (B=1, DP cannot help)", 5, [(1, 1.0), (2, 1.0), (4, 1.0), (8, 1.0)]),
    ]
    for lab, sl, pts in series:
        d = " ".join(f"{px(n):.1f},{py(s):.1f}" for n, s in pts)
        out.append(f'<polyline points="{d}" fill="none" stroke="var(--s{sl})" stroke-width="2.2" '
                   f'stroke-linejoin="round"/>')
        for n, s in pts:
            out.append(f'<circle cx="{px(n):.1f}" cy="{py(s):.1f}" r="4" fill="var(--s{sl})" '
                       f'stroke="var(--panel)" stroke-width="1.6"><title>{lab}: {s}x on {n}</title></circle>')
        ln, ls = pts[-1]
        out.append(f'<text x="{px(ln) + 9:.1f}" y="{py(ls) + 4:.1f}" class="sm">{s}x</text>')
    out.append("</svg>")
    return "".join(out)


def blocked(width=W, height=250):
    """What the impossible cases need vs what one board has. Log scale."""
    rows = [
        ("case 2 weights (fits)", 1_594_368, 3),
        ("flash budget, app partition", 3_145_728, 8),
        ("case 8 weights, int16", 50_417_664, 2),
        ("case 8 weights, fp32", 100_835_328, 2),
        ("case 14 weights, fp32", 50_421_760, 2),
        ("case 13 one activation", 524_288, 5),
        ("SRAM budget, dram0_0_seg", 321_296, 8),
        ("case 14 one activation", 409_600_000, 5),
        ("case 14 KV cache, full batch", 26_206_208_000, 5),
    ]
    rows.sort(key=lambda r: r[1])
    lab_w, pad_r = 250, 92
    plot = width - lab_w - pad_r
    lo, hi = 1e5, 5e10
    bx = lambda v: plot * (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
    bar_h, gap = 16, 10
    h = len(rows) * (bar_h + gap) + 30
    out = [f'<svg viewBox="0 0 {width} {h}" width="100%">']
    for d in range(5, 11):
        x = lab_w + bx(10 ** d)
        out.append(f'<line class="gl" x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{len(rows) * (bar_h + gap):.0f}"/>')
        unit = ["100 KB", "1 MB", "10 MB", "100 MB", "1 GB", "10 GB"][d - 5]
        out.append(f'<text x="{x:.1f}" y="{h - 10}" text-anchor="middle" class="sm">{unit}</text>')
    for i, (lab, val, sl) in enumerate(rows):
        y = i * (bar_h + gap)
        budget = sl == 8
        out.append(f'<text x="{lab_w - 12}" y="{y + bar_h - 3}" text-anchor="end" class="lab">{lab}</text>')
        w = max(2, bx(val))
        out.append(f'<rect x="{lab_w}" y="{y}" width="{w:.1f}" height="{bar_h}" rx="2" '
                   f'fill="var(--s{sl})" {"stroke=\"var(--ink-3)\" stroke-dasharray=\"3 2\"" if budget else ""}/>')
        t = (f"{val / 1e9:.1f} GB" if val >= 1e9 else
             f"{val / 1e6:.1f} MB" if val >= 1e6 else f"{val / 1e3:.0f} KB")
        out.append(f'<text x="{lab_w + w + 8:.1f}" y="{y + bar_h - 3}" class="big">{t}</text>')
    out.append("</svg>")
    return "".join(out)


def fig(svg, cap, leg=""):
    return f'<figure class="figure wide">{svg}{leg}<figcaption>{cap}</figcaption></figure>'


def sec(no, kicker, title, body):
    return (f'<section id="s{no}"><div class="col"><div class="sec-no">{no:02d}</div>'
            f'<h2>{kicker}</h2><h3>{title}</h3></div>{body}</section>')


def col(html):
    return f'<div class="col">{html}</div>'


def table(head, rows, cls=""):
    h = "".join(f"<th>{c}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="scroll wide"><table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


P = []
A = P.append

# ── hero ───────────────────────────────────────────────────────────────
A('''<header class="hero"><div class="col">
<div class="eyebrow">TechJam 2026 · Engineering report</div>
<h1>A Transformer on a $5 microcontroller</h1>
<p class="standfirst">The benchmark asks you to make a Transformer layer fast on a GPU.
We asked the opposite question — what is the <em>smallest</em> machine that can run it
at all — and answered it on a 160&nbsp;MHz RISC-V chip with no floating-point unit and
321&nbsp;KB of RAM. One forward pass went from 42.15&nbsp;seconds to 1.996. Then we
built a cluster.</p>
<p class="byline">Seeed XIAO ESP32-C3 · 10 of 14 official cases measured on physical
hardware · every result gated per element against the PyTorch reference</p>
</div></header>''')

A('''<div class="stats wide">
<div><div class="k">Case-2 forward</div><div class="v">42.15<small> s</small></div>
  <div class="n">Where we started. Measured, one board.</div></div>
<div><div class="k">After 24 steps</div><div class="v">1.996<small> s</small></div>
  <div class="n">21.1x faster, same board, same weights.</div></div>
<div><div class="k">Two boards</div><div class="v">1.276<small> s</small></div>
  <div class="n">33.0x against the original baseline.</div></div>
<div><div class="k">Accuracy</div><div class="v">25/25</div>
  <div class="n">Device seeds, zero failing elements.</div></div>
</div>''')

# ── 01 why ─────────────────────────────────────────────────────────────
A(sec(1, "The premise", "Why point a GPU benchmark at a microcontroller", col('''
<p class="lede">The official brief is unambiguous: optimise a fixed Transformer layer
<em>on a given GPU</em>, submit GPU kernels, and report the speedup against a PyTorch
baseline. We did not do that. It is worth saying so plainly before anything else in this
report is read.</p>

<p>What we did instead was take the same fourteen shapes, the same reference
implementation, and the same per-element accuracy gate, and ask a question the brief
does not ask: <strong>how far down the hardware ladder does this workload still
run?</strong> Not on a smaller GPU — on a microcontroller costing about five dollars,
with no floating-point unit, no DRAM, no operating system worth the name, and less
memory than a single one of the benchmark's activation tensors.</p>

<p>The reason this is interesting is that a GPU makes the interesting constraints
invisible. On an H200 the case-2 forward runs at under 1% model-FLOP utilisation and
nobody notices, because it finishes in milliseconds either way. Move the same arithmetic
onto a part with a 160&nbsp;MHz single-issue in-order core and every decision that a GPU
absorbs — where a tensor lives, what numeric format it carries, how many times a weight
is read from flash — becomes measurable, and usually dominant. The workload stops being
a throughput problem and becomes a <em>memory and numerics</em> problem.</p>

<div class="callout"><span class="lbl">Stated honestly</span>
<p>This is a re-framing of the brief, not a fulfilment of it. We are not claiming a GPU
kernel result. What we claim is a reproducible study of the same benchmark on hardware
roughly twelve million times slower in peak arithmetic, where the answers are different
in kind rather than in degree — and where 10 of the 14 official shapes still pass the
gate.</p></div>

<p>There is a second reason, and it is the practical one. A cluster of eight of these
boards costs about $40 and draws a couple of watts. If a Transformer of this size can be
made to run there, inference stops needing a datacentre for a whole class of embedded
problems. That is a claim the benchmark cannot make on a GPU, because the GPU was never
the constraint.</p>
''')))

# ── 02 the machine ─────────────────────────────────────────────────────
A(sec(2, "The machine", "What an ESP32-C3 actually is", col('''
<p class="lede">Everything in this report follows from four lines of a datasheet, so
they are worth stating precisely before the engineering starts.</p>
''') + '''<div class="stats wide">
<div><div class="k">Core</div><div class="v">160<small> MHz</small></div>
  <div class="n">Single RV32IMC core. In-order, single-issue, 4-stage.</div></div>
<div><div class="k">Floating point</div><div class="v">none</div>
  <div class="n">No FPU. Every fp32 operation is a library call.</div></div>
<div><div class="k">Usable RAM</div><div class="v">321<small> KB</small></div>
  <div class="n">321,296 B of dram0_0_seg. No PSRAM, no DRAM.</div></div>
<div><div class="k">Flash</div><div class="v">4<small> MB</small></div>
  <div class="n">3.1 MB app partition, read through a 16 KB cache.</div></div>
</div>''' + col('''
<p>The board is a Seeed XIAO ESP32-C3, about the size of a thumbnail and about five
dollars. There is no data cache — the 16&nbsp;KB cache covers execute-in-place reads from
flash only. IRAM and DRAM are aliases of the same physical SRAM pool, which matters later:
you cannot buy instruction-memory speed by spending data memory, because there is only
one pool.</p>

<p>For scale: the case-2 forward is 0.134&nbsp;GFLOP of arithmetic over 398,592
parameters. At fp32 those weights are 1,594,368 bytes — they fit in flash with room to
spare, and do not fit in RAM at all, four times over. That single fact shapes the entire
implementation.</p>
''')))

# ── 03 no FPU ──────────────────────────────────────────────────────────
A(sec(3, "Challenge one", "There is no floating-point unit", col('''
<p class="lede">The Transformer reference is written in fp32. The chip cannot do fp32.
Every add, multiply, divide and exponential in the reference becomes a call into
libgcc's soft-float library, and those calls are not cheap in the way a cache miss is
expensive — they are cheap operations that have simply become a hundred times slower.</p>
''') + fig(hbars([
    ("int16 multiply (C3)", 1, 3, "one instruction"),
    ("fp32 add, hardware FPU", 25, 8, "for reference"),
    ("fp32 divide, hardware FPU", 69, 8, "for reference"),
    ("fp32 add, C3 soft-float", 100, 2, ""),
    ("fp32 divide, C3 soft-float", 102, 2, ""),
    ("old exp-index path", 450, 2, "int64→float→int"),
], fmt=lambda v: f"{v:g} cyc"),
    "<b>The soft-float tax.</b> Cycles per operation. A part with a hardware FPU pays "
    "25 cycles for an fp32 add; the C3 pays about 100, because the operation is "
    "synthesised in software. The integer multiply the chip <em>can</em> do costs one. "
    "This ratio — roughly 100:1 — is why the entire optimisation effort is a migration "
    "from floating point to fixed point, and not a search for better cache behaviour. "
    "Figures from Espressif's own performance guidance, recorded in the project's "
    "optimisation log.") + col('''
<p>Profiling the untouched implementation made the consequence concrete. Of the 42.15
seconds a single forward took, <strong>30.09 seconds — 71.4% — was attention</strong>,
and essentially all of it was soft-float: 16.7 million fp32 dequantise-multiplies in the
QK dot products, about a million <code>expf</code> calls in the softmax, and 8.4 million
fp32 multiply-accumulates in the PV product.</p>
'''
) + fig(stack([("attention", 70.6, 2), ("Q/K/V projections", 13.5, 1),
               ("other GEMMs", 8.6, 3), ("GELU", 3.4, 4),
               ("quantise", 1.9, 5), ("LayerNorm + residual", 2.1, 7)]),
        "<b>Where the 42.15 seconds went.</b> Attention was not the bottleneck in the "
        "usual sense of being the largest matrix multiply — the GEMMs move far more "
        "data. It dominated because it was the part still doing floating-point "
        "arithmetic element by element.",
        legend([("attention", 2), ("Q/K/V projections", 1), ("other GEMMs", 3),
                ("GELU", 4), ("quantise", 5), ("LayerNorm + residual", 7)]))
    + col('''
<p>The GEMMs were already running on an integer path and were still slow for a different
reason: measured at <strong>36 cycles per multiply-accumulate</strong> where an ideal
int32 MAC loop on this core should cost 5&ndash;8. That is a code-generation and
register-pressure problem, not a numerics one, and it needed a different fix.</p>
''')))

# ── 04 memory ──────────────────────────────────────────────────────────
A(sec(4, "Challenge two", "The memory is smaller than one tensor", col('''
<p class="lede">The linker grants the application 321,296 bytes. A single 128x128 fp32
activation is 65,536 bytes. The reference implementation wants several of them live at
once, plus per-head projections, plus quantisation scratch — and the framework, the WiFi
stack and the stack itself all come out of the same pool.</p>
''') + fig(capacity([
    ("activation buffers", 196_608, 1),
    ("Q15 scratch", 32_768, 3),
    ("per-head Q/K/V", 24_576, 4),
    ("framework, drivers", 13_080, 7),
    ("initialised data", 7_308, 5),
], 321_296, note="321,296 B dram0_0_seg — read from the link map, not assumed"),
    "<b>The whole budget, one bar.</b> The optimised case-2 firmware: three 64&nbsp;KB "
    "activation buffers, a 32&nbsp;KB Q15 scratch, three per-head buffers, and roughly "
    "13&nbsp;KB of framework state, leaving about 47&nbsp;KB. Earlier in the "
    "optimisation sequence one build sat at <b>99.9% of the segment with 384 bytes "
    "free</b>. Weights are not on this chart at all — all 1.59&nbsp;MB of them live in "
    "flash and are streamed.",
    legend([("activation buffers", 1), ("Q15 scratch", 3), ("per-head Q/K/V", 4),
            ("framework, drivers", 7), ("initialised data", 5)]))
    + col('''
<p>Three consequences run through the whole project.</p>

<p><strong>Weights never enter RAM.</strong> They are embedded in the flash image and
read through the 16&nbsp;KB execute-in-place cache as the kernels stream them: 786,432
bytes of int16 weight matrices per forward, derived from measured call counts. Any
scheme that required a resident weight matrix was off the table from the start.</p>

<p><strong>Buffers are aliased, not allocated.</strong> The final layout re-uses the
same 64&nbsp;KB regions under different types at different points in the forward — the
attention context is an int16 view of a buffer that held fp32 a moment earlier. Adding a
buffer was almost never an option; the residual path was eventually carried as int32
inside a union with the fp32 view precisely so it would cost zero new bytes.</p>

<p><strong>Some shapes simply do not fit.</strong> Case 9 (H=1) and case 10 (H=2)
failed to link — the linker reported <code>dram0_0_seg overflowed by 73,072 bytes</code>
and <code>by 23,920 bytes</code> respectively. Per-head state scales with S&middot;HD, so
fewer heads means larger per-head buffers. Both were eventually made to fit by patching
the SDK's own static allocations, which is discussed in section 8.</p>

<div class="callout"><span class="lbl">The constraint that bit hardest</span>
<p>Adding WiFi to a compute node costs roughly 85,500&nbsp;B of static DRAM plus about
69&nbsp;KB of heap. The full-sequence firmware needs 274,564&nbsp;B. Together that is
about 429&nbsp;KB against a 328&nbsp;KB budget — short by roughly 100&nbsp;KB. A board
could compute, or it could talk, but not both, until the schedule was rewritten.</p></div>
''')))

# ── 05 communication ───────────────────────────────────────────────────
A(sec(5, "Challenge three", "Talking between boards costs more than computing", col('''
<p class="lede">If one board is too small, use several. But the moment a forward is split
across boards, the parts have to exchange activations — and on this hardware the link is
slower than the arithmetic it is meant to accelerate.</p>

<p>The numbers set the scale. A case-2 forward has to move 64&nbsp;KB in and 64&nbsp;KB
out. Over USB CDC, paced to the rate the chip's native USB will accept without dropping
bytes, that is about <strong>1.3 seconds per input</strong> — comfortably longer than the
1.996&nbsp;second forward it feeds. Transport was 51% of end-to-end time in the two-board
batch runs.</p>

<p>So the radio had to be made fast. We built a dedicated link benchmark to find out what
an ESP32-C3 pair can actually sustain, and the answer depended almost entirely on
configuration rather than on protocol design.</p>
''') + fig(hbars([
    ("ESP-NOW, default config", 60.9, 8, "P=240 B"),
    ("ESP-NOW, tuned", 197.0, 3, "3.2x"),
    ("TCP over WiFi, best case", 723.0, 1, "4096 B payload"),
    ("USB CDC, paced", 50.0, 5, "the thing we were escaping"),
], fmt=lambda v: f"{v:,.0f} KB/s"),
    "<b>Link throughput, measured.</b> Tuning ESP-NOW — disabling power save, unicasting "
    "to a paired peer, forcing MCS7 with HT40, raising TX power — took it from 60.9 to "
    "197.0&nbsp;KB/s at a 240-byte payload, with median round-trip latency falling from "
    "1,915&nbsp;&micro;s to 850. The best raw figure is TCP at a 4&nbsp;KB payload "
    "(5.787&nbsp;Mbps), but TCP's window on this stack caps a single connection well "
    "below what the PHY allows.")
    + col('''
<p>Even at 197&nbsp;KB/s, a naive split loses. The first working two-board case-2 build
used blocking TCP and spent <strong>1.83 seconds of a 4.64-second forward</strong>
waiting on the link. Streaming it per attention head over TCP made it worse — 2.57 to
3.38 seconds blocked — because the per-transfer overhead multiplied.</p>

<p>What worked was UDP with negative-acknowledgement recovery, streamed per head, on a
dedicated link task, so that a board releases each head's K/V block the instant it is
projected and only blocks when attention actually needs the peer's. That took time spent
waiting from 1.83 seconds down to <strong>5&ndash;88 milliseconds</strong> against
roughly 1.6 seconds of raw transfer — the transfer did not get faster, it got hidden.</p>
''') + table(
    ["link strategy", "wall time", "blocked on link"],
    [["blocking TCP, whole 32 KB block", "4.64 s", "1.83 s"],
     ["TCP streamed per head, overlapped", "5.87&ndash;6.62 s", "2.57&ndash;3.38 s"],
     ["<b>UDP + NAK, streamed per head</b>", "<b>3.06 s</b>", "<b>0.003&ndash;0.032 s</b>"]])
    + col('''
<p class="note" style="font-size:13.5px;color:var(--ink-3)">Measured on the same two
boards on opt18-era kernels; the wall times predate the final kernel work, so compare the
columns to each other rather than to the headline figure.</p>
''')))

# ── 06 the ladder ──────────────────────────────────────────────────────
LADDER = [
    ("baseline", 42.15, 2, "42.15 s"), ("int QK", 15.21, 2, None),
    ("exp LUT", 13.70, 2, None), ("GEMM tiling", 6.91, 1, None),
    ("fused quant", 6.56, 3, None), ("core4", 5.27, 1, None),
    ("head GEMM", 4.862, 1, None), ("fused LN", 4.784, 3, None),
    ("int LN", 4.160, 3, None), ("ctx fusion", 3.969, 3, None),
    ("int PV", 3.688, 2, None), ("core4_v2", 3.664, 1, None),
    ("int amax", 3.205, 3, None), ("core5", 2.982, 1, None),
    ("KB0 asm", 2.838, 6, None), ("Q15 epilogue", 2.701, 3, None),
    ("bias fold", 2.447, 6, None), ("QK unroll", 2.386, 2, None),
    ("int residual", 2.122, 5, None), ("compose", 2.056, 6, None),
    ("asm fix", 1.996, 6, "1.996 s"),
]

A(sec(6, "The method", "Twenty-four steps from 42.15 to 1.996 seconds", col('''
<p class="lede">Case 2 — one input, 128 tokens, 128 dimensions, 4 heads, 4 layers — is the
shape everything else was built on. It is the only case whose baseline is a real
measurement rather than a projection, and it is where every technique was developed and
gated before being applied elsewhere.</p>
''') + fig(ladder(LADDER),
    "<b>The ladder.</b> Seconds per forward, log scale, one point per recorded "
    "optimisation. There is no single win: the largest step (integer attention) accounts "
    "for about 64% of the total gain, and the remaining 36% is twenty-three steps of "
    "three to eleven percent each. Point colour marks the family of change.",
    legend([("integer numerics", 2), ("GEMM kernels", 1), ("fusion / pass removal", 3),
            ("hand-written assembly", 6), ("integer dataflow", 5)]))
    + col('''
<p>The work falls into five families, and they are worth separating because they attack
genuinely different costs.</p>

<h4>1. Move attention into fixed point</h4>
<p>The largest single win, and the obvious one once the profile existed. QK dot products
became int64 accumulations with the dequantisation scale folded into a single fp32
multiply per logit. The softmax exponential became a 513-entry int16 lookup table with
7-bit linear interpolation — and then, one step later, an all-integer table index: a
single Q32 fixed-point multiply replacing an int64&rarr;float conversion, two fp32
multiplies and a float&rarr;int conversion, about 450 cycles reduced to one multiply.
<strong>That step alone took the exp stage from 194.8 to 23.2&nbsp;&micro;s per
call.</strong> The PV product went last: rescaling each softmax row so it sums to
approximately 32,767 keeps the accumulation inside int32, so each element costs one
16x16 multiply with no 64-bit accumulate.</p>

<h4>2. Rewrite the GEMM inner loop, six times</h4>
<p>Measured at 36 cycles per MAC at the start. The progression — <code>core</code>
through <code>core5</code> — is entirely about register tiling: reuse each fetched weight
across more output rows, then reuse each activation across more output columns, then
hoist loads across a K-pair to hide flash latency. The instructive failure is
<code>core4_v2</code>, which needed roughly 36 live registers against the RV32IMC ABI's
28 and spilled on every iteration. <code>core5</code> uses a smaller 4x2 tile, fits in
about 20 registers, and runs faster despite doing less per iteration: <strong>7.48 to
6.21 cycles per MAC</strong>. Every kernel in the chain is bit-identical to its
predecessor, because the accumulation order never changes.</p>

<h4>3. Delete passes rather than speed them up</h4>
<p>The recurring discovery was that the scaffolding around the arithmetic cost more than
the arithmetic. Separate amax-scan, quantise and staging passes over 64&nbsp;KB buffers
were removed one at a time by fusing them into the producer: GELU folded into the FFN
quantisation; LayerNorm's output scan replaced by an <em>analytic</em> bound on the
maximum, so the scale is known before normalising rather than after; attention writing
Q15 context directly so the output projection's quantise step disappears entirely.</p>

<h4>4. Hand-write the hot loops in assembly</h4>
<p>Once the C kernels were at the register limit, the remaining wins were scheduling ones
the compiler would not make. The head-GEMM inner loop became 28 instructions per 8 MACs
with a read-after-write distance of 8 — far enough that the multiplier never stalls.
A C fallback is kept for every assembly path so the host tests still build.</p>

<h4>5. Make the dataflow integer end to end</h4>
<p>The last floating-point object was the residual itself, and it was forcing three costs
per layer: float GEMM epilogues, separate residual-add passes, and a float
quantise inside every LayerNorm. Carrying it as int32 at a fixed scale — inside a union
with the fp32 view, so it cost no new memory — let the GEMMs accumulate straight into the
residual at about <strong>10 instructions per element instead of 268</strong>, and
deleted the residual passes entirely. Accuracy improved.</p>
''') + fig(stack([("Q/K/V projections", 28.9, 1), ("other GEMMs", 36.1, 3),
                  ("attention", 18.7, 2), ("LayerNorm + final", 9.5, 7),
                  ("residual", 3.6, 5), ("GELU", 2.6, 4)]),
    "<b>Where the time goes now.</b> The same forward, profiled again. Attention has "
    "fallen from 71% of the total to about 19%, and the GEMMs — which were always doing "
    "most of the arithmetic — are now correctly the largest cost. This is what a "
    "successful optimisation looks like in a profile: not everything smaller, but the "
    "bottleneck moved to where the real work is.",
    legend([("Q/K/V projections", 1), ("other GEMMs", 3), ("attention", 2),
            ("LayerNorm + final", 7), ("residual", 5), ("GELU", 4)]))
    + col('''
<p>The cumulative effect on utilisation is the cleanest summary: model-FLOP utilisation
went from <strong>2.0% to 42.2%</strong> of the board's derived peak, and held within
0.1 percentage points across cases 1 through 5.</p>
''')
    + '''<details class="wide"><summary>Things that did not work — and why they are in this report</summary>
<p>An optimisation log that records only the wins is a sales document. Several failures
here changed the design more than the successes did.</p>
<ul>
<li><strong>Putting hot code in IRAM.</strong> Not possible. IRAM and DRAM are aliases of
one 313&nbsp;KB pool, and the build was at 99.9% of it. Moving even 128 bytes of kernel
text overflowed the data segment.</li>
<li><strong>Assuming flash streaming was the bottleneck.</strong> A device microbenchmark
said otherwise: 3.65 cycles per MAC with weights in flash against 3.32 in SRAM. The
512-byte working set stays in the 16&nbsp;KB cache. We had been about to optimise the
wrong thing.</li>
<li><strong>A bigger GEMM tile.</strong> A 4x4 tile with 16 accumulators regressed to
3.80&ndash;4.43 cycles per MAC on register spills. Packed 32-bit weight loads also
regressed. Both were reverted.</li>
<li><strong>A compile-time GELU table.</strong> Mathematically impossible, not merely
inaccurate: the erf argument scales with each layer's runtime maximum, so a table indexed
by the quantised value cannot know the real input. Measured error reached 2.04 raw units.
The table is rebuilt per layer instead.</li>
<li><strong>A Q15 residual.</strong> Its least-significant bit was too coarse — error
4.1e-3 against a 2e-3 gate. Narrowing the range got to 1.78e-3 but left no headroom for
outliers. int32 with a per-row rescale was the resolution.</li>
<li><strong>Round-to-nearest in the context epilogue.</strong> A truncating cast was both
52&nbsp;ms faster <em>and</em> more accurate (1.27e-3 to 1.12e-3). Not what we expected.</li>
<li><strong>Two assembly bugs worth the warning.</strong> A register-reuse error made one
GEMM column compute (act&middot;w0)&middot;w1 instead of act&middot;w1; and a regenerated
probe lost its local label, so a branch bound to an unrelated function in the same
translation unit and the device crashed on a load fault.</li>
</ul></details>'''))

# ── 07 parallelising ───────────────────────────────────────────────────
SPLIT_SVG = '''<svg viewBox="0 0 900 250" width="100%">
<text x="0" y="14" class="ax">A — batch data-parallel (cases 1, 3, 4, 5, 7, 9-12)</text>
''' + "".join(
    f'<rect x="{18 + i * 108}" y="34" width="86" height="52" rx="4" fill="none" '
    f'stroke="var(--s1)" stroke-width="1.6"/>'
    f'<text x="{61 + i * 108}" y="58" text-anchor="middle" class="lab">board {i + 1}</text>'
    f'<text x="{61 + i * 108}" y="74" text-anchor="middle" class="sm">B/N inputs</text>'
    f'<text x="{61 + i * 108}" y="26" text-anchor="middle" class="sm">input {i + 1}</text>'
    f'<line x1="{61 + i * 108}" y1="30" x2="{61 + i * 108}" y2="34" stroke="var(--ink-3)"/>'
    for i in range(8)) + '''
<text x="18" y="106" class="sm">Boards exchange nothing. Speedup is exactly min(B, N).</text>
<line class="rule" x1="0" y1="126" x2="900" y2="126"/>
<text x="0" y="152" class="ax">B — token-row split, inside one forward (case 2, B=1)</text>
<rect x="18" y="172" width="200" height="56" rx="4" fill="none" stroke="var(--s2)" stroke-width="1.6"/>
<text x="118" y="196" text-anchor="middle" class="lab">board 1 — even rows</text>
<text x="118" y="212" text-anchor="middle" class="sm">64 of 128 tokens</text>
<rect x="330" y="172" width="200" height="56" rx="4" fill="none" stroke="var(--s2)" stroke-width="1.6"/>
<text x="430" y="196" text-anchor="middle" class="lab">board 2 — odd rows</text>
<text x="430" y="212" text-anchor="middle" class="sm">64 of 128 tokens</text>
<path d="M222 188 H326" stroke="var(--s4)" stroke-width="1.8" marker-end="url(#ar)"/>
<path d="M326 212 H222" stroke="var(--s4)" stroke-width="1.8" marker-end="url(#ar2)"/>
<text x="274" y="182" text-anchor="middle" class="sm">K/V, 32,800 B</text>
<text x="274" y="228" text-anchor="middle" class="sm">per layer, each way</text>
<text x="556" y="192" class="sm">Causal attention needs j &lt;= i, so every layer</text>
<text x="556" y="208" class="sm">forces one K/V exchange: 131,200 B per forward.</text>
<defs>
<marker id="ar" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
<path d="M0 0 L8 4 L0 8 z" fill="var(--s4)"/></marker>
<marker id="ar2" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
<path d="M0 0 L8 4 L0 8 z" fill="var(--s4)"/></marker>
</defs></svg>'''

A(sec(7, "Scaling out", "Two ways to split, and why they scale differently", col('''
<p class="lede">Thirteen of the fourteen cases have a batch dimension greater than one.
That single fact splits the parallelisation problem cleanly in two, and the two halves
behave nothing alike.</p>
''') + fig(SPLIT_SVG,
    "<b>The two shapes.</b> When B &gt; 1 the inputs are independent forwards over shared "
    "weights, so board <em>i</em> takes input <em>i mod N</em> and no board ever speaks to "
    "another. When B = 1 there is only one forward to split, and it has to be cut "
    "<em>inside</em> the model — which causal attention makes expensive.")
    + col('''
<p>The batch case is as good as parallelism gets. The boards share no state, exchange
nothing during a forward, and the speedup is exactly <code>min(B, N)</code>. Measured on
eight physical boards over WiFi: <strong>8.00x</strong> for cases 1, 4 and 5, with 213 of
213 forwards passing the gate and zero failing elements. Case 3 saturates at 4.00x
because B=4 — four boards work and four sit idle, which we report as 4.00x rather than
counting the idle boards.</p>

<p>Case 2 is the interesting one, because B=1 means data parallelism has nothing to
divide. The only remaining axis is the token dimension. Every operator in the case-2 body
is per-token — LayerNorm, all four projections, both residuals, the FFN, GELU, the final
norm — <em>except</em> causal attention, which needs every key and value at or before the
current row. So each board takes half the token rows and the boards exchange one K/V
block per layer.</p>

<p>A detail that matters more than it looks: rows are assigned by parity, not by cutting
the sequence in half. Under a causal mask, row <em>i</em> does <em>i+1</em> units of work,
so a contiguous split would give one board 2,080 score rows and the other 6,176. The
parity interleave gives 4,096 against 4,160 — <strong>a 1.6% imbalance instead of a
3x one</strong>.</p>
''') + fig(scaling(),
    "<b>Measured scaling.</b> Batch-parallel cases track the ideal line exactly, because "
    "there is nothing to synchronise. Case 3 stops at 4x because it runs out of batch. "
    "Case 2 gets 1.56x on two boards and cannot use data parallelism at all — its single "
    "input activates exactly one node no matter how many are available.",
    legend([("batch data-parallel", 2), ("case 3, B=4", 4),
            ("case 2, token-row split", 1), ("case 2 under data parallelism", 5)]))
    + col('''
<h4>Why 1.56x and not 2.00x</h4>
<p>This is the most honest number in the project, so it is worth explaining rather than
rounding up. Three costs do not halve when the tokens do:</p>
<ul>
<li><strong>Weight traffic is replicated.</strong> Both boards stream all 24 quantised
weight matrices from flash every layer. That cost is per board, not per row, so splitting
the rows does not touch it.</li>
<li><strong>Per-call fixed costs double.</strong> Every GEMM, LayerNorm and quantise is
now invoked twice over half the rows each, so the amax scans, the per-layer GELU table
rebuild and the scale arithmetic are all paid twice.</li>
<li><strong>Attention keeps a 1.6% imbalance</strong>, plus the extra per-source logit
conversion in the merged softmax.</li>
</ul>
<p>There is a further wrinkle worth recording: the same split measured <strong>1.73x on
earlier kernels</strong>. The partition never changed — the single-board baseline got 2.6x
faster underneath it, so the fixed costs became a larger fraction of a smaller number. A
parallel speedup is a ratio, and optimising the denominator makes it look worse.</p>
'''
)))

# ── 08 generalising ────────────────────────────────────────────────────
CASES = [
    ("01", "64,128,128,4,128,4", "2,697.6 s <span class='tag t-est'>est</span>",
     "127.36 s", "63.7 s", "33.713 s", "ok", "64/64 PASS"),
    ("02", "1,128,128,4,128,4", "42.15 s", "1.990 s", "<b>1.276 s</b>",
     "4.220 s &dagger;", "ok", "25/25 seeds"),
    ("03", "4,128,128,4,128,4", "168.6 s <span class='tag t-est'>est</span>",
     "7.96 s", "4.0 s", "4.218 s &Dagger;", "ok", "4/4 PASS"),
    ("04", "16,128,128,4,128,4", "674.4 s <span class='tag t-est'>est</span>",
     "31.84 s", "15.9 s", "8.438 s", "ok", "16/16 PASS"),
    ("05", "128,128,128,4,128,4", "5,395.2 s <span class='tag t-est'>est</span>",
     "254.72 s", "127.4 s", "67.451 s", "ok", "128/128 PASS"),
    ("06", "10000,128,128,4,128,4", "&mdash;", "&mdash;", "&mdash;", "&mdash;",
     "part", "builds, host gate PASS"),
    ("07", "64,128,32,4,32,4", "&mdash;", "30.427 s", "15.822 s", "3.963 s",
     "ok", "64/64 PASS"),
    ("08", "64,128,1024,4,1024,4", "&mdash;", "&mdash;", "&mdash;", "&mdash;",
     "no", "weights exceed flash"),
    ("09", "64,128,128,1,128,4", "&mdash;", "138.027 s", "&mdash;", "28.508 s",
     "ok", "64/64 PASS"),
    ("10", "64,128,128,2,128,4", "&mdash;", "138.536 s", "119.101 s", "29.793 s",
     "ok", "64/64 PASS"),
    ("11", "64,128,128,16,128,4", "&mdash;", "138.610 s", "206.354 s", "51.604 s",
     "ok", "64/64 PASS"),
    ("12", "64,32,128,4,128,4", "&mdash;", "33.879 s", "17.091 s", "4.282 s",
     "ok", "64/64 PASS"),
    ("13", "64,1024,128,4,128,4", "&mdash;", "&mdash;", "&mdash;", "&mdash;",
     "no", "activations exceed SRAM"),
    ("14", "32,100000,1024,16,1024,2", "&mdash;", "&mdash;", "&mdash;", "&mdash;",
     "no", "impossible by ~4 orders"),
]
TAGS = {"ok": '<span class="tag t-ok">measured</span>',
        "part": '<span class="tag t-part">partial</span>',
        "no": '<span class="tag t-no">not run</span>'}

A(sec(8, "Coverage", "Ten of fourteen shapes, on physical hardware", col('''
<p class="lede">Case 2 was the development shape. The question that decides whether any
of this generalises is what happens when the geometry changes — and the answer was mostly
determined by memory, not by arithmetic.</p>
''') + table(
    ["case", "shape (B,S,D,H,F,L)", "baseline 1 board", "optimised 1 board",
     "2 boards", "8-node WiFi", "status", "gate"],
    [[c[0], f"<code>{c[1]}</code>", c[2], c[3], c[4], c[5], TAGS[c[6]], c[7]] for c in CASES])
    + col('''
<p style="font-size:13.5px;color:var(--ink-3)">All times are the device's own measurement
of the complete four-layer body, excluding host serial transfer. &dagger; case 2 has B=1,
so only one data-parallel node activates — this is the tiled single-forward time, not an
eight-board speedup. &Dagger; case 3 has B=4 and saturates at four active nodes.
<b>Only case 2's baseline is a measurement</b>; cases 1, 3, 4 and 5 were never run on the
pre-optimisation firmware, so their baselines are projected as B x 42.15 s and marked
accordingly.</p>

<h4>What porting actually took</h4>
<p>The kernels were made shape-parametric, so cases 7 and 12 — which only shrink D, F or
S — needed no new numerics at all. Case 7 (D=F=32) is small enough that its WiFi image
links at 108,300&nbsp;B of static RAM with no tiling required.</p>

<p>The head-count cases were harder, and in an unintuitive direction: <strong>fewer heads
is worse</strong>. Per-head state scales with S&middot;HD, so at H=1 the per-head buffers
are eight times larger than at H=8. Cases 9 (H=1) and 10 (H=2) did not link — over by
73,072 and 23,920 bytes. Making them run meant going below our own code and shrinking the
SDK's static allocations: the interrupt stack, the coredump stack, the process-status
buffer. Case 9 now links at 273,180&nbsp;B and runs; case 10 needed a buffer overlay and
sits at 97.25% of the segment with 8,848 bytes to spare.</p>

<p>Case 11 (H=16) fit comfortably and is the counterexample that confirms the mechanism —
more heads means smaller per-head state, 256,180&nbsp;B and 78.2% of budget.</p>

<div class="callout"><span class="lbl">A result we did not want</span>
<p>To fit WiFi alongside the model, a 16-row sequential tile schedule was added. It works
— but a tiled forward takes <strong>4.215 s against 1.990 s</strong> on the untiled USB
build. Tiling is a memory enabler, not a speed optimisation, and it costs 2.12x per
device. For case 11 the consequence is blunt: two tiled replicas are <em>1.49x slower</em>
than one plain board, and four only just cross over at 1.34x.</p></div>
''')))

# ── 09 what cannot run ─────────────────────────────────────────────────
A(sec(9, "The limits", "Four cases we cannot run, and exactly why", col('''
<p class="lede">Three of the four failures are memory, and the arithmetic is not close.
It is worth doing the sums explicitly rather than saying "too big", because the margin
tells you whether more boards would fix it.</p>
''') + fig(blocked(),
    "<b>What the hard cases need against what one board has.</b> Log scale &mdash; the "
    "dashed bars are the budgets. Case 8's fp32 weights are 32x the flash partition, and "
    "still 16x over when quantised to int16. Case 14's single activation tensor is 1,250x "
    "the entire SRAM, and its KV cache across the batch is roughly 24&nbsp;GiB.",
    legend([("weights", 2), ("activations / KV", 5), ("fits today", 3), ("board budget", 8)]))
    + col('''
<h4>Case 6 &mdash; (10000, 128, 128, 4, 128, 4). Not a memory problem at all.</h4>
<p>The firmware builds, links at 83.2% of RAM, and passes the host gate 50/50. The
blocker is time: 10,000 forwards at 1.990&nbsp;s is <strong>19,900 seconds &mdash; about
5.5 hours</strong> of device compute, plus roughly 1.31&nbsp;GB of serial transfer to
feed it. We did not run it, and we do not claim it. With enough boards it is
straightforwardly feasible; the project's own estimate is around 300.</p>

<h4>Case 8 &mdash; (64, 128, 1024, 4, 1024, 4). The weights do not fit in flash.</h4>
<p>D=F=1024 gives 25,208,832 parameters &mdash; <strong>100,835,328 bytes in fp32, 32.1x
the 3.1&nbsp;MB app partition</strong>. Quantising to int16 halves it and leaves it 16x
over. A <em>single</em> D&times;D projection matrix in int16 is 2&nbsp;MiB, which is 6.5x
the entire RAM budget, so no one weight matrix is ever resident. Compute is not the
issue: 429.5&nbsp;GFLOP would take about 1.8 hours at the measured rate. This one needs
weight sharding across roughly 28&ndash;32 boards.</p>

<h4>Case 13 &mdash; (64, 1024, 128, 4, 128, 4). The weights fit; the activations do not.</h4>
<p>The weights are identical to case 2's and sit happily in flash. S=1024 is what breaks
it: one S&times;D fp32 activation is <strong>524,288 bytes against 321,296 of RAM</strong>
&mdash; a single tensor exceeds the whole budget before anything else is allocated. The
dense attention score matrix would be 4&nbsp;MiB per head, 13x the segment. This is the
case that genuinely needs online, block-wise attention that never materialises S&times;S,
which we designed but did not implement.</p>

<h4>Case 14 &mdash; (32, 100000, 1024, 16, 1024, 2). Out by four orders of magnitude.</h4>
<p>One S&times;D fp32 activation is 409,600,000 bytes &mdash; <strong>1,250x the
SRAM</strong>, and about 100x the entire flash part. The KV cache across the batch is
roughly 24&nbsp;GiB. The dense score matrix for one head of one layer would be 40&nbsp;GB.
The arithmetic is 2.70&nbsp;PFLOP, which at the measured single-board rate is
<strong>464 days</strong>. Storage alone would need something over 5,600 boards. This
case is not a scaling problem; it is a different machine.</p>
''')))

# ── 10 how we know ─────────────────────────────────────────────────────
A(sec(10, "Evidence", "How we know any of this is true", col('''
<p class="lede">Two claims in this report are load-bearing &mdash; that it is fast, and
that it is still correct &mdash; and the second is the one that makes the first mean
anything.</p>

<h4>The gate</h4>
<p>The benchmark's own criterion, applied per output element:
<code>|error| &le; 0.002 <b>or</b> |error| &le; 0.02 &times; |reference|</code>. Note the
disjunction &mdash; it is not <code>torch.isclose</code>. The official harness runs 5
random trials; we run <strong>25 seeded trials and require zero failing elements</strong>,
on the device and again on the host, in both the integer FAST path and a pure fp32 EXACT
path kept bit-identical throughout as a reference.</p>

<p>Across every case the worst device error observed was <strong>1.46e-3 against a 2e-3
tolerance</strong> &mdash; a margin of about 1.4x maintained through 24 numeric changes.
Many individual steps are stronger than the gate: six of the kernel rewrites are
<em>bit-exact</em> against their predecessors, verified element by element rather than
statistically.</p>

<p>One thing worth being clear about: the benchmark ships no dataset and no trained
weights. The model is built from seeded random values, so "correct" means <em>matching
the PyTorch reference on those same random weights</em>. There is no accuracy claim about
a task here, only an equivalence claim about arithmetic.</p>

<h4>The profiler</h4>
<p>Halfway through, the measurements themselves became the weak link. The per-op timings
in the optimisation log were transcribed by hand from serial output; every memory figure
was copied out of the build summary; the flash-traffic number was arithmetic on assumed
call counts. So we built <strong>tinyprof</strong>, an operator-level profiler for this
board, and pointed it at both firmwares.</p>

<p>It found three things immediately. Zones were timed with a 1&nbsp;&micro;s timer and
discarded when they rounded to zero, so the residual-add op was <em>absent from every
profile</em> and its sibling reported 2 calls out of 12 &mdash; fixed by timing with the
cycle counter at 6.25&nbsp;ns. Two ops were nested inside others and being double-counted
in the rankings. And a 32&nbsp;KB scratch buffer was missing from the memory census,
caught by cross-checking the firmware's own declaration against the linked ELF.</p>

<p>It also reproduces the previously hand-computed 768&nbsp;KiB/forward of flash traffic
from measured call counts &mdash; and finds 9,216 bytes of LayerNorm parameter reads the
hand calculation had missed.</p>
''') + '''<div class="callout wide"><span class="lbl">What we would flag to a reviewer</span>
<ul style="margin-bottom:0">
<li><b>Four of five baselines are projections.</b> Only case 2's 42.15&nbsp;s was measured
on pre-optimisation firmware. Cases 1, 3, 4 and 5 are B x 42.15&nbsp;s and marked as such
everywhere.</li>
<li><b>The 8.00x node scaling costs per-device efficiency.</b> Tiled WiFi nodes run at
19.9% utilisation against 42.2% for the best single board, so against that board the true
eight-node gain is <b>3.78x, not 8.00x</b>. Both numbers are real; only one of them is
the honest headline.</li>
<li><b>Not every port was a win.</b> Case 12's optimised build measured 0.529&nbsp;s
against an earlier 0.493&nbsp;s &mdash; about 7% slower. It is in the table at its
measured value.</li>
<li><b>The peak used for utilisation is derived, not from a datasheet.</b> 160&nbsp;MFLOP/s
per board is our own bound. Counting only unmasked causal attention rather than dense
S&sup2; would scale every utilisation figure down about 13%.</li>
<li><b>The radio was never shown to steal compute.</b> Per-forward time is constant
regardless of node count, and the documented cost of WiFi is memory. But channel
contention across eight nodes was not measured.</li>
</ul></div>'''))

A('''<footer><div class="col">
<p style="color:var(--ink-3);font-size:13.5px;line-height:1.6">
Hardware: Seeed XIAO ESP32-C3, RV32IMC at 160&nbsp;MHz, no FPU, 400&nbsp;KB SRAM
(321,296&nbsp;B usable), 4&nbsp;MB flash. Toolchain: PlatformIO with the Arduino-ESP32
core on ESP-IDF 4.4. Reference: the competition's own
<code>torch_transformer_benchmark.py</code>, seed 1234. Every figure in this report is
either a physical device measurement or explicitly labelled as a projection or a derived
bound. Per-case evidence, raw captures and the optimisation log live in the repository
under <code>benchmarks/case-NN/</code>.</p>
</div></footer>''')

A("</div>")
A('''<script>
// Scroll progress. Passive listener, and it respects reduced motion by simply
// being a width change with no transition.
(function(){
  var el = document.getElementById('prog');
  function upd(){
    var h = document.documentElement.scrollHeight - window.innerHeight;
    el.style.width = (h > 0 ? (window.scrollY / h) * 100 : 0) + '%';
  }
  window.addEventListener('scroll', upd, {passive:true});
  window.addEventListener('resize', upd, {passive:true});
  upd();
})();
</script>''')

OUT.write_text(SHELL + "\n".join(P) + "\n", encoding="utf-8")
print(f"wrote {OUT}")
