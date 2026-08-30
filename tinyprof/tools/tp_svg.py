#!/usr/bin/env python3
"""tp_svg.py - dependency-free SVG primitives for the tinyprof report.

Hand-built rather than matplotlib for three reasons: CONTRIBUTING.md asks that
coordinators stay standard-library Python, the report has to be one file a judge
can open with no toolchain, and inline SVG stays crisp when the demo video zooms
in on it.

Marks follow the repo-independent chart rules: thin marks, 2 px surface gaps
between adjacent fills, recessive axes, values as text tokens rather than in the
series colour, and a legend whenever there is more than one series.
"""
from __future__ import annotations

import html
import math

# Categorical slots 1..7 of the validated default palette, light/dark. Assigned
# in fixed order and never cycled: a series keeps its colour when the op set
# changes, which is what makes two charts on this page comparable by eye.
SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9"]
OTHER_SLOT = 7  # "Other" always takes the muted slot, never a hue


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def _t(x, y, text, cls="lbl", anchor="start", dy=0):
    return (f'<text x="{x:.1f}" y="{y + dy:.1f}" class="{cls}" '
            f'text-anchor="{anchor}">{esc(text)}</text>')


def stacked_share(rows, width=760, height=54, gap=2):
    """One 100%-stacked horizontal bar. `rows` = [(label, pct, slot_index)].

    The share bar is the chart that carries this report's actual argument: not
    "attention got faster" but "attention stopped being the whole forward". A
    stacked bar is the right form because the parts are shares of one whole and
    the reader's question is composition, not magnitude.
    """
    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
           f'role="img" class="chart">']
    x = 0.0
    total = sum(max(0.0, r[1]) for r in rows) or 1.0
    for label, pct, slot in rows:
        w = max(0.0, pct) / total * width
        if w <= 0:
            continue
        draw = max(0.0, w - gap)
        out.append(
            f'<rect x="{x:.2f}" y="0" width="{draw:.2f}" height="26" rx="3" '
            f'fill="var(--s{slot})"><title>{esc(label)}: {pct:.1f}% of the forward</title></rect>')
        # Direct-label only segments wide enough to hold the text. This is the
        # relief the palette validator requires for the low-contrast slots, and
        # it keeps the legend from being the only way to read the chart.
        if draw > 46:
            out.append(_t(x + draw / 2, 43, f"{label} {pct:.0f}%", "seg", "middle"))
        x += w
    out.append("</svg>")
    return "".join(out)


def grouped_log_bars(labels, series, width=760, row_h=26, pad_left=96):
    """Grouped horizontal bars on a log scale.

    Log, not linear: the values here span from 30 s down to 0.3 us. On a linear
    axis every op except attention is an invisible sliver, which is precisely
    the information an optimisation report needs to show. Each bar is labelled
    with its real value so the log axis never has to be mentally inverted.

    `series` = [(name, slot, [values...])]; None means the op is absent on that
    side, and it is drawn as a gap with a marker rather than as a zero.
    """
    n = len(labels)
    ns = len(series)
    bar_h = max(6, (row_h - 6) // ns)
    height = n * row_h + 46
    plot_w = width - pad_left - 62

    vals = [v for _, _, vs in series for v in vs if v and v > 0]
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    lo = max(lo, hi / 1e6)
    lg_lo, lg_hi = math.log10(lo), math.log10(hi)
    span = (lg_hi - lg_lo) or 1.0

    def bx(v):
        return plot_w * (math.log10(max(v, lo)) - lg_lo) / span

    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
           f'role="img" class="chart">']

    # Decade gridlines, recessive, behind the marks.
    d0, d1 = math.floor(lg_lo), math.ceil(lg_hi)
    for d in range(int(d0), int(d1) + 1):
        if not (lg_lo <= d <= lg_hi):
            continue
        gx = pad_left + plot_w * (d - lg_lo) / span
        out.append(f'<line x1="{gx:.1f}" y1="14" x2="{gx:.1f}" y2="{n * row_h + 16:.1f}" '
                   f'class="grid"/>')
        unit = f"{10 ** d:g} us" if d >= 0 else f"{10 ** (d + 3):g} ns"
        out.append(_t(gx, 10, unit, "tick", "middle"))

    for i, lab in enumerate(labels):
        y0 = 18 + i * row_h
        out.append(_t(pad_left - 8, y0 + bar_h * ns / 2 + 4, lab, "lbl", "end"))
        for k, (_nm, slot, vs) in enumerate(series):
            v = vs[i]
            y = y0 + k * (bar_h + 2)
            if v is None:
                out.append(_t(pad_left + 4, y + bar_h - 1, "absent", "absent"))
                continue
            w = max(2.0, bx(v))
            out.append(
                f'<rect x="{pad_left}" y="{y}" width="{w:.1f}" height="{bar_h}" rx="3" '
                f'fill="var(--s{slot})"><title>{esc(_nm)} {esc(lab)}: {_fmt_us(v)}</title></rect>')
            out.append(_t(pad_left + w + 5, y + bar_h - 1, _fmt_us(v), "val"))
    out.append("</svg>")
    return "".join(out)


def _fmt_us(v: float) -> str:
    if v >= 1e6:
        return f"{v / 1e6:.2f} s"
    if v >= 1e3:
        return f"{v / 1e3:.1f} ms"
    if v >= 1:
        return f"{v:.1f} us"
    return f"{v * 1000:.0f} ns"


def capacity_bar(segments, capacity, width=760, height=52, label=""):
    """A stacked bar against a known capacity, with the free remainder shown.

    Drawing the remainder is the point: "267 KB of .bss" means nothing without
    "of 321 KB", and this project's whole memory story is how little was left.
    """
    if not capacity:
        return ""
    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
           f'role="img" class="chart">']
    x = 0.0
    used = 0
    for name, val, slot in segments:
        w = val / capacity * width
        used += val
        out.append(f'<rect x="{x:.2f}" y="0" width="{max(0.0, w - 2):.2f}" height="24" rx="3" '
                   f'fill="var(--s{slot})"><title>{esc(name)}: {val:,} B</title></rect>')
        if w > 60:
            out.append(_t(x + w / 2, 16, name, "seg", "middle"))
        x += w
    free = max(0, capacity - used)
    fw = free / capacity * width
    out.append(f'<rect x="{x:.2f}" y="0" width="{max(0.0, fw - 2):.2f}" height="24" rx="3" '
               f'class="free"><title>free: {free:,} B</title></rect>')
    out.append(_t(0, 42, f"{label}{used:,} B used of {capacity:,} B - {free:,} B free "
                         f"({100.0 * used / capacity:.1f}% used)", "cap"))
    out.append("</svg>")
    return "".join(out)


def legend(items):
    """items = [(label, slot)] - always rendered for >= 2 series."""
    parts = ['<div class="legend">']
    for label, slot in items:
        parts.append(f'<span class="li"><i style="background:var(--s{slot})"></i>{esc(label)}</span>')
    parts.append("</div>")
    return "".join(parts)
