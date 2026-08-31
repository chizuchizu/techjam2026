#!/usr/bin/env python3
"""Build a production brief and reusable scientific figures for Claude Design.

This is deliberately not a presentation mock-up.  It is a handoff document:
global design rules, verified competition facts, exact slide copy, composition
instructions, chart previews, asset filenames, and evidence caveats.
"""

from __future__ import annotations

import csv
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "claude_design_instruction_pack.pdf"
ASSETS = HERE / "figure_assets"

PAGE_W, PAGE_H = 11.69, 8.27  # A4 landscape: a specification sheet, not 16:9 slides

INK = "#172033"
MUTED = "#5E6B7D"
FAINT = "#8C98A8"
LINE = "#D9DEE7"
PAPER = "#F7F8FA"
WHITE = "#FFFFFF"
NAVY = "#0B1426"
BLUE = "#2367D1"
ORANGE = "#E7652B"
GREEN = "#12805C"
RED = "#C54242"
PURPLE = "#6F55B5"
YELLOW = "#C88A13"
CYAN = "#178A9E"

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.edgecolor": LINE,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


LADDER = [
    ("baseline", 42.15, ORANGE),
    ("int QK", 15.21, ORANGE),
    ("exp LUT", 13.70, ORANGE),
    ("GEMM tile", 6.91, BLUE),
    ("fused quant", 6.56, GREEN),
    ("core4", 5.27, BLUE),
    ("head GEMM", 4.862, BLUE),
    ("fused LN", 4.784, GREEN),
    ("int LN", 4.160, GREEN),
    ("ctx fusion", 3.969, GREEN),
    ("int PV", 3.688, ORANGE),
    ("core4 v2", 3.664, BLUE),
    ("int amax", 3.205, GREEN),
    ("core5", 2.982, BLUE),
    ("KB0 asm", 2.838, PURPLE),
    ("Q15 epilogue", 2.701, GREEN),
    ("bias fold", 2.447, PURPLE),
    ("QK unroll", 2.386, ORANGE),
    ("int residual", 2.122, CYAN),
    ("compose", 2.056, PURPLE),
    ("asm fix", 1.996, PURPLE),
]

CASES = [
    ("01", "64,128,128,4,128,4", "127.36", "63.7", "33.713", "measured", "64/64 PASS"),
    ("02", "1,128,128,4,128,4", "1.990", "1.276", "4.220*", "measured", "25/25 seeds"),
    ("03", "4,128,128,4,128,4", "7.96", "4.0", "4.218**", "measured", "4/4 PASS"),
    ("04", "16,128,128,4,128,4", "31.84", "15.9", "8.438", "measured", "16/16 PASS"),
    ("05", "128,128,128,4,128,4", "254.72", "127.4", "67.451", "measured", "128/128 PASS"),
    ("06", "10000,128,128,4,128,4", "-", "-", "-", "partial", "host gate PASS"),
    ("07", "64,128,32,4,32,4", "30.427", "15.822", "3.963", "measured", "64/64 PASS"),
    ("08", "64,128,1024,4,1024,4", "-", "-", "-", "not run", "weights > flash"),
    ("09", "64,128,128,1,128,4", "138.027", "-", "28.508", "measured", "64/64 PASS"),
    ("10", "64,128,128,2,128,4", "138.536", "119.101", "29.793", "measured", "64/64 PASS"),
    ("11", "64,128,128,16,128,4", "138.610", "206.354", "51.604", "measured", "64/64 PASS"),
    ("12", "64,32,128,4,128,4", "33.879", "17.091", "4.282", "measured", "64/64 PASS"),
    ("13", "64,1024,128,4,128,4", "-", "-", "-", "not run", "activation > SRAM"),
    ("14", "32,100000,1024,16,1024,2", "-", "-", "-", "not run", "KV/state infeasible"),
]


def wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False))


def wrap_source(value: str, width: int = 34) -> str:
    """Wrap repository paths at slashes without inserting visible spaces."""
    lines = []
    for source in value.split(" · "):
        parts = source.split("/")
        current = ""
        for index, part in enumerate(parts):
            token = part + ("/" if index < len(parts) - 1 else "")
            if current and len(current) + len(token) > width:
                lines.append(current)
                current = token
            else:
                current += token
        if current:
            lines.append(current)
    return "\n".join(lines)


def page(bg=PAPER):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H), facecolor=bg)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, *, fc=WHITE, ec=LINE, lw=0.8, radius=0.008):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.003,rounding_size={radius}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    return patch


def text(ax, x, y, value, *, size=9, color=INK, weight="normal", ha="left", va="top", linespacing=1.2):
    return ax.text(
        x,
        y,
        value,
        fontsize=size,
        color=color,
        fontweight=weight,
        ha=ha,
        va=va,
        linespacing=linespacing,
        transform=ax.transAxes,
    )


def label(ax, x, y, value, color=BLUE):
    text(ax, x, y, value.upper(), size=7.2, color=color, weight="bold")


def rule(ax, x1, y1, x2, y2, color=LINE, lw=0.8):
    ax.plot([x1, x2], [y1, y2], color=color, lw=lw, transform=ax.transAxes, clip_on=False)


def save_page(pdf, fig):
    pdf.savefig(fig, facecolor=fig.get_facecolor(), bbox_inches=None)
    plt.close(fig)


def clean_chart(ax):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="y", color=LINE, linewidth=0.7, zorder=0)
    ax.tick_params(length=0, labelsize=8)
    ax.set_axisbelow(True)


def save_asset(fig, stem):
    png_path = ASSETS / f"{stem}.png"
    svg_path = ASSETS / f"{stem}.svg"
    fig.savefig(png_path, dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    # Matplotlib emits trailing spaces in SVG path data. Normalize the text so
    # generated vector assets remain clean in git without changing rendering.
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    plt.close(fig)


def generate_assets():
    ASSETS.mkdir(parents=True, exist_ok=True)

    # 01: FPU comparison. Source values are explicitly documented in the repo.
    fig, ax = plt.subplots(figsize=(7.2, 4.0), facecolor="white")
    ops = ["fp32 add", "fp32 divide", "cosf"]
    c3 = [100, 102, 2377]
    s3 = [25, 69, 121]
    x = range(len(ops))
    ax.bar([i - 0.18 for i in x], c3, width=0.36, color=ORANGE, label="ESP32-C3 · software FP")
    ax.bar([i + 0.18 for i in x], s3, width=0.36, color=BLUE, label="ESP32-S3 · hardware FPU")
    ax.set_yscale("log")
    ax.set_ylabel("cycles / operation · log scale")
    ax.set_xticks(list(x), ops)
    ax.set_ylim(10, 5000)
    ax.grid(axis="y", color=LINE, linewidth=0.7, which="both")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(length=0)
    for i, (a, b) in enumerate(zip(c3, s3)):
        ax.text(i - 0.18, a * 1.12, str(a), ha="center", fontsize=8, fontweight="bold", color=ORANGE)
        ax.text(i + 0.18, b * 1.12, str(b), ha="center", fontsize=8, fontweight="bold", color=BLUE)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout()
    save_asset(fig, "fig01_fpu_cost_c3_vs_s3")

    # 02: physical memory/storage orders of magnitude.
    fig, ax = plt.subplots(figsize=(7.6, 4.3), facecolor="white")
    names = [
        "C3 SRAM budget",
        "case 13 · one activation",
        "case 2 · fp32 weights",
        "C3 app partition",
        "case 8 · int16 weights",
        "case 14 · one activation",
        "case 14 · full-batch KV",
    ]
    values_kb = [321.296, 524.288, 1600, 3100, 50400, 409600, 26200000]
    colors = [FAINT, RED, GREEN, FAINT, ORANGE, RED, PURPLE]
    ypos = list(range(len(names)))
    ax.barh(ypos, values_kb, color=colors, height=0.58)
    ax.set_xscale("log")
    ax.set_yticks(ypos, names)
    ax.invert_yaxis()
    ax.set_xlabel("capacity / requirement · KB · log scale")
    ax.grid(axis="x", color=LINE, linewidth=0.7, which="both")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(length=0, labelsize=8)
    value_labels = ["321 KB", "524 KB", "1.6 MB", "3.1 MB", "50.4 MB", "409.6 MB", "26.2 GB"]
    for y, v, lab in zip(ypos, values_kb, value_labels):
        ax.text(v * 1.12, y, lab, va="center", fontsize=8, fontweight="bold", color=INK)
    ax.set_xlim(100, 80000000)
    fig.tight_layout()
    save_asset(fig, "fig02_memory_scale")

    # 03: final one-board SRAM budget.
    fig, ax = plt.subplots(figsize=(7.4, 2.4), facecolor="white")
    names = ["activation", "scratch", "head buffers", "framework", "init", "free"]
    values = [196608, 32768, 24576, 13080, 7308, 46956]
    colors = [BLUE, ORANGE, GREEN, PURPLE, CYAN, "#DDE3EA"]
    left = 0
    for name, value, color in zip(names, values, colors):
        ax.barh([0], [value], left=left, color=color, height=0.52, label=name)
        if value >= 13000:
            ax.text(left + value / 2, 0, f"{value/1024:.0f} KiB", ha="center", va="center", fontsize=7.5,
                    color="white" if name != "free" else MUTED, fontweight="bold")
        left += value
    ax.set_xlim(0, 321296)
    ax.set_yticks([])
    ax.set_xlabel("bytes in dram0_0_seg · total usable 321,296 B")
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0, labelsize=8)
    ax.legend(ncol=6, frameon=False, fontsize=7.3, loc="upper center", bbox_to_anchor=(0.5, 1.25))
    fig.tight_layout()
    save_asset(fig, "fig03_final_sram_budget")

    # 04: baseline profile.
    fig, ax = plt.subplots(figsize=(7.0, 4.0), facecolor="white")
    names = ["Attention", "Q/K/V projections", "Other GEMMs", "GELU", "LayerNorm + residual", "Quantize"]
    values = [70.6, 13.5, 8.6, 3.4, 2.1, 1.9]
    colors = [ORANGE, BLUE, GREEN, YELLOW, PURPLE, CYAN]
    ypos = list(range(len(names)))
    ax.barh(ypos, values, color=colors, height=0.6)
    ax.set_yticks(ypos, names)
    ax.invert_yaxis()
    ax.set_xlim(0, 78)
    ax.set_xlabel("normalized timed-operator share (%)")
    ax.grid(axis="x", color=LINE, linewidth=0.7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(length=0, labelsize=8)
    for y, value in zip(ypos, values):
        ax.text(value + 1.0, y, f"{value:.1f}%", va="center", fontsize=8, fontweight="bold")
    fig.tight_layout()
    save_asset(fig, "fig04_baseline_profile")

    # 05: cumulative optimization log.
    fig, ax = plt.subplots(figsize=(8.3, 4.2), facecolor="white")
    xs = list(range(len(LADDER)))
    ys = [row[1] for row in LADDER]
    ax.plot(xs, ys, color=INK, linewidth=1.5, zorder=2)
    ax.scatter(xs, ys, c=[row[2] for row in LADDER], s=30, edgecolor="white", linewidth=0.6, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(1.6, 55)
    ticks = [0, 2, 3, 6, 9, 13, 16, 18, 20]
    ax.set_xticks(ticks, [LADDER[i][0] for i in ticks], rotation=25, ha="right")
    ax.set_yticks([2, 5, 10, 20, 50], ["2", "5", "10", "20", "50"])
    ax.set_ylabel("seconds / forward · log scale")
    ax.grid(axis="y", color=LINE, linewidth=0.7, which="both")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.tick_params(length=0, labelsize=8)
    ax.annotate("42.15 s", (0, 42.15), xytext=(0.8, 47), color=ORANGE, fontweight="bold", fontsize=8)
    ax.annotate("1.996 s", (20, 1.996), xytext=(17.3, 2.55), color=PURPLE, fontweight="bold", fontsize=8,
                arrowprops=dict(arrowstyle="-", color=PURPLE, linewidth=0.8))
    legend_items = [
        mpl.patches.Patch(color=ORANGE, label="integer numerics"),
        mpl.patches.Patch(color=BLUE, label="GEMM kernels"),
        mpl.patches.Patch(color=GREEN, label="fusion / pass removal"),
        mpl.patches.Patch(color=PURPLE, label="assembly / scheduling"),
        mpl.patches.Patch(color=CYAN, label="integer dataflow"),
    ]
    ax.legend(handles=legend_items, frameon=False, fontsize=6.8, ncol=3, loc="upper right")
    fig.tight_layout()
    save_asset(fig, "fig05_optimization_ladder")

    # 06: kernel result.
    fig, ax = plt.subplots(figsize=(5.3, 3.7), facecolor="white")
    names = ["core4_v2", "core5"]
    values = [7.48, 6.21]
    bars = ax.bar(names, values, color=[FAINT, BLUE], width=0.58)
    ax.set_ylim(0, 8.4)
    ax.set_ylabel("cycles / MAC")
    clean_chart(ax)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, value + 0.18, f"{value:.2f}", ha="center", fontweight="bold")
    fig.tight_layout()
    save_asset(fig, "fig06_gemm_kernel")

    # 07: isolated pass-elimination changes.
    fig, ax = plt.subplots(figsize=(6.5, 3.8), facecolor="white")
    labels = ["fused quant", "fused LayerNorm", "context fusion"]
    before = [6.91, 4.862, 4.160]
    after = [6.56, 4.784, 3.969]
    x = list(range(3))
    for i, (b, a) in enumerate(zip(before, after)):
        ax.plot([i, i], [a, b], color=LINE, linewidth=5, solid_capstyle="round")
        ax.scatter([i], [b], color=FAINT, s=55, zorder=3)
        ax.scatter([i], [a], color=GREEN, s=55, zorder=3)
        ax.text(i + 0.08, b, f"{b:g}", va="center", fontsize=8, color=MUTED)
        ax.text(i + 0.08, a, f"{a:g}", va="center", fontsize=8, color=GREEN, fontweight="bold")
    ax.set_xticks(x, labels)
    ax.set_ylabel("cumulative seconds / forward")
    ax.set_ylim(3.7, 7.15)
    clean_chart(ax)
    fig.tight_layout()
    save_asset(fig, "fig07_fusion_deltas")

    # 08: accuracy gate and observed maximum.
    fig, ax = plt.subplots(figsize=(7.0, 2.5), facecolor="white")
    ax.barh([0], [0.002], color="#E3E8EF", height=0.45)
    ax.barh([0], [0.00146], color=GREEN, height=0.45)
    ax.axvline(0.002, color=RED, linestyle="--", linewidth=1.3)
    ax.text(0.00146, 0, " 1.46e-3 observed", va="center", ha="right", color="white", fontsize=8, fontweight="bold")
    ax.text(0.002, 0.34, "2e-3 absolute tolerance", ha="center", color=RED, fontsize=8, fontweight="bold")
    ax.set_xlim(0, 0.0022)
    ax.set_yticks([])
    ax.set_xlabel("maximum absolute error")
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0, labelsize=8)
    fig.tight_layout()
    save_asset(fig, "fig08_accuracy_gate")

    # 09: WiFi memory fit. Stacks keep static/runtime/free quantities comparable.
    fig, ax = plt.subplots(figsize=(7.0, 3.8), facecolor="white")
    labels = ["untiled · estimated need", "tiled · measured after association"]
    static = [274.56 + 85.50, 173.06]
    runtime = [69.00, 56.24]
    free = [0.00, 98.38]
    y = [0, 1]
    ax.barh(y, static, color=BLUE, height=0.52, label="static model + stack")
    ax.barh(y, runtime, left=static, color=ORANGE, height=0.52, label="WiFi runtime heap")
    ax.barh(y, free, left=[a + b for a, b in zip(static, runtime)], color="#DDE3EA", height=0.52, label="free")
    ax.axvline(327.68, color=RED, linestyle="--", linewidth=1.2)
    ax.text(327.68, 0.96, "327.68 KB linker region", transform=ax.get_xaxis_transform(),
            ha="center", va="top", color=RED, fontsize=8, fontweight="bold")
    ax.text(sum(static[:1]) + runtime[0] + 5, 0, "429 KB need", va="center", fontsize=8, color=RED, fontweight="bold")
    ax.text(static[1] + runtime[1] + free[1] - 3, 1, "98.38 KB free", va="center", ha="right", fontsize=8, color=MUTED, fontweight="bold")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 455)
    ax.set_xlabel("DRAM footprint / capacity · KB")
    ax.grid(axis="x", color=LINE, linewidth=0.7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(length=0, labelsize=8)
    ax.legend(frameon=False, fontsize=7.2, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    fig.tight_layout()
    save_asset(fig, "fig09_wifi_memory_fit")

    # 10: communication throughput.
    fig, ax = plt.subplots(figsize=(6.7, 3.8), facecolor="white")
    names = ["ESP-NOW default\n240 B payload", "ESP-NOW tuned\n240 B payload", "TCP / WiFi best\n4096 B payload", "USB CDC\npaced"]
    values = [61, 197, 723, 50]
    colors = [FAINT, GREEN, BLUE, PURPLE]
    bars = ax.bar(names, values, color=colors, width=0.58)
    ax.set_ylim(0, 790)
    ax.set_ylabel("KB/s")
    clean_chart(ax)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, value + 16, str(value), ha="center", fontweight="bold")
    fig.tight_layout()
    save_asset(fig, "fig10_link_throughput")

    # 11: measured scaling.
    fig, ax = plt.subplots(figsize=(7.4, 4.2), facecolor="white")
    boards = [1, 2, 4, 8]
    ax.plot(boards, boards, linestyle="--", color=FAINT, linewidth=1.2, label="ideal")
    ax.plot(boards, [1, 2, 4, 8], marker="o", color=BLUE, linewidth=2, label="batch DP · B≥8")
    ax.plot(boards, [1, 2, 4, 4], marker="o", color=YELLOW, linewidth=2, label="case 3 · B=4")
    ax.plot([1, 2], [1, 1.56], marker="o", color=ORANGE, linewidth=2, label="case 2 · token rows")
    ax.plot(boards, [1, 1, 1, 1], marker="o", color=PURPLE, linewidth=1.6, label="case 2 · data parallel")
    ax.set_xscale("log", base=2)
    ax.set_xticks(boards, ["1", "2", "4", "8"])
    ax.set_yticks([1, 2, 4, 6, 8], ["1×", "2×", "4×", "6×", "8×"])
    ax.set_ylim(0.7, 8.5)
    ax.set_xlabel("boards")
    ax.set_ylabel("speedup vs each series' one-node baseline")
    ax.grid(color=LINE, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0, labelsize=8)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.text(0.99, 0.12, "device compute wall · host transfer excluded", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7.2, color=MUTED, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=2))
    fig.tight_layout()
    save_asset(fig, "fig11_cluster_scaling")

    # Machine-readable tables for Claude or a human designer.
    with (ASSETS / "optimization_ladder.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["step", "seconds_per_forward", "category_color"])
        writer.writerows(LADDER)
    with (ASSETS / "benchmark_cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["case", "B_S_D_H_F_L", "one_C3_s", "two_C3_s", "eight_C3_wifi_s", "status", "gate"])
        writer.writerows(CASES)


def header(ax, page_number, section, title_value, subtitle=None):
    text(ax, 0.045, 0.952, f"{page_number:02d}", size=8.2, color=ORANGE, weight="bold")
    text(ax, 0.082, 0.952, section.upper(), size=8.2, color=BLUE, weight="bold")
    text(ax, 0.045, 0.902, title_value, size=22, color=INK, weight="bold")
    if subtitle:
        text(ax, 0.045, 0.852, subtitle, size=9.5, color=MUTED)
    rule(ax, 0.045, 0.815, 0.955, 0.815)


def footer(ax, note="Source of truth: COMPETITION_RULES.MD · docs/report/index.html · benchmarks/README.md"):
    rule(ax, 0.045, 0.044, 0.955, 0.044)
    text(ax, 0.045, 0.026, note, size=6.8, color=FAINT, va="center")
    text(ax, 0.955, 0.026, "CLAUDE DESIGN PRODUCTION BRIEF", size=6.8, color=FAINT, weight="bold", ha="right", va="center")


def cover_page(pdf):
    fig, ax = page(WHITE)
    ax.add_patch(Rectangle((0, 0), 0.31, 1, facecolor=NAVY, transform=ax.transAxes))
    text(ax, 0.055, 0.91, "PRODUCTION HANDOFF", size=8.5, color="#87ACFF", weight="bold")
    text(ax, 0.055, 0.82, "Not a slide deck.", size=20, color=WHITE, weight="bold")
    text(ax, 0.055, 0.765, "A build specification.", size=12.5, color="#CBD6E7", weight="bold")
    text(ax, 0.055, 0.69, "Instructions + figures\nfor Claude Design", size=15, color="#CBD6E7", weight="bold")
    text(ax, 0.055, 0.17, "A Transformer on a\n$5 microcontroller", size=15, color=WHITE, weight="bold")
    text(ax, 0.055, 0.095, "Event label is unresolved:\nconfirm before deck generation", size=8.2, color="#93A4BE")

    text(ax, 0.37, 0.90, "What this PDF gives Claude", size=22, color=INK, weight="bold")
    items = [
        ("01", "A fixed 21-slide narrative", "Every slide has one judge takeaway and exact minimal copy."),
        ("02", "Reusable scientific figures", "Eleven charts are supplied as editable SVG and high-resolution PNG."),
        ("03", "Measured-versus-derived discipline", "Labels prevent projections, estimates, and hardware results from being conflated."),
        ("04", "Photo insertion instructions", "Four named slots are reserved for the user's real device and bench photographs."),
    ]
    for i, (num, heading, body) in enumerate(items):
        y = 0.71 - i * 0.15
        text(ax, 0.38, y, num, size=9, color=ORANGE, weight="bold")
        text(ax, 0.44, y, heading, size=13, color=INK, weight="bold")
        text(ax, 0.44, y - 0.045, body, size=8.8, color=MUTED)
    box(ax, 0.37, 0.08, 0.56, 0.11, fc="#FFF4EA", ec="#F0C4A8")
    text(ax, 0.39, 0.16, "OUTPUT REQUEST", size=7.3, color=ORANGE, weight="bold")
    text(ax, 0.39, 0.125, "Create a 16:9 presentation from this specification. Do not redesign this PDF itself.", size=10.2, color=INK, weight="bold")
    text(ax, 0.39, 0.095, "Use supplied figures and replace only the marked photo assets.", size=8.6, color=MUTED)
    save_page(pdf, fig)


def master_prompt_page(pdf):
    fig, ax = page()
    header(ax, 2, "Execution prompt", "Prompt Claude Design with this contract")
    box(ax, 0.055, 0.14, 0.89, 0.62, fc=WHITE, ec=LINE)
    prompt = (
        "Create a 21-slide, 16:9 technical competition presentation titled ‘A Transformer on a $5 microcontroller’. "
        "The audience is an expert HPC jury that already understands attention, quantization, MFU, and parallel scaling. "
        "Use the slide specifications on pages 05–25 exactly: preserve the proposed slide order, exact numerical values, units, "
        "and measured/derived/projected labels. Keep on-slide text sparse: one conclusion headline, at most three short support "
        "statements, and one dominant evidence object. Use the supplied SVG figures without turning them into decorative dashboards. "
        "Use a restrained scientific visual system: off-white or deep-navy fields, high-contrast typography, one orange difficulty "
        "accent, one blue systems accent, and one green verified-result accent. Insert only user-supplied photographs in the four "
        "named photo slots; do not synthesize boards or laboratory scenes. Never imply this is a compliant GPU kernel, never present "
        "a non-run case as a benchmark, never turn correctness equivalence into task-model accuracy, and keep host-transfer exclusions "
        "visible. Add short source notes in 8–10 pt type. Return an editable deck and preserve chart axes."
    )
    text(ax, 0.085, 0.71, wrap(prompt, 132), size=10.2, color=INK, linespacing=1.42)
    label(ax, 0.085, 0.25, "Delivery checklist", GREEN)
    checks = (
        "□ 21 slides, 16:9   □ editable vector figures   □ four real-photo slots   □ no invented results   "
        "□ all axes and units retained   □ measured / derived / projected labels visible"
    )
    text(ax, 0.085, 0.205, wrap(checks, 125), size=9.2, color=MUTED, weight="bold")
    footer(ax)
    save_page(pdf, fig)


def competition_page(pdf):
    fig, ax = page()
    header(ax, 3, "Source of truth", "Competition concept, rules, and evaluation", "These facts control the story; they are not optional design suggestions.")

    box(ax, 0.05, 0.47, 0.275, 0.30)
    label(ax, 0.07, 0.735, "Task")
    text(ax, 0.07, 0.685, wrap("Optimize a fixed pre-LayerNorm Transformer body", 29), size=10.5, weight="bold")
    text(ax, 0.07, 0.605, wrap("Fourteen official (B,S,D,H,F,L) shapes. Output remains [B,S,D]. Final LayerNorm is included.", 38), size=8.2, color=MUTED)
    text(ax, 0.07, 0.515, wrap("Accuracy failure blocks performance scoring.", 34), size=8.2, color=RED, weight="bold")

    box(ax, 0.36, 0.47, 0.275, 0.30)
    label(ax, 0.38, 0.735, "Elementwise gate", GREEN)
    text(ax, 0.38, 0.675, "|user − ref| ≤ 0.002", size=15, color=GREEN, weight="bold")
    text(ax, 0.38, 0.615, "OR", size=8, color=ORANGE, weight="bold")
    text(ax, 0.38, 0.575, "|user − ref| ≤ 0.02 |ref|", size=15, color=GREEN, weight="bold")
    text(ax, 0.38, 0.515, wrap("Official evaluation: 5 trials. Project validation uses stronger seeded gates where reported.", 40), size=7.8, color=MUTED)

    box(ax, 0.67, 0.47, 0.28, 0.30)
    label(ax, 0.69, 0.735, "Judging weights", ORANGE)
    weights = [("Technical", 35, BLUE), ("Innovation", 20, ORANGE), ("Impact", 20, GREEN), ("Feasibility", 15, PURPLE), ("Presentation", 10, CYAN)]
    y = 0.68
    for name, value, color in weights:
        text(ax, 0.69, y, name, size=8.5, color=MUTED, weight="bold")
        ax.add_patch(Rectangle((0.77, y - 0.013), 0.13 * value / 35, 0.018, facecolor=color, transform=ax.transAxes))
        text(ax, 0.92, y, f"{value}%", size=8.5, color=color, weight="bold", ha="right")
        y -= 0.045

    box(ax, 0.05, 0.14, 0.90, 0.24, fc="#EEF3FA", ec="#C7D5EA")
    label(ax, 0.07, 0.345, "Framing that must stay honest", BLUE)
    statements = [
        "This project preserves the model semantics and numerical gate, but it is not a GPU-kernel submission lane.",
        "Performance means complete forward time on physical ESP32-C3 hardware unless the label says host-only, estimate, or projection.",
        "The implementation uses seeded random weights for equivalence testing; do not claim trained-model application accuracy.",
    ]
    for i, statement in enumerate(statements):
        text(ax, 0.075, 0.292 - i * 0.055, f"{i+1}.", size=8.5, color=ORANGE, weight="bold")
        text(ax, 0.105, 0.292 - i * 0.055, statement, size=8.6, color=INK)
    footer(ax, "Competition source: COMPETITION_RULES.MD · event-name discrepancy must be resolved before submission")
    save_page(pdf, fig)


def design_system_page(pdf):
    fig, ax = page()
    header(ax, 4, "Design system", "Make it look like an HPC result, not an AI template")
    cols = [0.05, 0.37, 0.69]
    titles = ["HIERARCHY", "FIGURES", "PHOTOGRAPHS"]
    bodies = [
        [
            "Conclusion headline: 32–40 pt",
            "Support copy: 18–22 pt",
            "Source/caveat: 8–10 pt",
            "≤35 on-slide words, excluding labels",
            "One visual argument per slide",
        ],
        [
            "Use supplied SVG first; PNG is fallback",
            "No 3D bars, gauges, or fake telemetry",
            "Keep axes, units, and log-scale labels",
            "Direct-label important values",
            "Orange=difficulty; blue=system; green=pass",
        ],
        [
            "photo_01_hero_device",
            "photo_02_board_closeup",
            "photo_03_cluster_wide",
            "photo_04_profiler_bench",
            "Crop documentary, not cinematic",
        ],
    ]
    for x, heading, rows in zip(cols, titles, bodies):
        box(ax, x, 0.39, 0.27, 0.36)
        label(ax, x + 0.02, 0.71, heading, [BLUE, GREEN, ORANGE][titles.index(heading)])
        for i, row in enumerate(rows):
            text(ax, x + 0.025, 0.655 - i * 0.056, "•", size=9, color=ORANGE, weight="bold")
            text(ax, x + 0.047, 0.655 - i * 0.056, wrap(row, 29), size=7.8, color=INK)

    box(ax, 0.05, 0.14, 0.91, 0.18, fc=WHITE)
    label(ax, 0.07, 0.285, "Asset bundle", GREEN)
    text(ax, 0.07, 0.245, "figure_assets/fig01…fig11", size=10, color=INK, weight="bold")
    text(ax, 0.27, 0.245, "SVG + 240 dpi PNG", size=9, color=MUTED)
    text(ax, 0.47, 0.245, "optimization_ladder.csv", size=9, color=INK, weight="bold")
    text(ax, 0.68, 0.245, "benchmark_cases.csv", size=9, color=INK, weight="bold")
    text(ax, 0.07, 0.195, "Do not trace charts from PDF screenshots. Import the SVG or use the CSV to rebuild them with the same scale and labels.", size=8.7, color=MUTED)
    footer(ax)
    save_page(pdf, fig)


def draw_composition(ax, spec):
    """Small structural diagram: visibly a wireframe, never a slide mock-up."""
    x, y, w, h = 0.39, 0.29, 0.28, 0.27
    box(ax, x, y, w, h, fc="#F1F3F6", ec="#B9C1CD", radius=0.004)
    text(ax, x + 0.012, y + h - 0.012, "WIREFRAME ONLY", size=5.8, color=FAINT, weight="bold")
    layout = spec.get("layout", "visual_right")
    if layout == "photo_full":
        ax.add_patch(Rectangle((x + 0.015, y + 0.015), w - 0.03, h - 0.07, facecolor="#D9DEE6", edgecolor="#AAB3C0", transform=ax.transAxes))
        text(ax, x + w/2, y + h/2, "USER PHOTO", size=8, color=FAINT, weight="bold", ha="center", va="center")
        ax.add_patch(Rectangle((x + 0.02, y + h - 0.055), w * 0.58, 0.025, facecolor=NAVY, transform=ax.transAxes))
    elif layout == "two_panel":
        ax.add_patch(Rectangle((x + 0.02, y + 0.035), w * 0.42, h * 0.63, facecolor=WHITE, edgecolor=LINE, transform=ax.transAxes))
        ax.add_patch(Rectangle((x + w * 0.52, y + 0.035), w * 0.43, h * 0.63, facecolor=WHITE, edgecolor=LINE, transform=ax.transAxes))
        ax.add_patch(Rectangle((x + 0.02, y + h - 0.055), w * 0.72, 0.025, facecolor=NAVY, transform=ax.transAxes))
    elif layout == "table":
        ax.add_patch(Rectangle((x + 0.02, y + 0.03), w - 0.04, h * 0.64, facecolor=WHITE, edgecolor=LINE, transform=ax.transAxes))
        for i in range(5):
            rule(ax, x + 0.025, y + 0.05 + i * 0.032, x + w - 0.025, y + 0.05 + i * 0.032, color=LINE, lw=0.5)
        ax.add_patch(Rectangle((x + 0.02, y + h - 0.055), w * 0.62, 0.025, facecolor=NAVY, transform=ax.transAxes))
    elif layout == "diagram":
        for i in range(3):
            ax.add_patch(Rectangle((x + 0.025 + i * 0.078, y + 0.095), 0.055, 0.055, facecolor=WHITE, edgecolor=BLUE, transform=ax.transAxes))
        ax.add_patch(Rectangle((x + 0.02, y + h - 0.055), w * 0.68, 0.025, facecolor=NAVY, transform=ax.transAxes))
    else:
        ax.add_patch(Rectangle((x + 0.02, y + h - 0.055), w * 0.65, 0.025, facecolor=NAVY, transform=ax.transAxes))
        ax.add_patch(Rectangle((x + 0.02, y + 0.04), w * 0.27, h * 0.55, facecolor=WHITE, edgecolor=LINE, transform=ax.transAxes))
        ax.add_patch(Rectangle((x + w * 0.34, y + 0.04), w * 0.61, h * 0.55, facecolor="#DCE5F3", edgecolor="#B5C6DF", transform=ax.transAxes))


def slide_spec_page(pdf, page_number, spec):
    fig, ax = page()
    header(
        ax,
        page_number,
        f"Slide {spec['number']:02d} / {spec['section']}",
        spec["title"],
        f"Judge takeaway: {spec['takeaway']}",
    )

    # Left: exact words that belong on the eventual slide.
    box(ax, 0.045, 0.105, 0.31, 0.66, fc=WHITE)
    label(ax, 0.065, 0.725, "Exact on-slide copy", ORANGE)
    headline = wrap(spec["headline"], 28)
    headline_lines = headline.count("\n") + 1
    text(ax, 0.065, 0.675, headline, size=13.2, color=INK, weight="bold")
    y = 0.59 - (headline_lines - 1) * 0.047
    for item in spec["copy"]:
        text(ax, 0.07, y, "•", size=9, color=ORANGE, weight="bold")
        text(ax, 0.09, y, wrap(item, 41), size=8.7, color=INK)
        y -= 0.07 if len(item) < 60 else 0.09
    if spec.get("badge"):
        box(ax, 0.065, 0.145, 0.25, 0.07, fc="#ECF7F2", ec="#B8DDCE")
        text(ax, 0.19, 0.18, spec["badge"], size=9.5, color=GREEN, weight="bold", ha="center", va="center")

    # Middle: production construction, with a wireframe and optional asset preview.
    label(ax, 0.39, 0.725, "Composition instruction", BLUE)
    text(ax, 0.39, 0.685, wrap(spec["composition"], 48), size=8.5, color=INK)
    draw_composition(ax, spec)
    asset = spec.get("asset")
    if asset:
        asset_path = ASSETS / f"{asset}.png"
        preview = fig.add_axes([0.405, 0.115, 0.25, 0.14], facecolor=WHITE)
        preview.imshow(plt.imread(asset_path))
        preview.axis("off")
        text(ax, 0.39, 0.265, f"ASSET PREVIEW · {asset}.svg", size=6.5, color=GREEN, weight="bold")
    else:
        box(ax, 0.405, 0.12, 0.25, 0.125, fc="#EEF1F5", ec="#C6CDD7")
        text(ax, 0.53, 0.182, spec.get("slot", "NO EXTERNAL ASSET"), size=7.3, color=FAINT, weight="bold", ha="center", va="center")

    # Right: the content/evidence controls.
    box(ax, 0.70, 0.105, 0.255, 0.66, fc="#F1F4F8", ec="#D1D8E2")
    label(ax, 0.72, 0.725, "Data / evidence", GREEN)
    text(ax, 0.72, 0.685, wrap(spec["evidence"], 37), size=8.25, color=INK)
    label(ax, 0.72, 0.44, "Must preserve", PURPLE)
    text(ax, 0.72, 0.40, wrap(spec["preserve"], 37), size=8.2, color=MUTED)
    label(ax, 0.72, 0.245, "Source path", BLUE)
    text(ax, 0.72, 0.205, wrap_source(spec["source"], 34), size=6.7, color=MUTED)
    footer(ax, "This page instructs Claude Design; it is not a candidate slide layout")
    save_page(pdf, fig)


SLIDES = [
    {
        "number": 1, "section": "Title", "title": "Open with the physical claim",
        "takeaway": "The hardware constraint is the hook; the result is real and measured.",
        "headline": "A Transformer on a $5 microcontroller",
        "copy": ["ESP32-C3 · 160 MHz · no FPU · 321 KB usable SRAM", "[EVENT NAME — CONFIRM] · team / affiliation"],
        "badge": "21.1× one-board speedup",
        "composition": "Use the user's strongest close-up device photograph as the dominant field. Put the title in clear negative space; keep only one result badge.",
        "layout": "photo_full", "slot": "PHOTO · photo_01_hero_device",
        "evidence": "Baseline case 2: 42.15 s. Optimization-log endpoint: 1.996 s; authoritative repeat median: 1.990 s.",
        "preserve": "Do not use a synthetic chip image. Do not imply the $5 price is a formal bill of materials.",
        "source": "docs/report/index.html · benchmarks/README.md",
    },
    {
        "number": 2, "section": "Introduction", "title": "State the deployment vision without overselling it",
        "takeaway": "Local inference on tiny endpoints is valuable because connectivity, power, and hardware budgets are finite.",
        "headline": "What if every sensor could run its own Transformer?",
        "copy": ["Local inference: private, resilient, low-latency", "Constraint: consumer- and industrial-grade accelerators do not fit every endpoint", "Potential: offline industrial sensing and private local monitoring"],
        "composition": "Use a three-node edge-to-cloud schematic: sensor → local decision → optional cloud. Make the local path solid and the cloud path secondary/dashed.",
        "layout": "diagram",
        "evidence": "This is motivation, not a demonstrated application workload. The benchmark uses seeded random weights and a fixed Transformer body.",
        "preserve": "Use the words ‘potential’ or ‘vision’. Do not claim energy, privacy, or latency measurements that are absent.",
        "source": "COMPETITION_RULES.MD · docs/report/index.html",
    },
    {
        "number": 3, "section": "Thesis", "title": "Define exactly what the project proves",
        "takeaway": "The contribution is systems co-design under extreme constraints, not a smaller model architecture.",
        "headline": "We kept the Transformer. We changed the execution system.",
        "copy": ["Fixed-point attention and integer residuals", "SRAM-lifetime scheduling and register-tiled RV32 kernels", "Two distributed execution modes across eight boards"],
        "badge": "10 / 14 official shapes measured",
        "composition": "Use three equal blocks—Numerics, Memory, Parallelism—feeding one verified-output block. This is the map for the rest of the talk.",
        "layout": "diagram",
        "evidence": "All reported hardware results preserve the elementwise competition gate. Four official shapes are explicitly not device runs.",
        "preserve": "Say ‘Transformer body’ rather than foundation model or LLM. No trained-model claims.",
        "source": "COMPETITION_RULES.MD · docs/report/index.html",
    },
    {
        "number": 4, "section": "Hardware", "title": "Introduce the exact constraint surface",
        "takeaway": "The ESP32-C3 is a scalar RISC-V microcontroller with neither accelerator nor FPU.",
        "headline": "ESP32-C3: a deliberately hostile Transformer target",
        "copy": ["RV32IMC · single core · 160 MHz", "No floating-point unit", "321,296 B usable DRAM segment · 4 MB flash · 3.1 MB app partition"],
        "composition": "Place the user's board close-up on the left. On the right, use four technical callouts anchored to the real components; avoid lifestyle-card styling.",
        "layout": "two_panel", "slot": "PHOTO · photo_02_board_closeup",
        "evidence": "Usable SRAM is the linked dram0_0_seg, not the marketing SRAM total. Board: Seeed XIAO ESP32-C3.",
        "preserve": "If comparing to ESP32-S3, state that S3 has an FPU and larger memory options; do not imply it is the tested device.",
        "source": "docs/esp32_implementation_summary.md · docs/HARDWARE_COMPARISON.md",
    },
    {
        "number": 5, "section": "Difficulty 1", "title": "Explain why no FPU changes the algorithm",
        "takeaway": "Every fp32 inner-loop operation becomes software work; fixed-point is an architectural requirement.",
        "headline": "Without an FPU, fp32 becomes a function call",
        "copy": ["C3 fp32: add 100 cycles · divide 102 · cosf 2,377", "S3 hardware FPU: 25 · 69 · 121 cycles", "Baseline attention consumed 30.09 s of a 42.15 s forward"],
        "composition": "Use the log-scale C3-versus-S3 chart as the dominant object. Add one small annotation connecting software fp32 to attention's baseline wall time.",
        "layout": "visual_right", "asset": "fig01_fpu_cost_c3_vs_s3",
        "evidence": "Cycle figures are documented Espressif measurements quoted in the repository. The C3 has RV32IMC, not the F extension.",
        "preserve": "Q15 multiplication uses a native integer instruction; do not claim a universal one-cycle latency.",
        "source": "benchmarks/case-02/optimisation/esp32-baseline/optimisations/research.md · docs/esp32_fastest_kernels_research.md",
    },
    {
        "number": 6, "section": "Difficulty 2", "title": "Make the memory scale physically legible",
        "takeaway": "A single tensor can exceed the entire machine; algorithmic tiling is mandatory.",
        "headline": "321 KB is not a memory budget. It is a scheduling constraint.",
        "copy": ["Final case-2 arena: 274,340 B used · 46,956 B free", "Case 13 one activation: 524 KB", "Case 14 full-batch KV state: 26.2 GB"],
        "composition": "Use the log-scale memory figure as the main visual and the final SRAM stack as a small inset. Keep budget bars visually distinct from requirements.",
        "layout": "visual_right", "asset": "fig02_memory_scale",
        "evidence": "Secondary inset: figure_assets/fig03_final_sram_budget.svg. The earlier 384 B-free build is a different intermediate state and must not be mixed in.",
        "preserve": "Always show the logarithmic axis label. Distinguish SRAM, flash/app partition, weights, and activations.",
        "source": "docs/report/index.html · benchmarks/case-02/optimisation/esp32-baseline/optimisations/25_tinyprof.md",
    },
    {
        "number": 7, "section": "Method", "title": "Show that profiling—not intuition—selected the work",
        "takeaway": "Attention was the measured bottleneck before optimization.",
        "headline": "The baseline profile told us where not to guess",
        "copy": ["42.15 s complete case-2 forward", "30.09 s attention wall time · 71.4%", "512k expf calls plus fp32 QK dequantization and PV accumulation"],
        "composition": "Use the horizontal profile chart. Keep the 70.6% normalized operator share in the chart and the 71.4% raw wall ratio in a separate annotation.",
        "layout": "visual_right", "asset": "fig04_baseline_profile",
        "evidence": "Normalized timed-zone shares sum to about 100%; raw wall ratio is 30.09/42.15. They are related but not interchangeable.",
        "preserve": "Label the normalization difference rather than rounding both to one number.",
        "source": "benchmarks/case-02/optimisation/esp32-baseline/optimisations/00_baseline_profile.md · docs/report/index.html",
    },
    {
        "number": 8, "section": "Optimization overview", "title": "Present the work as an evidence-backed sequence",
        "takeaway": "The 21× speedup came from accumulated numerical, kernel, fusion, and assembly changes.",
        "headline": "No silver bullet: 24 engineering steps",
        "copy": ["42.15 s → 1.996 s across 24 steps grouped into 21 recorded milestones", "Numerics → kernels → fusion → hand-written RV32 assembly", "MFU: 2.0% → 42.2% of the derived scalar peak"],
        "badge": "21.1× faster",
        "composition": "Use the optimization ladder full-width. Direct-label only the start, major phase boundaries, and endpoint; do not annotate every point.",
        "layout": "visual_right", "asset": "fig05_optimization_ladder",
        "evidence": "The graph is cumulative, not an ablation. The authoritative repeat median is 1.990 s; keep it as a note, not a second endpoint.",
        "preserve": "Keep the y-axis logarithmic and distinguish 1.996 optimization endpoint from 1.990 repeat median.",
        "source": "docs/report/index.html · benchmarks/case-02/optimisation/README.md",
    },
    {
        "number": 9, "section": "Optimization 1", "title": "Explain fixed-point attention as scale factoring",
        "takeaway": "Quantization moved high-volume QK and PV work to integer arithmetic while retaining a stable rowwise softmax.",
        "headline": "Move attention into fixed point—without moving the error gate",
        "copy": ["QK: int64 Q15 dot products", "Softmax: LUT plus one fp32 scale per logit at the 13.70 s milestone", "PV: integer accumulation with fp32 only at row end; later made fully integer"],
        "composition": "Draw a before/after attention pipeline. Before: fp32 QK → expf → fp32 PV. After: Q15 QK → LUT softmax → Q15 PV. Put scale factors above arrows, not in prose.",
        "layout": "two_panel",
        "evidence": "Cumulative timing: 42.15 → 15.21 s after integer QK, then 13.70 s after first exp LUT. Integer PV lands later: 3.969 → 3.688 s.",
        "preserve": "Keep chronology exact. Do not describe the 13.70 s milestone as fully integer attention.",
        "source": "benchmarks/case-02/optimisation/README.md · docs/report/index.html",
    },
    {
        "number": 10, "section": "Optimization 2", "title": "Make the GEMM result about registers, not generic tiling",
        "takeaway": "A 4×2 accumulator tile balanced reuse and register pressure; larger tiles spilled.",
        "headline": "Register pressure—not tile area—set GEMM speed",
        "copy": ["core4_v2: 7.48 cycles/MAC", "core5: 6.21 cycles/MAC", "~20 live registers; 4×4 regressed from spills"],
        "composition": "Pair a simple 4×2 accumulator-grid diagram with the two-bar measured kernel comparison. Make the failed 4×4 attempt a small gray annotation.",
        "layout": "two_panel", "asset": "fig06_gemm_kernel",
        "evidence": "Use the comparable core4_v2 → core5 microbenchmark pair. Other cumulative forward timings are not the same measurement.",
        "preserve": "Say cycles/MAC, not FLOP/s. Mention the unsuccessful larger tile to demonstrate engineering rigor.",
        "source": "benchmarks/case-02/optimisation/esp32-baseline/optimisations/15_gemm_core5_iblk4.md",
    },
    {
        "number": 11, "section": "Optimization 3", "title": "Connect fusion and integer residuals to eliminated passes",
        "takeaway": "SRAM traffic fell because intermediate tensors stopped being materialized and rescanned.",
        "headline": "The fastest memory pass is the one the schedule deletes",
        "copy": ["Fuse producer → scale bound → Q15 consumer", "Delete tensor write → amax scan → requantization passes", "Integer residual path: 268 → 10 instructions/element"],
        "composition": "Use a before/after dataflow with crossed-out write, amax, and requantize passes. Add the isolated timing deltas as a small evidence panel.",
        "layout": "two_panel", "asset": "fig07_fusion_deltas",
        "evidence": "Isolated cumulative steps: 6.91→6.56 fused quant; 4.862→4.784 fused LN; 4.160→3.969 context fusion.",
        "preserve": "These deltas occur at different points in the cumulative sequence; do not add them as an independent total.",
        "source": "benchmarks/case-02/optimisation/README.md · docs/report/index.html",
    },
    {
        "number": 12, "section": "Correctness", "title": "Put the numerical gate before the performance claim",
        "takeaway": "Every reported speed result was accepted only after the elementwise equivalence test passed.",
        "headline": "Speed counts only after every element passes",
        "copy": ["Pass if absolute error ≤0.002 OR relative error ≤2%", "Case 2 device gate: 25/25 seeds", "Worst observed device absolute error across measured cases: 1.46e−3"],
        "badge": "zero failing elements in reported gates",
        "composition": "Use the tolerance bar plus the Boolean gate written once as a compact equation. Keep ‘seeded random weights’ in the footer.",
        "layout": "visual_right", "asset": "fig08_accuracy_gate",
        "evidence": "Official rules specify 5 accuracy trials. The project reports stronger per-case gates; do not generalize 25 seeds to every row unless the table says so.",
        "preserve": "This is numerical equivalence, not task accuracy on a trained model.",
        "source": "COMPETITION_RULES.MD · benchmarks/case-02/README.md · benchmarks/README.md",
    },
    {
        "number": 13, "section": "Scale-out transition", "title": "Use the real cluster as the narrative reset",
        "takeaway": "After optimizing one core, the next limit was coordinated execution across eight memory-starved nodes.",
        "headline": "Then we built an eight-node cluster",
        "copy": ["Eight ESP32-C3 boards", "Two decompositions: batch data parallel + token-row split", "One shared problem: communication must coexist with the model arena"],
        "composition": "Make photo_03_cluster_wide nearly full bleed. Use only the headline and three small technical labels anchored to boards/router/wiring.",
        "layout": "photo_full", "slot": "PHOTO · photo_03_cluster_wide",
        "evidence": "The cluster uses real WiFi/TCP/UDP paths described in the report; this is not a simulated topology.",
        "preserve": "Show the actual eight-board setup. No network-globe illustration.",
        "source": "docs/MULTI_ESP32_DESIGN.md · docs/report/index.html",
    },
    {
        "number": 14, "section": "Difficulty 3", "title": "Explain how WiFi broke the memory budget",
        "takeaway": "The team rewrote tensor lifetimes so networking and inference could fit on the same board.",
        "headline": "A node could compute—or communicate—but not both",
        "copy": ["Full forward + WiFi need: ~429 KB vs 327.68 KB linker region", "Rewrite: 16-row tiles + one attention head at a time", "Tiled + WiFi static: 173.06 KB · 98.38 KB heap after association"],
        "composition": "Use the three-bar memory fit chart and a small schedule strip showing row tiles/head-at-a-time. Make the ~100 KB deficit visible.",
        "layout": "visual_right", "asset": "fig09_wifi_memory_fit",
        "evidence": "Tiling makes WiFi possible but costs performance: 4.215 s tiled versus 1.990 s best untiled single board.",
        "preserve": "Use 327.68 KB for this WiFi linker-region comparison; separately reported optimized map uses 321,296 B usable.",
        "source": "benchmarks/case-02/optimisation/esp32-baseline/optimisations/24_sequence_tiled_wifi.md",
    },
    {
        "number": 15, "section": "Communication", "title": "Show why the final result is an overlap story",
        "takeaway": "The team chose WiFi for fan-out and reduced exposed wait by streaming K/V per head.",
        "headline": "Bandwidth mattered. Overlap mattered more.",
        "copy": ["TCP/WiFi link test peak: 723 KB/s", "First blocking TCP wait: 1.83 s", "Final UDP + NAK cluster wait: 5–88 ms"],
        "composition": "Use the throughput chart on the left and a projection/transfer/attention timeline on the right. Overlap the transfer bar with compute; do not imply zero communication.",
        "layout": "two_panel", "asset": "fig10_link_throughput",
        "evidence": "Why WiFi: one host reaches all eight nodes; only two usable host USB ports; paced USB CDC measured about 50 KB/s.",
        "preserve": "Throughput bars are a dedicated link benchmark. Wait-time numbers come from different implementation stages.",
        "source": "benchmarks/case-02/multiboard/esp32-linkbench/README.md · docs/WIFI_ON_A_COMPUTE_NODE.md",
    },
    {
        "number": 16, "section": "Parallelism", "title": "Teach the two decompositions with one comparison",
        "takeaway": "Data parallelism is communication-free during the forward, while B=1 requires causal token-row work sharing.",
        "headline": "Two parallel decompositions for two batch regimes",
        "copy": ["Batch DP: speedup = min(B,N); no inter-node forward messages", "Token-row split: even/odd causal query rows across two boards", "Parity imbalance: 1.6% vs 3× for a contiguous split"],
        "composition": "Use two equal technical diagrams. Left: independent batch samples to boards. Right: an S×S causal attention triangle colored in alternating query rows, with bidirectional K/V exchange.",
        "layout": "two_panel",
        "evidence": "Token-row mode exchanges 131,200 B per board in each direction per forward; weights remain replicated.",
        "preserve": "K and V move between peers; queries and assigned output rows remain local. Do not call token-row split tensor parallelism.",
        "source": "docs/report/index.html · docs/MULTI_ESP32_DESIGN.md",
    },
    {
        "number": 17, "section": "Scaling", "title": "Report both honest baselines",
        "takeaway": "Eight nodes scale ideally for sufficient batch, but the WiFi-capable single node is slower than the best untiled node.",
        "headline": "8.00× node scaling; 3.78× compute-wall gain",
        "copy": ["8.00× vs one tiled WiFi node", "3.78× compute wall vs best untiled single board", "Case 2 token-row split: 1.56× on two boards · host transfer excluded"],
        "badge": "213 / 213 WiFi forwards pass · cases 1–5",
        "composition": "Use the measured scaling plot. Put the two baseline definitions in separate callouts of equal visual weight.",
        "layout": "visual_right", "asset": "fig11_cluster_scaling",
        "evidence": "Per-device MFU: best opt23 42.2%; token-row split 32.9%; tiled WiFi replicas 19.9%.",
        "preserve": "Case 2 data parallel stays at 1× because B=1. Case 3 saturates at 4× because B=4.",
        "source": "docs/report/index.html · benchmarks/README.md",
    },
    {
        "number": 18, "section": "Coverage", "title": "Give the jury the complete benchmark accounting",
        "takeaway": "Ten official shapes have physical-device measurements; status and numerical gate are separate facts.",
        "headline": "10 of 14 official shapes ran on hardware",
        "copy": ["Complete-case seconds; host transfer excluded", "One C3 · two C3 · eight-node WiFi", "Measured / partial / not run shown explicitly"],
        "composition": "Build a compact 14-row table from benchmark_cases.csv. Use seven columns: case, shape, 1 C3, 2 C3, 8 C3 WiFi, status, gate. Highlight case 2 only.",
        "layout": "table",
        "evidence": "Footnotes: * B=1 activates one DP node. ** B=4 activates four nodes. Case 6 is host-only/partial; 8,13,14 are not run.",
        "preserve": "Do not put dashes in a green PASS style. Do not merge status and gate. Keep all time values in seconds.",
        "source": "figure_assets/benchmark_cases.csv · docs/report/index.html",
    },
    {
        "number": 19, "section": "Limits", "title": "Turn non-running cases into a precise systems boundary",
        "takeaway": "Each failure mode has a different remedy; hiding them would weaken the engineering story.",
        "headline": "Four cases do not run—and the arithmetic says why",
        "copy": ["Case 6: time scale · host gate only", "Cases 8/13: 50.4 MB int16 weights / 524 KB activation exceed flash / SRAM", "Case 14: 26.2 GB full-batch KV state"],
        "composition": "Use four horizontal rows, not four decorative cards. Each row must be formatted as requirement → board limit → next architectural step.",
        "layout": "table", "asset": "fig02_memory_scale",
        "evidence": "Next steps: case 8 weight sharding; case 13 activation tiling + online attention; case 14 requires a different machine/system scale.",
        "preserve": "No device runtime for cases 6/8/13/14. Do not present projected board counts as achieved scaling.",
        "source": "docs/report/index.html · benchmarks/README.md",
    },
    {
        "number": 20, "section": "Tooling contribution", "title": "Show the profiler as part of the contribution",
        "takeaway": "tinyprof made cycle, call-tree, and static-memory bottlenecks reproducible on the C3.",
        "headline": "We built the measurement tool this port was missing",
        "copy": ["6.25 ns cycle-counter zones + nested-call correction", "ELF census found a missing 32 KB scratch buffer", "[INSERT DEVICE TIMING COMPARISON AFTER CAPTURE]"],
        "composition": "Use a four-stage pipeline: cycle zones → call tree → ELF census → report. Reserve the right third for photo_04_profiler_bench.",
        "layout": "two_panel", "slot": "PHOTO · photo_04_profiler_bench",
        "evidence": "Build/host findings: 768 KiB weight traffic and 9,216 B LayerNorm reads. Device captures are still marked outstanding in the tinyprof log; leave the result panel as a memo until captured.",
        "preserve": "Before submission, add the exact AI tool model/version and the prompt → host gate → device profile → accept/revert workflow.",
        "source": "benchmarks/case-02/optimisation/esp32-baseline/optimisations/25_tinyprof.md",
    },
    {
        "number": 21, "section": "Summary", "title": "Close on contributions and the honest frontier",
        "takeaway": "The result is a complete numerical, kernel, memory, and distributed-systems co-design for a severe MCU target.",
        "headline": "A tiny machine forced better systems thinking",
        "copy": ["21.1× one-board case-2 speedup", "1.276 s two-board token-row forward", "10/14 official shapes measured · zero failing elements across reported gates"],
        "composition": "Use four large numeric outcomes, then one restrained line of next steps: online long-sequence attention · weight sharding · recover tiled-node MFU.",
        "layout": "visual_right",
        "evidence": "Technical contributions: fixed-point attention, integer residual dataflow, RV32 kernels, SRAM-aware WiFi execution, two parallel modes, and tinyprof.",
        "preserve": "Add an explicit scope footer: device compute wall; host transfer excluded; seeded random weights; four cases without device runs.",
        "source": "docs/report/index.html · benchmarks/README.md",
    },
]


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    generate_assets()
    with PdfPages(OUT) as pdf:
        cover_page(pdf)
        master_prompt_page(pdf)
        competition_page(pdf)
        design_system_page(pdf)
        for page_number, spec in enumerate(SLIDES, start=5):
            slide_spec_page(pdf, page_number, spec)
    print(f"wrote {OUT}")
    print(f"wrote {len(list(ASSETS.glob('fig*.svg')))} SVG figures and {len(list(ASSETS.glob('fig*.png')))} PNG figures")


if __name__ == "__main__":
    main()
