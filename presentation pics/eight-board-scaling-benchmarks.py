"""Plot baseline, optimized, and eight-board throughput on a linear scale.

The metric is effective device-compute seconds per completed input:
complete-case compute wall divided by batch size. This is the inverse of
throughput, not single-request latency. It keeps cases with different batch
sizes comparable while preserving a linear axis.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).with_suffix("")


def load_result(path: str, batch: int | None = None) -> dict:
    """Load one result record, optionally selecting a batch from a list."""
    records = json.loads((ROOT / path).read_text(encoding="utf-8"))
    if batch is None:
        if len(records) != 1:
            raise ValueError(f"Expected one record in {path}")
        return records[0]
    matches = [record for record in records if record.get("batch") == batch]
    if len(matches) != 1:
        raise ValueError(f"Expected one batch={batch} record in {path}")
    return matches[0]


# Cases 01-05 share the exact S=128/D=128/H=4/F=128/L=4 forward. Use the
# canonical physical seed-0 baseline reading verbatim rather than averaging:
# benchmarks/case-02/baseline/results/teammate_esp32_baseline_seed0_v1.log
# reports TM 1 42124528 us (displayed as 42.152 s). B changes only how many
# independent forwards constitute the complete case.
BASELINE_01_05_S = 42.124528

# Cases 09/10 could not link the original baseline because it exceeded SRAM.
# Their estimates begin with the equal-FLOP H=4 measured baseline and apply
# the observed optimized head-count penalty relative to Case 01:
# baseline_est = BASELINE_01_05_S * Topt(case) / Topt(case 01).
rows = [
    {
        "case": "01", "batch": 64, "baseline_per_input": BASELINE_01_05_S,
        "baseline_estimated": False, "optimized_total": 127.36, "active": 8,
        "eight_path": "benchmarks/batch-dp/results_cases_1_3_4_5_eight_c3_wifi.json",
    },
    {
        "case": "02", "batch": 1, "baseline_per_input": BASELINE_01_05_S,
        "baseline_estimated": False, "optimized_total": 1.990, "active": 1,
        "eight_path": "benchmarks/batch-dp/results_case2_one_active_eight_available_c3_wifi.json",
    },
    {
        "case": "03", "batch": 4, "baseline_per_input": BASELINE_01_05_S,
        "baseline_estimated": False, "optimized_total": 7.96, "active": 4,
        "eight_path": "benchmarks/batch-dp/results_case3_four_active_eight_available_c3_wifi.json",
    },
    {
        "case": "04", "batch": 16, "baseline_per_input": BASELINE_01_05_S,
        "baseline_estimated": False, "optimized_total": 31.84, "active": 8,
        "eight_path": "benchmarks/batch-dp/results_cases_1_3_4_5_eight_c3_wifi.json",
    },
    {
        "case": "05", "batch": 128, "baseline_per_input": BASELINE_01_05_S,
        "baseline_estimated": False, "optimized_total": 254.72, "active": 8,
        "eight_path": "benchmarks/batch-dp/results_cases_1_3_4_5_eight_c3_wifi.json",
    },
    {
        "case": "07", "batch": 64, "baseline_per_input": 0.491,
        "baseline_estimated": False, "optimized_total": 30.4272, "active": 8,
        "eight_path": "benchmarks/case-07/multiboard/results_case7_eight_c3_wifi.json",
    },
    {
        "case": "09", "batch": 64,
        "baseline_per_input": BASELINE_01_05_S * (138.0273 / 127.36),
        "baseline_estimated": True, "optimized_total": 138.0273, "active": 8,
        "eight_path": "benchmarks/case-09/multiboard/results_case9_eight_c3_wifi.json",
    },
    {
        "case": "10", "batch": 64,
        "baseline_per_input": BASELINE_01_05_S * (138.5358 / 127.36),
        "baseline_estimated": True, "optimized_total": 138.5358, "active": 8,
        "eight_path": "benchmarks/case-10/multiboard/results_case10_eight_c3_wifi.json",
    },
    {
        "case": "11", "batch": 64, "baseline_per_input": 2.462,
        "baseline_estimated": False, "optimized_total": 138.6104, "active": 8,
        "eight_path": "benchmarks/case-11/multiboard/results_case11_eight_c3_wifi.json",
    },
    {
        "case": "12", "batch": 64, "baseline_per_input": 0.493,
        "baseline_estimated": False, "optimized_total": 33.8794, "active": 8,
        "eight_path": "benchmarks/case-12/multiboard/results_case12_eight_c3_wifi.json",
    },
]

for row in rows:
    result = load_result(row["eight_path"], row["batch"])
    row["optimized_per_input"] = row["optimized_total"] / row["batch"]
    row["eight_total"] = result["compute_wall_s"]
    row["eight_per_input"] = row["eight_total"] / row["batch"]
    row["optimized_speedup"] = row["baseline_per_input"] / row["optimized_per_input"]
    row["optimized_to_eight"] = row["optimized_total"] / row["eight_total"]
    row["baseline_to_eight"] = row["baseline_per_input"] / row["eight_per_input"]

labels = [row["case"] for row in rows]
x = np.arange(len(rows))

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titleweight": "bold",
})

fig, ax = plt.subplots(figsize=(16, 9), facecolor="white")
fig.subplots_adjust(left=0.08, right=0.985, top=0.82, bottom=0.275)

width = 0.25
offsets = (-width, 0.0, width)

# Every bar is normalized to its one-board unoptimized baseline. Solid grey
# baselines are physical readings; hatching marks the derived unavailable
# baselines for Cases 09 and 10.
for i, row in enumerate(rows):
    estimated = row["baseline_estimated"]
    ax.bar(
        x[i] + offsets[0], 1.0, width=width,
        color="#cfd8dc" if estimated else "#78909c",
        edgecolor="#607d8b", linewidth=0.9,
        hatch="///" if estimated else None, zorder=3,
    )
    shared = row["case"] in {"01", "03", "04", "05"}
    suffix = "*" if estimated else ("†" if shared else "")
    ax.text(
        x[i] + offsets[0], 1.7, f"1.0×{suffix}",
        ha="center", va="bottom", rotation=90, fontsize=8.2,
        color="#455a64", zorder=5,
    )

bars_opt = ax.bar(
    x + offsets[1], [row["optimized_speedup"] for row in rows], width=width,
    color="#f9a825", edgecolor="white", linewidth=0.8, zorder=3,
)
bars_eight = ax.bar(
    x + offsets[2], [row["baseline_to_eight"] for row in rows], width=width,
    color="#1976d2", edgecolor="white", linewidth=0.8, zorder=3,
)

for bar, row in zip(bars_opt, rows):
    value = row["optimized_speedup"]
    suffix = "*" if row["baseline_estimated"] else ""
    ax.text(
        bar.get_x() + bar.get_width() / 2, value + 1.1,
        f"{value:.1f}×{suffix}",
        ha="center", va="bottom", rotation=90, fontsize=8.4,
        color="#8d6e00", fontweight="bold", zorder=5,
    )

for bar, row in zip(bars_eight, rows):
    value = row["baseline_to_eight"]
    estimate_suffix = "*" if row["baseline_estimated"] else ""
    active_label = f"\n{row['active']}/8 active" if row["active"] < 8 else ""
    ax.text(
        bar.get_x() + bar.get_width() / 2 + 0.025, value + 1.2,
        f"{value:.1f}× total{estimate_suffix}\n{row['optimized_to_eight']:.2f}× vs opt"
        f"{active_label}",
        ha="center", va="bottom", fontsize=8.1,
        color="#0d47a1", fontweight="bold", linespacing=1.02, zorder=5,
    )

ax.axhline(1.0, color="#90a4ae", linewidth=1.0, linestyle=(0, (3, 3)), zorder=1)
ax.set_ylim(0, 112)
ax.yaxis.set_major_locator(MultipleLocator(10))
ax.yaxis.set_minor_locator(MultipleLocator(2))
ax.grid(axis="y", which="major", color="#b0bec5", alpha=0.42, linewidth=0.8, zorder=0)
ax.grid(axis="y", which="minor", color="#cfd8dc", alpha=0.15, linewidth=0.5, zorder=0)
ax.set_axisbelow(True)

ax.set_title("Speedup over the one-board baseline: optimized → eight-board cluster", loc="left", fontsize=17, pad=13)
ax.set_ylabel("Speedup over one-board baseline (×) — linear scale, higher is better", labelpad=10)
ax.set_xlabel("Official benchmark case", labelpad=10)
ax.set_xticks(x, labels)
ax.set_xlim(-0.65, len(rows) - 0.35)

family_colors = {
    "01": "#607d8b", "02": "#607d8b", "03": "#607d8b", "04": "#607d8b", "05": "#607d8b",
    "07": "#ef6c00", "09": "#6a1b9a", "10": "#3949ab", "11": "#c62828", "12": "#2e7d32",
}
for tick, row in zip(ax.get_xticklabels(), rows):
    tick.set_color(family_colors[row["case"]])
    tick.set_fontweight("bold")

ax.axvline(4.5, color="#cfd8dc", linewidth=1.0, zorder=1)
ax.text(2.0, 109.5, "common D=128 / H=4 body · batch sweep", ha="center", fontsize=9, color="#78909c")
ax.text(7.0, 109.5, "shape variants", ha="center", fontsize=9, color="#78909c")

fig.suptitle(
    "ESP32-C3 optimization and eight-board scaling",
    fontsize=22, fontweight="bold", y=0.96,
)
fig.text(
    0.5, 0.914,
    "Linear speedup view · higher is better · estimated baseline-derived ratios are marked * · 160 MHz XIAO ESP32-C3",
    ha="center", fontsize=11.5, color="#455a64",
)

fig.legend(
    handles=[
        Patch(facecolor="#78909c", edgecolor="#607d8b", label="physical baseline reading († shared H=4 run)"),
        Patch(facecolor="#cfd8dc", edgecolor="#607d8b", hatch="///", label="derived unavailable baseline *"),
        Patch(facecolor="#f9a825", edgecolor="white", label="1 board · optimized"),
        Patch(facecolor="#1976d2", edgecolor="white", label="8-board cluster"),
    ],
    loc="lower center", bbox_to_anchor=(0.5, 0.153),
    ncol=4, frameon=False, fontsize=9.8,
)

fig.text(
    0.5, 0.072,
    "Speedup = baseline seconds/input ÷ result seconds/input. Orange is software optimization; blue combines optimization and cluster scaling. "
    f"† Cases 01/03/04/05 reuse the direct {BASELINE_01_05_S:.6f} s H=4 physical forward from Case 02 because B does not enter one forward. "
    "* Cases 09/10 had no runnable baseline, so ratios use clearly marked 45.65/45.82 s head-adjusted estimates. "
    "All orange timings and raw 8-node compute walls are experimental. Case 02 has 1/8 active; Case 03 has 4/8 active.",
    ha="center", fontsize=9.1, color="#455a64", wrap=True,
)

fig.savefig(OUT.with_suffix(".svg"), format="svg", bbox_inches="tight")
fig.savefig(OUT.with_suffix(".png"), format="png", dpi=220, bbox_inches="tight")

print(f"wrote {OUT.with_suffix('.svg')}")
print(f"wrote {OUT.with_suffix('.png')}")
for row in rows:
    source = "estimated" if row["baseline_estimated"] else "measured"
    print(
        f"case {row['case']}: baseline={row['baseline_per_input']:.6f}s/input ({source}), "
        f"optimized={row['optimized_per_input']:.6f}s/input, "
        f"8-cluster={row['eight_per_input']:.6f}s/input, "
        f"opt->8={row['optimized_to_eight']:.3f}x, total={row['baseline_to_eight']:.3f}x"
    )
