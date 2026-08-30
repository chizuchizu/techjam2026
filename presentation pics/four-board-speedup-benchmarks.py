"""Plot multiboard speedups over the best optimized one-board benchmark.

The single figure compares the measured two-board and four-board complete-case
compute walls with the best published optimized one-board total for each case.
All values are device compute time; host USB/WiFi transfer is excluded.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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


# These are the best published optimized one-board full-case totals from the
# case READMEs and benchmarks/README.md. T2/T4 are read from raw result JSONs.
# Case 09 intentionally has no T2: only a two-worker smoke test was recorded,
# not a complete two-worker batch.
rows = [
    {
        "case": "01",
        "one": 127.36,
        "batch": 64,
        "two_path": "benchmarks/batch-dp/results_two_c3_dp.json",
        "four_path": "benchmarks/batch-dp/results_cases_1_3_4_5_four_c3_wifi.json",
    },
    {
        "case": "03",
        "one": 7.96,
        "batch": 4,
        "two_path": "benchmarks/batch-dp/results_two_c3_dp.json",
        "four_path": "benchmarks/batch-dp/results_cases_1_3_4_5_four_c3_wifi.json",
    },
    {
        "case": "04",
        "one": 31.84,
        "batch": 16,
        "two_path": "benchmarks/batch-dp/results_two_c3_dp.json",
        "four_path": "benchmarks/batch-dp/results_cases_1_3_4_5_four_c3_wifi.json",
    },
    {
        "case": "05",
        "one": 254.72,
        "batch": 128,
        "two_path": "benchmarks/batch-dp/results_two_c3_dp.json",
        "four_path": "benchmarks/batch-dp/results_cases_1_3_4_5_four_c3_wifi.json",
    },
    {
        "case": "07",
        "one": 30.427,
        "batch": 64,
        "two_path": "benchmarks/case-07/multiboard/results_case7_two_c3_wifi.json",
        "four_path": "benchmarks/case-07/multiboard/results_case7_four_c3_wifi.json",
    },
    {
        "case": "09",
        "one": 138.027,
        "batch": 64,
        "two_path": None,
        "four_path": "benchmarks/case-09/multiboard/results_case9_four_c3_wifi.json",
    },
    {
        "case": "10",
        "one": 138.536,
        "batch": 64,
        "two_path": "benchmarks/case-10/multiboard/results_case10_two_c3_wifi.json",
        "four_path": "benchmarks/case-10/multiboard/results_case10_four_c3_wifi.json",
    },
    {
        "case": "11",
        "one": 138.610,
        "batch": 64,
        "two_path": "benchmarks/case-11/multiboard/results_case11_two_c3_wifi.json",
        "four_path": "benchmarks/case-11/multiboard/results_case11_four_c3_wifi.json",
    },
    {
        "case": "12",
        "one": 33.879,
        "batch": 64,
        "two_path": "benchmarks/case-12/multiboard/results_case12_two_c3_wifi.json",
        "four_path": "benchmarks/case-12/multiboard/results_case12_four_c3_wifi.json",
    },
]

for row in rows:
    row["two"] = (
        load_result(row["two_path"], row["batch"])["compute_wall_s"]
        if row["two_path"]
        else None
    )
    row["four"] = load_result(row["four_path"], row["batch"])["compute_wall_s"]
    row["speedup_two"] = row["one"] / row["two"] if row["two"] is not None else None
    row["speedup_four"] = row["one"] / row["four"]

labels = [row["case"] for row in rows]
x = np.arange(len(rows))

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titleweight": "bold",
    }
)

fig, ax = plt.subplots(figsize=(15, 8.5), facecolor="white")
fig.subplots_adjust(left=0.085, right=0.98, top=0.82, bottom=0.26)

width = 0.34
speedup_two = np.array(
    [row["speedup_two"] if row["speedup_two"] is not None else np.nan for row in rows]
)
speedup_four = np.array([row["speedup_four"] for row in rows])

bars_two = ax.bar(
    x - width / 2,
    np.nan_to_num(speedup_two, nan=0.0),
    width=width,
    color="#66bb6a",
    edgecolor="white",
    linewidth=0.8,
    zorder=3,
)
bars_four = ax.bar(
    x + width / 2,
    speedup_four,
    width=width,
    color="#1976d2",
    edgecolor="white",
    linewidth=0.8,
    zorder=3,
)

# Reference lines show ideal data-parallel scaling from one board.
ax.axhline(1.0, color="#90a4ae", linewidth=1.0, linestyle=(0, (2, 3)), zorder=1)
ax.axhline(2.0, color="#388e3c", linewidth=1.3, linestyle=(0, (5, 3)), zorder=1)
ax.axhline(4.0, color="#455a64", linewidth=1.3, linestyle=(0, (5, 3)), zorder=1)
ax.text(len(rows) - 0.12, 1.04, "1× baseline", ha="right", va="bottom", fontsize=9, color="#607d8b")
ax.text(len(rows) - 0.12, 2.04, "ideal 2×", ha="right", va="bottom", fontsize=9, color="#388e3c")
ax.text(len(rows) - 0.12, 4.04, "ideal 4×", ha="right", va="bottom", fontsize=9, color="#455a64")

# Annotate every measured bar with both the ratio and its absolute compute wall.
for bar, row in zip(bars_two, rows):
    if row["two"] is None:
        ax.bar(
            bar.get_x() + bar.get_width() / 2,
            0.16,
            width=bar.get_width(),
            color="#b0bec5",
            edgecolor="#607d8b",
            hatch="///",
            linewidth=0.8,
            zorder=4,
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            0.23,
            "n/a",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#455a64",
        )
    else:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.07,
            f"{row['speedup_two']:.2f}×\n{row['two']:.1f}s",
            ha="center",
            va="bottom",
            fontsize=9.3,
            color="#2e7d32",
            fontweight="bold",
            linespacing=1.05,
        )

for bar, row in zip(bars_four, rows):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.07,
        f"{row['speedup_four']:.2f}×\n{row['four']:.1f}s",
        ha="center",
        va="bottom",
        fontsize=9.3,
        color="#1565c0",
        fontweight="bold",
        linespacing=1.05,
    )

ax.set_title("Speedup over the best optimized one-board benchmark", loc="left", fontsize=17, pad=13)
ax.set_ylabel("Speedup  =  best one-board compute / cluster compute", labelpad=10)
ax.set_xlabel("Official benchmark case", labelpad=10)
ax.set_xticks(x, labels)
ax.set_ylim(0, 4.65)
ax.set_xlim(-0.65, len(rows) - 0.35)
ax.grid(axis="y", alpha=0.25, zorder=0)
ax.set_axisbelow(True)

# Separate the original H=4/D=128 batch sweep from later shape variants.
ax.axvline(3.5, color="#cfd8dc", linewidth=1.0, zorder=1)
ax.text(1.5, 4.53, "same H=4 / D=128 shape · batch sweep", ha="center", fontsize=9, color="#78909c")
ax.text(6.0, 4.53, "shape variants", ha="center", fontsize=9, color="#78909c")

family_colors = {
    "01": "#607d8b",
    "03": "#607d8b",
    "04": "#607d8b",
    "05": "#607d8b",
    "07": "#ef6c00",
    "09": "#6a1b9a",
    "10": "#3949ab",
    "11": "#c62828",
    "12": "#2e7d32",
}
for tick, row in zip(ax.get_xticklabels(), rows):
    tick.set_color(family_colors[row["case"]])
    tick.set_fontweight("bold")

fig.suptitle(
    "ESP32-C3 multiboard speedup across complete benchmarks",
    fontsize=21,
    fontweight="bold",
    y=0.96,
)
fig.text(
    0.5,
    0.915,
    "Two-board and four-board results shown as gains over the best optimized one-board total · 160 MHz XIAO ESP32-C3",
    ha="center",
    fontsize=11.5,
    color="#455a64",
)

fig.legend(
    handles=[
        Patch(facecolor="#66bb6a", edgecolor="white", label="2 boards"),
        Patch(facecolor="#1976d2", edgecolor="white", label="4 boards"),
        Patch(facecolor="#b0bec5", edgecolor="#607d8b", hatch="///", label="no complete 2-board run"),
        Line2D([0], [0], color="#455a64", linestyle=(0, (5, 3)), label="ideal scaling"),
    ],
    loc="lower center",
    bbox_to_anchor=(0.5, 0.145),
    ncol=4,
    frameon=False,
    fontsize=10,
)
fig.text(
    0.5,
    0.055,
    "Bars are T₁/T₂ and T₁/T₄; labels include speedup and cluster compute seconds. "
    "All plotted runs passed validation and exclude host transfer. Cases 01/03/04/05 use USB two-board DP versus WiFi four-board DP; "
    "cases 07/10/11/12 use WiFi for both. Case 09 has no complete T₂ run. Case 02 is excluded because B=1 and its four-node run used one active board.",
    ha="center",
    fontsize=9.1,
    color="#455a64",
    wrap=True,
)

fig.savefig(OUT.with_suffix(".svg"), format="svg", bbox_inches="tight")
fig.savefig(OUT.with_suffix(".png"), format="png", dpi=220, bbox_inches="tight")

print(f"wrote {OUT.with_suffix('.svg')}")
print(f"wrote {OUT.with_suffix('.png')}")
for row in rows:
    two = "—" if row["speedup_two"] is None else f"{row['speedup_two']:.3f}× vs 1"
    print(f"case {row['case']}: {two}; {row['speedup_four']:.3f}× vs 1 on 4 boards")
