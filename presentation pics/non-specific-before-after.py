from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

cases = ["TC1", "TC2", "TC3", "TC4", "TC5", "TC7", "TC11", "TC12"]
before = np.array([42.15, 42.15, 42.15, 42.15, 42.15, 0.491, 2.462, 0.493])
after = np.array([1.990, 1.996, 1.990, 1.990, 1.990, 0.475, 2.166, 0.529])
speedup = before / after

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, (ax_large, ax_small) = plt.subplots(
    2, 1, figsize=(13.333, 7.5), gridspec_kw={"height_ratios": [1.25, 1]}
)
fig.suptitle(
    "Single-board timing: before vs after shared Opts 1–23",
    fontsize=19,
    fontweight="bold",
    y=0.97,
)

before_color = "#78909c"
after_color = "#2e7d32"
width = 0.34

def draw_panel(ax, indices, x_limit, title):
    labels = [cases[i] for i in indices]
    y = np.arange(len(indices))
    b = before[indices]
    a = after[indices]
    s = speedup[indices]

    ax.barh(y + width / 2, b, height=width, color=before_color, label="Before")
    ax.barh(y - width / 2, a, height=width, color=after_color, label="After")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, x_limit)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)

    for row, old, new, ratio in zip(y, b, a, s):
        ax.text(old + x_limit * 0.012, row + width / 2, f"{old:g} s", va="center", fontsize=10)
        ax.text(new + x_limit * 0.012, row - width / 2, f"{new:.3f} s", va="center", fontsize=10)
        suffix = " slower" if ratio < 1 else ""
        ax.text(
            x_limit * 0.985,
            row,
            f"{ratio:.2f}×{suffix}",
            va="center",
            ha="right",
            fontweight="bold",
            color="#b26a00" if ratio < 1 else "#1b5e20",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
        )



draw_panel(ax_large, np.arange(5), 48, "TC1–TC5: shared D=128 geometry")
draw_panel(ax_small, np.array([5, 6, 7]), 3.2, "TC7, TC11, TC12: different geometries")
ax_small.set_xlabel("Time per forward (seconds)")
fig.legend(
    [plt.Rectangle((0, 0), 1, 1, color=before_color), plt.Rectangle((0, 0), 1, 1, color=after_color)],
    ["Before", "After"], loc="upper right", bbox_to_anchor=(0.97, 0.935),
    ncol=2, frameon=False,
)
fig.text(
    0.99,
    0.015,
    "Speedup = before ÷ after  |  single-board measurements",
    ha="right",
    fontsize=10,
    color="#444444",
)
fig.tight_layout(rect=[0.02, 0.04, 0.99, 0.94])

out = Path(__file__).with_suffix(".svg")
fig.savefig(out, format="svg", bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), format="png", dpi=180, bbox_inches="tight")
print(out)
