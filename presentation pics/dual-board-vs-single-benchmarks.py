from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Causal, physically measured comparisons. The scopes are intentionally shown
# in the labels because only the first row is a complete four-layer forward.
benchmarks = [
    "Complete Case 2 forward\n(4 layers, full Transformer)",
    "Layer 0: LN + Q/K/V +\ncausal attention",
    "Four independent causal\nattention heads",
]
single = np.array([1.990, 9.693, 3.0215])
dual = np.array([1.276, 4.850, 1.512])
speedups = single / dual

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
})

fig, ax = plt.subplots(figsize=(13.333, 7.5))
fig.patch.set_facecolor("white")

y = np.arange(len(benchmarks))
width = 0.34
single_color = "#78909c"
dual_color = "#2e7d32"

single_bars = ax.barh(
    y + width / 2,
    single,
    height=width,
    color=single_color,
    label="Single board",
    edgecolor="white",
    linewidth=0.8,
    zorder=3,
)
dual_bars = ax.barh(
    y - width / 2,
    dual,
    height=width,
    color=dual_color,
    label="Dual board",
    edgecolor="white",
    linewidth=0.8,
    zorder=3,
)

for row, (single_bar, dual_bar, one, two, ratio) in enumerate(
    zip(single_bars, dual_bars, single, dual, speedups)
):
    ax.text(
        one + 0.10,
        single_bar.get_y() + single_bar.get_height() / 2,
        f"{one:.3f} s",
        va="center",
        fontsize=10,
        color="#263238",
    )
    ax.text(
        two + 0.10,
        dual_bar.get_y() + dual_bar.get_height() / 2,
        f"{two:.3f} s",
        va="center",
        fontsize=10,
        fontweight="bold" if row == 0 else "normal",
        color="#1b5e20",
    )
    ax.text(
        10.55,
        row,
        f"{ratio:.2f}×",
        va="center",
        ha="right",
        fontsize=12,
        fontweight="bold",
        color="#1b5e20",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
    )

ax.set_yticks(y)
ax.set_yticklabels(benchmarks, fontsize=11)
ax.invert_yaxis()
ax.set_xlim(0, 10.8)
ax.set_xlabel("Wall time per benchmark (seconds, linear scale)", labelpad=10)
ax.set_ylabel("Benchmark scope")
ax.grid(axis="x", alpha=0.28, zorder=0)
ax.set_axisbelow(True)

fig.suptitle(
    "Dual-board versus single-board ESP32-C3 benchmarks",
    fontsize=19,
    fontweight="bold",
    y=0.965,
)
fig.text(
    0.5,
    0.915,
    "Causal workloads · two matched 160 MHz XIAO ESP32-C3 boards · lower is faster",
    ha="center",
    fontsize=11,
    color="#455a64",
)

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.03),
    ncol=2,
    frameon=False,
    handlelength=1.4,
)

fig.text(
    0.5,
    0.055,
    "The complete Case 2 row is the final distributed solution: 1.990 s → 1.276 s (1.56×). "
    "The other rows are partial-scope scaling benchmarks.",
    ha="center",
    fontsize=9.5,
    color="#455a64",
)
fig.text(
    0.5,
    0.025,
    "All reported rows passed their numerical validation gates; host serial transfer is excluded from the timed windows.",
    ha="center",
    fontsize=9.5,
    color="#455a64",
)

fig.tight_layout(rect=[0.03, 0.12, 0.98, 0.86])

out = Path(__file__).with_suffix(".svg")
fig.savefig(out, format="svg", bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), format="png", dpi=200, bbox_inches="tight")
print(out)
