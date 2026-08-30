from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

# Cumulative single-board Case-02 timings from the optimisation log.
# Case 2 has B=1, so batch time and time per input forward are identical.
stages = [
    "Baseline",
    "Opts 1–2",
    "Opts 3–8",
    "Opts 8b–14",
    "Opts 15–20",
    "Opts 21–23",
]
stage_labels = [
    "Baseline\nFP32 / hybrid",
    "Opts 1–2\nInteger attention\n+ exponential LUT",
    "Opts 3–8\nTiled GEMM\n+ integer GELU/quant",
    "Opts 8b–14\nQ15 context fusion\n+ integer LN/quant",
    "Opts 15–20\nRegister-fit GEMM\n+ RISC-V assembly",
    "Opts 21–23\nInteger residual\n+ assembly fixes",
]
seconds = np.array([42.15, 13.70, 5.27, 3.205, 2.386, 1.996])
speedups = seconds[0] / seconds
colors = [
    "#607d8b",
    "#e09f3e",
    "#3f88c5",
    "#2a9d8f",
    "#7b61a8",
    "#2e7d32",
]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
})

fig, ax = plt.subplots(figsize=(14, 8.5))
fig.patch.set_facecolor("white")

x = np.arange(len(stages))
bars = ax.bar(
    x,
    seconds,
    width=0.68,
    color=colors,
    edgecolor="white",
    linewidth=1.0,
    zorder=3,
)

for index, (bar, value, speedup) in enumerate(zip(bars, seconds, speedups)):
    value_text = f"{value:.2f} s" if value >= 10 else f"{value:.3f} s"
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + (0.65 if value > 10 else 0.35),
        f"{value_text}\n{speedup:.1f}×",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold" if index == len(stages) - 1 else "normal",
        color="#1b5e20" if index == len(stages) - 1 else "#263238",
    )

ax.set_ylim(0, 47)
ax.yaxis.set_major_locator(MultipleLocator(5))
ax.grid(axis="y", color="#b0bec5", alpha=0.42, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
ax.set_xticks(x)
ax.set_xticklabels(stage_labels, fontsize=10.5, ha="center")
ax.tick_params(axis="x", pad=10)
ax.set_ylabel("Time per complete forward (seconds)")
ax.set_xlabel("Cumulative optimisation stage — key changes are shown below each bar", labelpad=18)

fig.suptitle(
    "Case 2 single-board optimisation",
    fontsize=21,
    fontweight="bold",
    y=0.965,
)
fig.text(
    0.5,
    0.915,
    "One XIAO ESP32-C3 · B=1 · six cumulative stages · linear time scale",
    ha="center",
    fontsize=12,
    color="#455a64",
)
fig.text(
    0.5,
    0.055,
    "42.15 s → 1.996 s: 21.1× faster, a 95.3% reduction in forward time. "
    "Opt 20 was profiling; the Opts 15–20 bar ends at the Opt 19 timing.",
    ha="center",
    fontsize=10,
    color="#455a64",
)
fig.text(
    0.5,
    0.025,
    "Lower is faster  |  complete four-layer forward  |  single-board measurement",
    ha="center",
    fontsize=10,
    color="#455a64",
)

fig.tight_layout(rect=[0.04, 0.20, 0.98, 0.86])

out = Path(__file__).with_suffix(".svg")
fig.savefig(out, format="svg", bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), format="png", dpi=200, bbox_inches="tight")
print(out)
