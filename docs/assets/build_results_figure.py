#!/usr/bin/env python3
"""Regenerate the README cross-case result figure from benchmarks/README.md."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CASES = ["01", "03", "04", "05", "07", "09", "10", "11", "12"]
BASELINE = np.array(
    [2697.6, 168.6, 674.4, 5395.2, 295.05, 2697.6, 2697.6, 2697.6, 547.95]
)
OPTIMISED = np.array(
    [127.36, 7.96, 31.84, 254.72, 30.427, 138.027, 138.536, 138.610, 33.879]
)
CLUSTER = np.array(
    [33.713, 4.218, 8.438, 67.451, 3.963, 28.508, 29.793, 51.604, 4.282]
)
MARKERS = ["*", "*", "*", "*", "†", "†", "†", "†", "†"]


def main() -> None:
    x = np.arange(len(CASES))
    one_board_speedup = BASELINE / OPTIMISED
    cluster_speedup = BASELINE / CLUSTER
    cluster_vs_optimised = OPTIMISED / CLUSTER

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#243746",
            "axes.labelcolor": "#243746",
            "xtick.color": "#243746",
            "ytick.color": "#243746",
        }
    )
    fig, ax = plt.subplots(figsize=(15.04, 7.21), dpi=100)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfcfd")

    width = 0.28
    opt_bars = ax.bar(
        x - width / 2,
        one_board_speedup,
        width,
        color="#f5a623",
        label="Optimized, 1 C3",
        zorder=3,
    )
    cluster_bars = ax.bar(
        x + width / 2,
        cluster_speedup,
        width,
        color="#1674c9",
        label="Cluster result",
        zorder=3,
    )

    ax.axhline(1, color="#758b98", linewidth=1.4, linestyle=(0, (3, 3)), zorder=2)
    ax.text(-0.53, 2.2, "baseline 1.0x", color="#647985", fontsize=9)
    ax.axvline(3.5, color="#c7d3da", linewidth=1.2)

    for i, (opt_bar, cluster_bar) in enumerate(zip(opt_bars, cluster_bars)):
        marker = MARKERS[i]
        ax.text(
            opt_bar.get_x() + opt_bar.get_width() / 2,
            opt_bar.get_height() + 1.5,
            f"{one_board_speedup[i]:.1f}x{marker}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#9a6900",
        )
        ax.text(
            cluster_bar.get_x() + cluster_bar.get_width() / 2,
            cluster_bar.get_height() + 1.5,
            f"{cluster_speedup[i]:.1f}x total{marker}\n{cluster_vs_optimised[i]:.2f}x vs opt",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#0756a6",
            linespacing=1.05,
        )

    ax.set_ylim(0, 142)
    ax.set_yticks(np.arange(0, 141, 10))
    ax.set_xticks(x, CASES, fontsize=11, fontweight="bold")
    ax.set_xlabel("Official benchmark case", fontsize=11)
    ax.set_ylabel("Speedup over one-board baseline (x)", fontsize=11)
    ax.grid(axis="y", color="#dce4e8", linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=10)

    fig.suptitle(
        "ESP32-C3 single-board optimization and cluster scaling",
        fontsize=21,
        fontweight="bold",
        y=0.97,
    )
    ax.set_title(
        "Complete four-layer Transformer body · lower time is converted to higher speedup",
        fontsize=11,
        color="#526a78",
        pad=13,
    )
    fig.text(
        0.5,
        0.015,
        "* direct batch projection from measured Case 2 baseline    † FLOP-normalized baseline estimate",
        ha="center",
        fontsize=9,
        color="#526a78",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.84, bottom=0.14)
    fig.savefig(Path(__file__).with_name("optimisation-and-eight-board-scaling.png"))


if __name__ == "__main__":
    main()
