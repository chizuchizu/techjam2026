from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

# 16:9 PowerPoint canvas
fig = plt.figure(figsize=(13.333, 7.5), dpi=180, facecolor="#f7f9fc")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# Palette
navy = "#17324d"
blue = "#1565c0"
blue_dark = "#0d47a1"
teal = "#00796b"
green = "#2e7d32"
orange = "#ef6c00"
grey = "#607d8b"
light_blue = "#eaf2fb"
light_teal = "#e9f7f4"
light_green = "#edf7ee"
light_orange = "#fff3e8"
text = "#1f2933"
muted = "#52606d"

# Header
ax.text(0.03, 0.955, "ESP32-C3 single-board optimization", fontsize=20,
        fontweight="bold", color=navy, va="top")
ax.text(0.03, 0.918,
        "Case 2 · B=1, S=128, D=128, H=4, F=128, L=4 · 160 MHz RV32IMC · no FPU",
        fontsize=10.5, color=muted, va="top")

# Top headline metrics
metrics = [
    (0.60, "42.15 s", "initial", grey),
    (0.72, "→", "", navy),
    (0.79, "1.996 s", "final", green),
    (0.91, "21.1×", "faster · 95.3% less", blue),
]
for x, value, label, color in metrics:
    if value == "→":
        ax.text(x, 0.916, value, fontsize=25, fontweight="bold", color=color,
                ha="center", va="center")
    else:
        ax.text(x, 0.935, value, fontsize=18, fontweight="bold", color=color,
                ha="center", va="center")
        ax.text(x, 0.902, label, fontsize=8.5, color=muted, ha="center", va="center")

# Section label
ax.text(0.03, 0.855, "What changed — and why it mattered", fontsize=12.5,
        fontweight="bold", color=navy, va="center")

boxes = [
    {
        "x": 0.025, "color": blue, "fill": light_blue,
        "title": "OPTS 1–2\nATTENTION FIRST",
        "body": [
            "• Integer QK dot + PV",
            "• Exact-max integer softmax",
            "• 513-entry exp LUT",
            "\nWhy: eliminate FP32/expf()\nsoftware calls in the 71.4%\nattention bottleneck.",
        ],
        "impact": "42.15 → 13.70 s\n2.77× → 3.08×",
    },
    {
        "x": 0.218, "color": teal, "fill": light_teal,
        "title": "OPTS 3–8\nTILE + QUANTIZE",
        "body": [
            "• core2/core3/core4 GEMM tiling",
            "• Runtime-scale integer GELU",
            "• Fast rounding; packed QK/PV",
            "• Fused QKV Q30 quantization",
            "• Q32 LUT index; head GEMM",
            "\nWhy: reuse registers/weights and\nremove FP32 staging passes.",
        ],
        "impact": "13.70 → 5.27 s\n3.08× → 8.0×",
    },
    {
        "x": 0.411, "color": teal, "fill": light_teal,
        "title": "OPTS 9–14\nFUSE THE DATA PATH",
        "body": [
            "• Fused LayerNorm → Q15",
            "• Direct Q15 attention context",
            "• Integer PV + ctx epilogue",
            "• j-tile-2 + K-pair prefetch",
            "• Integer LN amax/quantization",
            "\nWhy: avoid writing, scanning,\nand converting the same tensor\nmultiple times.",
        ],
        "impact": "5.27 → 3.205 s\n8.0× → 13.1×",
    },
    {
        "x": 0.604, "color": orange, "fill": light_orange,
        "title": "OPTS 15–19\nSCHEDULE THE CORE",
        "body": [
            "• core5 4×2 tile avoids spills",
            "• RISC-V asm head GEMM",
            "• FFN1 direct Q15 output",
            "• Bias fold + asm requant",
            "• QK limb arithmetic + unroll-4",
            "• PV 8 accumulators",
            "\nWhy: keep operands in registers\nand hide in-order pipeline latency.",
        ],
        "impact": "3.205 → 2.386 s\n13.1× → 17.6×",
    },
    {
        "x": 0.797, "color": green, "fill": light_green,
        "title": "OPTS 21–23\nINTEGER RESIDUAL",
        "body": [
            "• int32 fixed-scale residual",
            "• Integer norms + epilogues",
            "• Remove res1/res2 passes",
            "• Compose with head-GEMM asm",
            "• Fix core5 col-1 assembly bug",
            "\nWhy: keep the entire FAST path\ninteger and remove final FP32\nconversion overhead.",
        ],
        "impact": "2.386 → 1.996 s\n17.6× → 21.1×",
    },
]

box_y = 0.40
box_w = 0.178
box_h = 0.405

for i, b in enumerate(boxes):
    patch = FancyBboxPatch((b["x"], box_y), box_w, box_h,
                           boxstyle="round,pad=0.008,rounding_size=0.012",
                           linewidth=1.3, edgecolor=b["color"], facecolor=b["fill"])
    ax.add_patch(patch)
    ax.text(b["x"] + box_w / 2, box_y + box_h - 0.025, b["title"],
            ha="center", va="top", fontsize=10.5, fontweight="bold", color=b["color"],
            linespacing=1.05)
    ax.text(b["x"] + 0.012, box_y + box_h - 0.105, "\n".join(b["body"]),
            ha="left", va="top", fontsize=8.25, color=text, linespacing=1.25)
    ax.text(b["x"] + box_w / 2, box_y + 0.027, b["impact"],
            ha="center", va="bottom", fontsize=9.1, fontweight="bold", color=b["color"],
            linespacing=1.1)
    if i < len(boxes) - 1:
        x1 = b["x"] + box_w + 0.003
        x2 = boxes[i + 1]["x"] - 0.003
        arrow = FancyArrowPatch((x1, box_y + box_h / 2), (x2, box_y + box_h / 2),
                                arrowstyle="-|>", mutation_scale=13, linewidth=1.5,
                                color="#78909c", connectionstyle="arc3,rad=0")
        ax.add_patch(arrow)

# Profiling / constraints strip (Opt 20)
opt20 = FancyBboxPatch((0.025, 0.255), 0.95, 0.095,
                       boxstyle="round,pad=0.008,rounding_size=0.01",
                       linewidth=1.1, edgecolor=grey, facecolor="#eef2f5")
ax.add_patch(opt20)
ax.text(0.04, 0.325, "OPT 20 · PROFILED THE LIMITS", fontsize=10.5,
        fontweight="bold", color=grey, va="center")
ax.text(0.04, 0.285,
        "Measured flash/SRAM traffic and register pressure. DRAM was ~99.9% full; larger tiles caused spills and IRAM placement overflowed. "
        "This redirected effort toward fusion, fixed-point arithmetic, and instruction scheduling.",
        fontsize=8.8, color=text, va="center")

# Bottom significance cards
cards = [
    (0.025, 0.06, 0.29, 0.15, blue, "ATTENTION", "30.09 s → ~0.52 s", "≈58× faster component"),
    (0.355, 0.06, 0.29, 0.15, green, "END TO END", "42.15 s → 1.996 s", "21.1× · 95.3% reduction"),
    (0.685, 0.06, 0.29, 0.15, orange, "CORRECTNESS", "25/25 device · 54/54 host", "Worst error ≈1.24×10⁻³"),
]
for x, y, w, h, color, heading, main, sub in cards:
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.01",
                       linewidth=1.1, edgecolor=color, facecolor="white")
    ax.add_patch(p)
    ax.text(x + 0.014, y + h - 0.027, heading, fontsize=9.5, fontweight="bold",
            color=color, va="top")
    ax.text(x + w / 2, y + 0.078, main, fontsize=14, fontweight="bold",
            color=navy, ha="center", va="center")
    ax.text(x + w / 2, y + 0.031, sub, fontsize=8.7, color=muted,
            ha="center", va="center")

ax.text(0.5, 0.022,
        "All timings are complete single-board forwards; host serial transfer excluded. FAST path validated against the reference tolerance gate.",
        ha="center", va="center", fontsize=7.8, color=muted)

out = Path(__file__).with_suffix(".svg")
fig.savefig(out, format="svg", bbox_inches="tight", facecolor=fig.get_facecolor())
fig.savefig(out.with_suffix(".png"), format="png", dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
print(out)
