#!/usr/bin/env python3
"""Fig 1: MSSV cv4fold, zero-shot inference, model ladder (publication layout)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from scripts.paper.plot_style import LADDER_COLORS, LADDER_LABELS, apply_paper_style, panel_label, save_figure

REPO = Path(__file__).resolve().parents[2]


def _rounded_box(ax, xy, w, h, text, *, fc="#F8FAFC", ec="#334155", fontsize=8, bold=False):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08", fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, fontweight="bold" if bold else "normal")


def draw_panel_a(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_label(ax, "A", x=-0.06, y=1.02)

    for i, (lab, x) in enumerate([("Lab 2", 0.12), ("Lab 3", 0.42), ("Lab 5", 0.72)]):
        _rounded_box(ax, (x, 0.62), 0.22, 0.22, f"{lab}\nEEG + EMG", fc="#EFF6FF", ec="#3B82F6", bold=True)

    ax.add_patch(Rectangle((0.05, 0.12), 0.9, 0.38, fill=False, ec="#94A3B8", lw=1.0, linestyle=(0, (4, 3))))
    ax.text(0.5, 0.44, "4-fold cross-validation (leave mice out)", ha="center", fontsize=9, fontweight="bold")
    _rounded_box(ax, (0.1, 0.18), 0.32, 0.18, "Train: 3 folds\n(all labs)", fc="#F0FDF4", ec="#16A34A")
    _rounded_box(ax, (0.58, 0.18), 0.32, 0.18, "Test: held-out mice\n(zero-shot)", fc="#FEF3C7", ec="#D97706")
    ax.annotate("", xy=(0.58, 0.27), xytext=(0.42, 0.27), arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#64748B"))
    ax.text(0.5, 0.04, "MSSV · OpenNeuro ds006366", ha="center", fontsize=7.5, color="#64748B", style="italic")


def draw_panel_b(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_label(ax, "B", x=-0.06, y=1.02)

    ax.text(0.25, 0.92, "Training", ha="center", fontsize=9, fontweight="bold", color="#334155")
    ax.text(0.75, 0.92, "Holdout inference", ha="center", fontsize=9, fontweight="bold", color="#1D4ED8")

    # Train path
    for label, x in [("Signal", 0.04), ("Encoder", 0.20), ("Latent", 0.36)]:
        _rounded_box(ax, (x, 0.55), 0.12, 0.18, label)
    _rounded_box(ax, (0.04, 0.28), 0.18, 0.16, "Decoder\n+ subject ID", fc="#FDF2F8", ec="#DB2777")
    for x0, x1 in [(0.16, 0.20), (0.32, 0.36)]:
        ax.annotate("", xy=(x1, 0.64), xytext=(x0, 0.64), arrowprops=dict(arrowstyle="-|>", lw=1.0))
    ax.annotate("", xy=(0.10, 0.44), xytext=(0.42, 0.55), arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#DB2777"))

    # Eval path
    for x, t in [(0.58, "Signal"), (0.74, "Encoder")]:
        _rounded_box(ax, (x, 0.55), 0.12, 0.18, t, fc="#EFF6FF", ec="#1D4ED8")
    _rounded_box(ax, (0.58, 0.28), 0.28, 0.16, "HMM–GMM prior\n→ sleep states", fc="#DBEAFE", ec="#1D4ED8", bold=True)
    ax.annotate("", xy=(0.74, 0.64), xytext=(0.70, 0.64), arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#1D4ED8"))
    ax.annotate("", xy=(0.72, 0.44), xytext=(0.80, 0.55), arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#1D4ED8"))
    ax.text(0.5, 0.06, "No subject ID at staging · no per-mouse calibration", ha="center", fontsize=7.5, color="#1D4ED8")


def draw_panel_c(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_label(ax, "C", x=-0.06, y=1.02)

    keys = ["hmm_features", "hmmgmvae_locked", "cgmvae_locked", "chmmgmvae_locked"]
    n = len(keys)
    w = 0.19
    gap = 0.025
    x0 = 0.06
    y = 0.45
    for i, key in enumerate(keys):
        x = x0 + i * (w + gap)
        color = LADDER_COLORS[key]
        label = LADDER_LABELS[key].replace("–", "-")  # matplotlib font
        ax.add_patch(FancyBboxPatch((x, y), w, 0.28, boxstyle="round,pad=0.03", fc=color, ec="white", lw=0.5))
        ax.text(x + w / 2, y + 0.14, label, ha="center", va="center", fontsize=7.5, color="white", fontweight="bold")
        if i < n - 1:
            ax.annotate("", xy=(x + w + gap * 0.2, y + 0.14), xytext=(x + w, y + 0.14),
                        arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#64748B"))
    ax.text(0.5, 0.22, "Locked recipes · identical features · prior-prediction NMI", ha="center", fontsize=8, color="#475569")
    ax.annotate("temporal prior", xy=(0.88, 0.78), xytext=(0.88, 0.62),
                arrowprops=dict(arrowstyle="-|>", color="#7C3AED", lw=1.2), fontsize=7, color="#7C3AED", ha="center")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "docs/paper/figures/Fig1.pdf")
    args = parser.parse_args()

    apply_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    draw_panel_a(axes[0])
    draw_panel_b(axes[1])
    draw_panel_c(axes[2])
    fig.subplots_adjust(wspace=0.35, left=0.06, right=0.98, top=0.92, bottom=0.08)
    save_figure(fig, args.out)
    plt.close(fig)

    import shutil
    staging = REPO / "paper/overleaf/figures/figure1_overview.pdf"
    staging.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.out, staging)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
