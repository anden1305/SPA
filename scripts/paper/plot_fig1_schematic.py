#!/usr/bin/env python3
"""Fig 1 overview: MSSV cv4fold, train vs zero-shot eval, model ladder."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = Path(__file__).resolve().parents[2]


def _panel_label(ax, letter: str) -> None:
    ax.text(-0.08, 1.05, letter, transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")


def draw_panel_a(ax) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    _panel_label(ax, "A")
    labs = [("Lab 2", 1.5), ("Lab 3", 5.0), ("Lab 5", 8.5)]
    for name, x in labs:
        ax.add_patch(FancyBboxPatch((x - 1, 3.5), 2, 1.8, boxstyle="round,pad=0.05", fc="#e8f4fc", ec="#333"))
        ax.text(x, 4.4, name, ha="center", fontsize=9, fontweight="bold")
        ax.text(x, 3.9, "mice", ha="center", fontsize=8)
    ax.add_patch(FancyBboxPatch((0.5, 0.8), 9, 2.0, boxstyle="round,pad=0.08", fc="#f5f5f5", ec="#666", linestyle="--"))
    ax.text(5, 2.3, "4-fold CV (leave mice out)", ha="center", fontsize=10, fontweight="bold")
    ax.text(2.5, 1.5, "Train folds\n(3 labs)", ha="center", fontsize=8)
    ax.text(7.5, 1.5, "Held-out fold\n(zero-shot eval)", ha="center", fontsize=8)
    ax.annotate("", xy=(6.5, 1.8), xytext=(3.5, 1.8), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(5, 0.3, "MSSV (OpenNeuro ds006366)", ha="center", fontsize=8, style="italic")


def draw_panel_b(ax) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    _panel_label(ax, "B")

    def box(x, y, w, h, text, fc="#fff", ec="#333"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=fc, ec=ec))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7)

    ax.text(2.5, 5.5, "Training", ha="center", fontsize=10, fontweight="bold")
    box(0.5, 3.8, 1.6, 0.9, "EEG/EMG")
    box(2.3, 3.8, 1.6, 0.9, "Encoder\n(no subj. ID)")
    box(4.1, 3.8, 1.4, 0.9, "Latent z")
    box(0.5, 2.3, 1.6, 0.9, "Decoder\n+ subj. emb.")
    ax.annotate("", xy=(2.2, 2.7), xytext=(4.8, 3.8), arrowprops=dict(arrowstyle="->", lw=1))
    ax.annotate("", xy=(1.3, 3.8), xytext=(0.5, 3.2), arrowprops=dict(arrowstyle="->", lw=1))
    ax.annotate("", xy=(2.3, 4.25), xytext=(1.3, 4.25), arrowprops=dict(arrowstyle="->", lw=1))
    ax.annotate("", xy=(4.1, 4.25), xytext=(3.9, 4.25), arrowprops=dict(arrowstyle="->", lw=1))

    ax.text(7.5, 5.5, "Holdout (zero-shot)", ha="center", fontsize=10, fontweight="bold", color="#1565c0")
    box(6.0, 3.8, 1.6, 0.9, "EEG/EMG")
    box(7.8, 3.8, 1.6, 0.9, "Encoder\n(no subj. ID)")
    box(6.0, 2.3, 3.4, 0.9, "HMM--GMM prior → states", fc="#e3f2fd", ec="#1565c0")
    ax.annotate("", xy=(7.8, 4.25), xytext=(6.8, 4.25), arrowprops=dict(arrowstyle="->", lw=1.5, color="#1565c0"))
    ax.annotate("", xy=(7.7, 3.2), xytext=(8.6, 3.8), arrowprops=dict(arrowstyle="->", lw=1.5, color="#1565c0"))
    ax.text(5, 0.5, "No subject ID · no AccuSleep calibration", ha="center", fontsize=8, color="#1565c0")


def draw_panel_c(ax) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    _panel_label(ax, "C")
    rungs = [
        ("HMM\n(features)", "#9e9e9e"),
        ("HMMGMVAE", "#4daf4a"),
        ("cGMVAE", "#377eb8"),
        ("cHMM--GMVAE", "#984ea3"),
    ]
    y = 4.5
    for i, (label, color) in enumerate(rungs):
        x = 0.8 + i * 2.2
        ax.add_patch(FancyBboxPatch((x, y), 1.8, 1.2, boxstyle="round,pad=0.05", fc=color, ec="#333", alpha=0.85))
        ax.text(x + 0.9, y + 0.6, label, ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        if i < len(rungs) - 1:
            ax.annotate("", xy=(x + 2.0, y + 0.6), xytext=(x + 1.85, y + 0.6),
                        arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(5, 2.5, "Locked recipes · same features · prior NMI", ha="center", fontsize=9)
    ax.text(5, 1.5, "+ temporal prior", ha="center", fontsize=8, style="italic")
    ax.annotate("", xy=(8.5, 3.2), xytext=(8.5, 2.8), arrowprops=dict(arrowstyle="->", lw=1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "docs/paper/figures/Fig1.tif",
    )
    args = parser.parse_args()

    fig, axes = plt.subplots(3, 1, figsize=(7, 9))
    draw_panel_a(axes[0])
    draw_panel_b(axes[1])
    draw_panel_c(axes[2])
    fig.suptitle("Fig 1 — Study overview", fontsize=12, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(args.out.with_suffix(".pdf"), dpi=200)
    plt.close(fig)
    # Mirror to overleaf staging
    staging = REPO / "paper/overleaf/figures/figure1_overview.pdf"
    staging.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(args.out.with_suffix(".pdf"), staging)
    print(f"Wrote {args.out} and {args.out.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
