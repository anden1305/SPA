#!/usr/bin/env python3
"""Fig 2: joint holdout ladder — mean + per-fold panels (publication quality)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.paper.plot_style import (
    LADDER_COLORS,
    LADDER_LABELS,
    THESIS_HMM_REF,
    apply_paper_style,
    panel_label,
    save_figure,
)

REPO = Path(__file__).resolve().parents[2]
MODEL_KEYS = ("cgmvae_locked", "chmmgmvae_locked")
FOLD_ORDER = ("1", "2", "3", "4")


def _load_joint_csv(csv_path: Path) -> dict[str, dict[str, float]]:
    by_fold: dict[str, dict[str, float]] = {f: {} for f in FOLD_ORDER}
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("scope") != "joint":
                continue
            fold, model = row["fold"], row["model"]
            if fold in by_fold:
                by_fold[fold][model] = float(row["nmi"])
    return by_fold


def plot_fig2_dual(csv_path: Path, summary_path: Path, out: Path) -> None:
    apply_paper_style()
    by_fold = _load_joint_csv(csv_path)
    thesis = THESIS_HMM_REF
    if summary_path.is_file():
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        thesis = data.get("thesis_baseline", {}).get("hmm_features", {}).get("nmi", thesis)
        joint = data.get("joint_by_model", {})
    else:
        joint = {}

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(5.5, 5.2), gridspec_kw={"height_ratios": [1, 1.15], "hspace": 0.38})

    # --- Panel A: head-to-head mean (cGMVAE vs cHMM) + thesis reference ---
    panel_label(ax_a, "A")
    compare_keys = ("cgmvae_locked", "chmmgmvae_locked")
    means, mins, maxs = [], [], []
    for key in compare_keys:
        if key in joint:
            means.append(joint[key]["mean_nmi"])
            mins.append(joint[key]["min_nmi"])
            maxs.append(joint[key]["max_nmi"])
        else:
            means.append(np.nan)
            mins.append(np.nan)
            maxs.append(np.nan)
    x = np.arange(len(compare_keys))
    colors = [LADDER_COLORS[k] for k in compare_keys]
    bars = ax_a.bar(x, means, color=colors, width=0.55, edgecolor="white", linewidth=0.8, zorder=3)
    for i, (m, lo, hi) in enumerate(zip(means, mins, maxs)):
        if np.isfinite(m):
            ax_a.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor="#334155", capsize=4, lw=1.2, zorder=4)
            ax_a.text(i, hi + 0.03, f"{m:.2f}", ha="center", fontsize=9, fontweight="bold")
    ax_a.axhline(thesis, color=LADDER_COLORS["hmm_features"], linestyle="--", lw=1.2, alpha=0.85, zorder=1)
    ax_a.text(1.45, thesis + 0.015, f"HMM baseline {thesis:.2f}", fontsize=7.5, color=LADDER_COLORS["hmm_features"])
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([LADDER_LABELS[k] for k in compare_keys])
    ax_a.set_ylabel("Prior NMI (joint holdout)")
    ax_a.set_ylim(0, 0.78)
    ax_a.set_title("Population mean across folds", fontsize=9, loc="left", pad=6)

    delta = means[1] - means[0] if all(np.isfinite(v) for v in means) else np.nan
    if np.isfinite(delta):
        ax_a.text(0.98, 0.95, f"$\\Delta$ = +{delta:.2f}", transform=ax_a.transAxes, ha="right", va="top",
                  fontsize=9, fontweight="bold", color=LADDER_COLORS["chmmgmvae_locked"])

    # --- Panel B: per-fold grouped ---
    panel_label(ax_b, "B")
    width = 0.34
    x = np.arange(len(FOLD_ORDER))
    for i, key in enumerate(compare_keys):
        vals = [by_fold[f].get(key, np.nan) for f in FOLD_ORDER]
        offset = (i - 0.5) * width
        bars = ax_b.bar(x + offset, vals, width, label=LADDER_LABELS[key], color=LADDER_COLORS[key],
                        edgecolor="white", linewidth=0.6)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax_b.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)

    ax_b.axhline(thesis, color=LADDER_COLORS["hmm_features"], linestyle="--", lw=1.2, alpha=0.85)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([f"Fold {f}" for f in FOLD_ORDER])
    ax_b.set_ylabel("Prior NMI (best of 3 seeds)")
    ax_b.set_ylim(0, 0.78)
    ax_b.legend(frameon=False, loc="upper right")
    ax_b.set_title("Per-fold holdout (labs 2, 3, 5 pooled)", fontsize=9, loc="left", pad=6)

    save_figure(fig, out)
    plt.close(fig)


def plot_by_fold(csv_path: Path, out: Path) -> None:
    """S1 Fig — same as panel B but full width."""
    apply_paper_style()
    by_fold = _load_joint_csv(csv_path)
    compare_keys = ("cgmvae_locked", "chmmgmvae_locked")
    width = 0.34
    x = np.arange(len(FOLD_ORDER))
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for i, key in enumerate(compare_keys):
        vals = [by_fold[f].get(key, np.nan) for f in FOLD_ORDER]
        offset = (i - 0.5) * width
        ax.bar(x + offset, vals, width, label=LADDER_LABELS[key], color=LADDER_COLORS[key], edgecolor="white")
    ax.axhline(THESIS_HMM_REF, color=LADDER_COLORS["hmm_features"], linestyle="--", lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {f}" for f in FOLD_ORDER])
    ax.set_ylabel("Prior NMI")
    ax.set_ylim(0, 0.78)
    ax.legend(frameon=False)
    save_figure(fig, out)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=REPO / "paper/overleaf/tables/holdout_ladder.csv")
    parser.add_argument("--summary", type=Path, default=REPO / "paper/overleaf/tables/holdout_ladder_summary.json")
    parser.add_argument("--out", type=Path, default=REPO / "docs/paper/figures/Fig2.pdf")
    parser.add_argument("--by-fold", action="store_true", help="S1 only (skip dual Fig 2)")
    args = parser.parse_args()

    if args.by_fold:
        out = REPO / "docs/paper/figures/S1_ladder_by_fold.pdf"
        plot_by_fold(args.csv, out)
        import shutil
        shutil.copy(out, REPO / "paper/overleaf/figures/s1_ladder_by_fold.pdf")
        print(f"Wrote {out}")
    else:
        plot_fig2_dual(args.csv, args.summary, args.out)
        import shutil
        shutil.copy(args.out, REPO / "paper/overleaf/figures/figure2_ladder.pdf")
        print(f"Wrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
