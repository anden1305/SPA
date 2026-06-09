#!/usr/bin/env python3
"""Bar chart for paper Fig 2 / S1 from holdout_ladder_summary.json or CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DISPLAY = {
    "hmm_features": "HMM\n(features)",
    "hmmgmvae_locked": "HMMGMVAE",
    "cgmvae_locked": "cGMVAE",
    "chmmgmvae_locked": "cHMMGMVAE",
}
MODEL_KEYS = ("hmmgmvae_locked", "cgmvae_locked", "chmmgmvae_locked")
FOLD_ORDER = ("1", "2", "3", "4")
COLORS = ["#9e9e9e", "#4daf4a", "#377eb8", "#984ea3"]


def _load_by_fold_csv(csv_path: Path) -> dict[str, dict[str, float]]:
    """fold -> model -> nmi (joint scope only)."""
    by_fold: dict[str, dict[str, float]] = {f: {} for f in FOLD_ORDER}
    if not csv_path.is_file():
        return by_fold
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("scope") != "joint":
                continue
            fold = row["fold"]
            model = row["model"]
            if fold in by_fold and model in DISPLAY:
                by_fold[fold][model] = float(row["nmi"])
    return by_fold


def plot_mean(summary_path: Path, out: Path, thesis: float) -> None:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    joint = data.get("joint_by_model", {})

    means, labels = [thesis], [DISPLAY["hmm_features"]]
    for key in MODEL_KEYS:
        means.append(joint[key]["mean_nmi"] if key in joint else np.nan)
        labels.append(DISPLAY[key])

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(x, means, color=COLORS[: len(labels)], edgecolor="black", linewidth=0.5)
    ax.axhline(thesis, color="#9e9e9e", linestyle="--", linewidth=1, label="Thesis HMM ref.")
    ax.set_ylabel("Holdout prior NMI (mean over folds)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 0.75)
    for b, v in zip(bars, means):
        if np.isfinite(v):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
        else:
            ax.text(b.get_x() + b.get_width() / 2, 0.05, "pending", ha="center", fontsize=8, rotation=90)
    ax.set_title("Joint cross-lab holdout (locked recipes)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".tif"), dpi=300, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def plot_by_fold(csv_path: Path, out: Path, thesis: float) -> None:
    by_fold = _load_by_fold_csv(csv_path)
    n_folds = len(FOLD_ORDER)
    n_models = len(MODEL_KEYS)
    width = 0.2
    x = np.arange(n_folds)

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, key in enumerate(MODEL_KEYS):
        vals = [by_fold[f].get(key, np.nan) for f in FOLD_ORDER]
        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=DISPLAY[key], color=COLORS[i + 1], edgecolor="black", linewidth=0.4)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=7)

    ax.axhline(thesis, color="#9e9e9e", linestyle="--", linewidth=1, label="HMM ref. 0.51")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {f}" for f in FOLD_ORDER])
    ax.set_ylabel("Prior NMI (best-of-three)")
    ax.set_ylim(0, 0.8)
    ax.set_title("Joint holdout by fold (S1 Fig)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".tif"), dpi=300, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=REPO / "paper/overleaf/tables/holdout_ladder_summary.json",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=REPO / "paper/overleaf/tables/holdout_ladder.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "paper/overleaf/figures/figure2_ladder.pdf",
    )
    parser.add_argument(
        "--by-fold",
        action="store_true",
        help="Grouped bars per fold (S1 Fig)",
    )
    args = parser.parse_args()

    thesis = 0.51
    if args.summary.is_file():
        data = json.loads(args.summary.read_text(encoding="utf-8"))
        thesis = data.get("thesis_baseline", {}).get("hmm_features", {}).get("nmi", 0.51)

    if args.by_fold:
        out = args.out if args.out.name != "figure2_ladder.pdf" else REPO / "docs/paper/figures/S1_ladder_by_fold.pdf"
        plot_by_fold(args.csv, out, thesis)
        staging = REPO / "paper/overleaf/figures/s1_ladder_by_fold.pdf"
        staging.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(out, staging)
        print(f"Wrote {out}")
    else:
        plot_mean(args.summary, args.out, thesis)
        staging_tif = REPO / "docs/paper/figures/Fig2.tif"
        staging_tif.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        tif_src = args.out.with_suffix(".tif")
        if tif_src.is_file():
            shutil.copy(tif_src, staging_tif)
        print(f"Wrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
