#!/usr/bin/env python3
"""S8 Fig: paired fold-level NMI gain (cHMM--GMVAE minus cGMVAE, best-of-3)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.paper.plot_style import apply_paper_style, save_figure

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO / "paper/overleaf/tables/holdout_ladder.csv"


def _joint_nmi(csv_path: Path, model: str) -> dict[str, float]:
    out: dict[str, float] = {}
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["scope"] == "joint" and row["lab"] == "all" and row["model"] == model:
                out[row["fold"]] = float(row["nmi"])
    return out


def plot_fold_delta(csv_path: Path, out_path: Path) -> None:
    apply_paper_style()
    cgm = _joint_nmi(csv_path, "cgmvae_locked")
    chmm = _joint_nmi(csv_path, "chmmgmvae_locked")
    folds = sorted(set(cgm) & set(chmm), key=int)
    deltas = [chmm[f] - cgm[f] for f in folds]

    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    x = np.arange(len(folds))
    colors = ["#2563EB" if d > 0 else "#DC2626" for d in deltas]
    ax.bar(x, deltas, color=colors, edgecolor="#1E3A5F", linewidth=0.6)
    ax.axhline(0, color="#64748B", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {f}" for f in folds], fontsize=9)
    ax.set_ylabel(r"$\Delta$ prior NMI (cHMM $-$ cGMVAE)", fontsize=9)
    ax.set_ylim(0, max(deltas) * 1.15)
    for i, d in enumerate(deltas):
        ax.text(i, d + 0.01, f"{d:+.2f}", ha="center", va="bottom", fontsize=8)
    ax.text(0.02, 0.98, f"mean $\\Delta$ = {np.mean(deltas):+.3f}", transform=ax.transAxes, va="top", fontsize=8)
    save_figure(fig, out_path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "docs/paper/figures/supplementary/S8_fold_delta.pdf",
    )
    args = parser.parse_args()
    plot_fold_delta(args.csv, args.out)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
