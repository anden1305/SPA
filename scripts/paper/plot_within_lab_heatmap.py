#!/usr/bin/env python3
"""Fig 3 / S2: within-lab holdout NMI heatmap (lab × model) with Δ annotations."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.paper.plot_style import apply_paper_style, panel_label, save_figure

REPO = Path(__file__).resolve().parents[2]
LABS = ("lab_2", "lab_3", "lab_5")
MODELS = ("cgmvae_locked", "chmmgmvae_locked")
MODEL_LABELS = ("cGMVAE", "cHMM–GMVAE")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=REPO / "paper/overleaf/tables/holdout_ladder.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "docs/paper/figures/archive/S2_within_lab_heatmap.pdf",
    )
    args = parser.parse_args()

    # lab -> model -> list of nmi (best per fold)
    data: dict[str, dict[str, list[float]]] = {lab: {m: [] for m in MODELS} for lab in LABS}
    with args.csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("scope") != "within_lab":
                continue
            lab, model = row["lab"], row["model"]
            if lab in data and model in data[lab]:
                data[lab][model].append(float(row["nmi"]))

    mat = np.full((len(LABS), len(MODELS)), np.nan)
    for i, lab in enumerate(LABS):
        for j, model in enumerate(MODELS):
            vals = data[lab][model]
            if vals:
                mat[i, j] = float(np.mean(vals))

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0.25, vmax=0.78)
    panel_label(ax, "A", x=-0.18, y=1.10)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(MODEL_LABELS, fontsize=8)
    ax.set_yticks(range(len(LABS)))
    ax.set_yticklabels([l.replace("_", " ").title() for l in LABS], fontsize=8)
    for i in range(len(LABS)):
        for j in range(len(MODELS)):
            if np.isfinite(mat[i, j]):
                tc = "white" if mat[i, j] > 0.55 else "#1E293B"
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color=tc, fontsize=9, fontweight="bold")
        cgmv = mat[i, 0]
        chmm = mat[i, 1]
        if np.isfinite(cgmv) and np.isfinite(chmm):
            delta = chmm - cgmv
            sign = "+" if delta >= 0 else ""
            ax.text(1.55, i, f"{sign}{delta:.2f}", va="center", fontsize=7.5, color="#475569")
    ax.set_title("Within-lab holdout (mean NMI)", fontsize=9)
    ax.set_xlabel("Model", fontsize=8)
    ax.set_ylabel("Laboratory", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.12, label="Prior NMI")
    cbar.ax.tick_params(labelsize=7)
    fig.text(0.92, 0.5, "Δ", fontsize=8, ha="center", va="center", color="#475569")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, args.out)
    plt.close(fig)
    staging = REPO / "paper/overleaf/figures/s2_within_lab_heatmap.pdf"
    import shutil
    shutil.copy(args.out, staging)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
