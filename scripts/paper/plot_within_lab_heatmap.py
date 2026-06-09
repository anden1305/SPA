#!/usr/bin/env python3
"""S2 Fig: within-lab holdout NMI heatmap (lab x model)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
LABS = ("lab_2", "lab_3", "lab_5")
MODELS = ("cgmvae_locked", "chmmgmvae_locked")
MODEL_LABELS = ("cGMVAE", "cHMMGMVAE")


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
        default=REPO / "docs/paper/figures/S2_within_lab_heatmap.pdf",
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

    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0.25, vmax=0.75)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(MODEL_LABELS)
    ax.set_yticks(range(len(LABS)))
    ax.set_yticklabels([l.replace("_", " ") for l in LABS])
    for i in range(len(LABS)):
        for j in range(len(MODELS)):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="white", fontsize=10)
    ax.set_title("Within-lab holdout mean NMI")
    fig.colorbar(im, ax=ax, label="Prior NMI")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    fig.savefig(args.out.with_suffix(".tif"), dpi=300, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    staging = REPO / "paper/overleaf/figures/s2_within_lab_heatmap.pdf"
    import shutil
    shutil.copy(args.out, staging)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
