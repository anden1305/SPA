#!/usr/bin/env python3
"""Stack PCA + t-SNE side-by-side panels for S7 Fig (K=4 latent embedding)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.substage_analysis.full_analysis_from_npz import (
    LABEL_COLORS_PRED,
    LABEL_COLORS_TRUE,
    N_SAMPLES,
    SEED,
    _fit_pca_scores,
    remap_labels,
)
from scripts.substage_analysis.labels import PREDICTED_SUBSTAGE_LABEL
from scripts.substage_analysis.plot_substages import _scatter_tsne_panel
from sklearn.manifold import TSNE

REPO = Path(__file__).resolve().parents[2]
DEFAULT_NPZ = (
    REPO
    / "results/cv4fold/paper_k_sweep/fold_4/K4/joint_k_sweep_f4_K4_20260609-183717/plots/3/results.npz"
)
DEFAULT_OUT = REPO / "docs/paper/figures/supplementary/S7_latent_embedding.pdf"


def _macro_names(k: int) -> list[str]:
    return ["Wake", "NREM", "REM"][:k]


def _substage_names(k: int) -> list[str]:
    return [f"Substage {i + 1}" for i in range(k)]


def build_latent_si_figure(
    npz_path: Path,
    out_path: Path,
    *,
    seed: int = SEED,
    n_samples: int = N_SAMPLES,
) -> None:
    data = np.load(npz_path)
    datapoints = data["x_latent"].reshape(-1, data["x_latent"].shape[-1])
    labels_true = data["y_true"].reshape(-1).astype(int)
    labels_pred = remap_labels(data["y_hat"].reshape(-1))
    k_true = int(labels_true.max()) + 1
    k_pred = int(labels_pred.max()) + 1
    names_true = _macro_names(k_true)
    names_pred = _substage_names(k_pred)
    colors_true = LABEL_COLORS_TRUE[:k_true]
    colors_pred = LABEL_COLORS_PRED[:k_pred]

    n = datapoints.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(n_samples, n), replace=False)

    scores, evr, _ = _fit_pca_scores(datapoints)
    pc1 = scores[idx, 0]
    pc2 = scores[idx, 1]
    y_true = labels_true[idx]
    y_pred = labels_pred[idx]

    x = datapoints[idx].astype(np.float64, copy=False)
    perp = float(min(30.0, max(5.0, (len(idx) - 1) / 3.0)))
    emb = TSNE(
        n_components=2,
        perplexity=perp,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(x)

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.2))
    fig.patch.set_facecolor("white")

    for ax, y, names, colors, subtitle in [
        (axes[0, 0], y_true, names_true, colors_true, "Expert labels"),
        (axes[0, 1], y_pred, names_pred, colors_pred, PREDICTED_SUBSTAGE_LABEL),
    ]:
        for c in range(len(names)):
            mask = y == c
            if not np.any(mask):
                continue
            ax.scatter(
                pc1[mask], pc2[mask], s=8, alpha=0.65,
                color=colors[c], label=names[c], edgecolors="none",
            )
        ax.set_xlabel(f"PC1 ({evr[0] * 100:.1f}%)", fontsize=8)
        ax.set_ylabel(f"PC2 ({evr[1] * 100:.1f}%)", fontsize=8)
        ax.set_title(subtitle, fontsize=9)
        ax.legend(loc="best", fontsize=6, frameon=True, markerscale=0.8)
        ax.grid(True, linewidth=0.3, alpha=0.35)
        ax.tick_params(labelsize=7)

    for ax, y, names, colors, subtitle in [
        (axes[1, 0], y_true, names_true, colors_true, "Expert labels"),
        (axes[1, 1], y_pred, names_pred, colors_pred, PREDICTED_SUBSTAGE_LABEL),
    ]:
        _scatter_tsne_panel(
            ax, emb, y, names, colors, title=subtitle, point_size=8, alpha=0.65,
        )
        ax.tick_params(labelsize=7)

    for label, ax in [("A", axes[0, 0]), ("B", axes[1, 0])]:
        ax.text(
            -0.12, 1.06, label, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", ha="left",
        )

    fig.suptitle("Latent embedding ($K{=}4$, fold 4 holdout)", fontsize=10, y=1.01)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    args = parser.parse_args()
    if not args.npz.is_file():
        raise SystemExit(f"Missing NPZ: {args.npz}")
    build_latent_si_figure(args.npz, args.out, seed=args.seed, n_samples=args.n_samples)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
