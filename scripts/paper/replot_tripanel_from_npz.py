#!/usr/bin/env python3
"""Replot tripanel PCA figures from results.npz with per-class alpha and z-order."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.visuals.layered_scatter import scatter_layered

REPO = Path(__file__).resolve().parents[2]
PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]
MACRO_NAMES = ["Awake", "NREM", "REM", "Artifact"]


def _sample_balanced(y: np.ndarray, max_per_class: int = 2000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx: list[int] = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        n = min(max_per_class, len(cls_idx))
        idx.extend(rng.choice(cls_idx, size=n, replace=False))
    return np.array(idx, dtype=int)


def replot(npz_path: Path, out_dir: Path | None = None, *, max_per_class: int = 2000) -> list[Path]:
    out_dir = out_dir or npz_path.parent
    data = np.load(npz_path)
    x = data["x_latent"].reshape(-1, data["x_latent"].shape[-1])
    y_true = data["y_true"].ravel()
    y_pred = data["y_hat"].ravel()

    idx = _sample_balanced(y_true, max_per_class=max_per_class)
    x = x[idx]
    y_true = y_true[idx]
    y_pred = y_pred[idx]

    k = min(3, max(2, x.shape[1]))
    u, s, _ = np.linalg.svd(x - x.mean(0, keepdims=True), full_matrices=False)
    proj = u[:, :k] * s[:k]

    saved: list[Path] = []
    for a, b in combinations(range(proj.shape[1]), 2):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
        for ax, arr, title, names in [
            (axes[0], y_pred, "cHMMGMVAE predicted", None),
            (axes[1], y_true, "True", MACRO_NAMES[: int(y_true.max()) + 1]),
        ]:
            lut = scatter_layered(
                ax,
                proj[:, a],
                proj[:, b],
                arr,
                label_names=names,
                palette=PALETTE,
                point_size=6,
            )
            ax.set_title(f"{title} (PC{a + 1} vs PC{b + 1})")
            ax.set_xlabel(f"PC{a + 1}")
            unique = np.unique(arr)
            if names and len(names) > int(unique.max()):
                texts = [names[int(v)] for v in unique]
            else:
                texts = [f"State {int(v)}" for v in unique]
            ax.legend(
                handles=[
                    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=lut[int(v)], label=texts[i], markersize=8)
                    for i, v in enumerate(unique)
                ],
                loc="best",
                title="States",
                fontsize=8,
            )
        axes[0].set_ylabel(f"PC{b + 1}")
        fig.tight_layout()
        out_path = out_dir / f"tripanel_pc{a + 1}_pc{b + 1}.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        saved.append(out_path)
        print(f"Wrote {out_path}")
    return saved


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("npz", type=Path, help="Path to plots/results.npz")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--max-per-class", type=int, default=2000)
    args = p.parse_args()
    replot(args.npz.resolve(), args.out_dir, max_per_class=args.max_per_class)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
