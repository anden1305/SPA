#!/usr/bin/env python3
"""S3 Fig: 3×3 macro confusion matrix (expert vs predicted Wake/NREM/REM)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.paper.plot_style import apply_paper_style, save_figure
from src.helpers.align_labels import align_labels_hungarian

REPO = Path(__file__).resolve().parents[2]
MACRO_NAMES = ["Wake", "NREM", "REM"]
MACRO_MATRIX_CMAP = "Blues"
MACRO_MATRIX_VMIN = 0.0
MACRO_MATRIX_VMAX = 1.0
MACRO_MATRIX_FIGSIZE = (3.35, 3.05)
MACRO_MATRIX_TICK_FONTSIZE = 9
MACRO_MATRIX_LABEL_FONTSIZE = 9
MACRO_MATRIX_ANNOT_FONTSIZE = 8


def _macro_matrix_text_color(value: float) -> str:
    return "white" if value > 0.45 else "#1E3A5F"
DEFAULT_NPZ = (
    REPO
    / "results/cv4fold/joint_holdout/fold_3/chmmgmvae_locked"
    / "joint_ho_f3_chmmgmvae_locked_20260610-010029/plots/2/results.npz"
)


def _dominant_macro_map(y_true: np.ndarray, y_pred: np.ndarray, n_macro: int = 3) -> dict[int, int]:
    yt = np.clip(y_true.astype(int), 0, n_macro - 1)
    mapping: dict[int, int] = {}
    for p in np.unique(y_pred.astype(int)):
        mask = y_pred == p
        if not mask.any():
            continue
        counts = np.bincount(yt[mask], minlength=n_macro)
        mapping[int(p)] = int(np.argmax(counts))
    return mapping


def _predicted_macro(y_true: np.ndarray, y_pred: np.ndarray, n_macro: int = 3) -> np.ndarray:
    """Map each epoch's predicted state to Wake/NREM/REM (Hungarian if K=3, else dominant macro)."""
    yt = np.clip(y_true.astype(int), 0, n_macro - 1)
    yp = y_pred.astype(int)
    n_pred = len(np.unique(yp))
    if n_pred == n_macro:
        try:
            aligned, _, _ = align_labels_hungarian(yt, yp)
            if isinstance(aligned, np.ndarray):
                return np.clip(aligned.astype(int), 0, n_macro - 1)
        except ValueError:
            pass
    macro_map = _dominant_macro_map(yt, yp, n_macro)
    return np.array([macro_map.get(int(p), 0) for p in yp], dtype=int)


def _confusion_matrix_3x3(y_true: np.ndarray, y_pred_macro: np.ndarray, n_macro: int = 3) -> np.ndarray:
    """Row-normalized: rows = expert macro, cols = predicted macro."""
    cm = np.zeros((n_macro, n_macro), dtype=np.float64)
    for i in range(n_macro):
        mask = y_true == i
        if not mask.any():
            continue
        counts = np.bincount(y_pred_macro[mask], minlength=n_macro).astype(float)
        cm[i] = counts / counts.sum()
    return cm


def plot_confusion(npz_path: Path, out_path: Path) -> dict:
    apply_paper_style()
    data = np.load(npz_path)
    y_true = data["y_true"].reshape(-1)
    y_pred = data["y_hat"].reshape(-1)
    valid = y_true < 3
    y_true = np.clip(y_true[valid].astype(int), 0, 2)
    y_pred = y_pred[valid]

    y_pred_macro = _predicted_macro(y_true, y_pred)
    cm = _confusion_matrix_3x3(y_true, y_pred_macro)
    counts = np.zeros((3, 3), dtype=int)
    for i in range(3):
        mask_i = y_true == i
        for j in range(3):
            counts[i, j] = int((mask_i & (y_pred_macro == j)).sum())

    fig, ax = plt.subplots(figsize=MACRO_MATRIX_FIGSIZE)
    im = ax.imshow(cm, cmap=MACRO_MATRIX_CMAP, vmin=MACRO_MATRIX_VMIN, vmax=MACRO_MATRIX_VMAX, aspect="equal")

    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels(MACRO_NAMES, fontsize=MACRO_MATRIX_TICK_FONTSIZE)
    ax.set_yticklabels(MACRO_NAMES, fontsize=MACRO_MATRIX_TICK_FONTSIZE)
    ax.set_xlabel("Predicted macro", fontsize=MACRO_MATRIX_LABEL_FONTSIZE)
    ax.set_ylabel("Expert macro", fontsize=MACRO_MATRIX_LABEL_FONTSIZE)
    ax.tick_params(length=0)

    for i in range(3):
        for j in range(3):
            v = cm[i, j]
            ax.text(
                j, i, f"{v:.2f}\n(n={counts[i, j]})",
                ha="center", va="center", fontsize=MACRO_MATRIX_ANNOT_FONTSIZE,
                color=_macro_matrix_text_color(v),
                fontweight="bold" if i == j else "normal",
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row fraction", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    save_figure(fig, out_path)
    plt.close(fig)

    recall = np.diag(cm)
    summary = {
        "npz": str(npz_path),
        "confusion_row_normalized": {MACRO_NAMES[i]: [float(cm[i, j]) for j in range(3)] for i in range(3)},
        "macro_recall": {MACRO_NAMES[i]: float(recall[i]) for i in range(3)},
        "n_epochs": int(y_true.size),
        "n_rem_expert": int((y_true == 2).sum()),
    }
    out_path.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "docs/paper/figures/main/Fig_macro_confusion_f3.pdf",
    )
    args = parser.parse_args()
    if not args.npz.is_file():
        raise FileNotFoundError(args.npz)
    summary = plot_confusion(args.npz, args.out)
    print(f"Wrote {args.out}")
    print(json.dumps(summary["macro_recall"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
