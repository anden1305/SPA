#!/usr/bin/env python3
"""Compact 3-class macro transition matrix from holdout cHMM--GMVAE predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.paper.plot_confusion_matrix import (
    MACRO_NAMES,
    MACRO_MATRIX_ANNOT_FONTSIZE,
    MACRO_MATRIX_CMAP,
    MACRO_MATRIX_FIGSIZE,
    MACRO_MATRIX_LABEL_FONTSIZE,
    MACRO_MATRIX_TICK_FONTSIZE,
    MACRO_MATRIX_VMAX,
    MACRO_MATRIX_VMIN,
    _macro_matrix_text_color,
    _predicted_macro,
)
from scripts.paper.plot_style import apply_paper_style, save_figure

REPO = Path(__file__).resolve().parents[2]
DEFAULT_NPZ = (
    REPO
    / "results/cv4fold/joint_holdout/fold_3/chmmgmvae_locked"
    / "joint_ho_f3_chmmgmvae_locked_20260610-010029/plots/2/results.npz"
)


def _change_transition_matrix(y_macro: np.ndarray, n_states: int = 3) -> np.ndarray:
    """Row-normalized P(next=j | current=i, next != i); diagonal NaN."""
    y = y_macro.astype(int)
    src, dst = y[:-1], y[1:]
    change = src != dst
    src, dst = src[change], dst[change]

    counts = np.zeros((n_states, n_states), dtype=np.float64)
    if src.size:
        np.add.at(counts, (src, dst), 1.0)

    row_sums = counts.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = counts / row_sums
    m[row_sums[:, 0] == 0, :] = np.nan
    np.fill_diagonal(m, np.nan)
    return m


def plot_macro_transition(npz_path: Path, out_path: Path, *, title: str = "") -> dict:
    apply_paper_style()
    data = np.load(npz_path)
    y_true = np.clip(data["y_true"].reshape(-1).astype(int), 0, 2)
    y_pred = data["y_hat"].reshape(-1)
    valid = y_true < 3
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    y_macro = _predicted_macro(y_true, y_pred)

    m = _change_transition_matrix(y_macro)

    cmap = plt.get_cmap(MACRO_MATRIX_CMAP).copy()
    cmap.set_bad(color="#f3f4f6", alpha=1.0)

    fig, ax = plt.subplots(figsize=MACRO_MATRIX_FIGSIZE)
    masked = np.ma.masked_invalid(m)
    im = ax.imshow(masked, aspect="equal", cmap=cmap, vmin=MACRO_MATRIX_VMIN, vmax=MACRO_MATRIX_VMAX)

    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels(MACRO_NAMES, fontsize=MACRO_MATRIX_TICK_FONTSIZE)
    ax.set_yticklabels(MACRO_NAMES, fontsize=MACRO_MATRIX_TICK_FONTSIZE)
    ax.set_xlabel("Next macro", fontsize=MACRO_MATRIX_LABEL_FONTSIZE, labelpad=2)
    ax.set_ylabel("Current macro", fontsize=MACRO_MATRIX_LABEL_FONTSIZE, labelpad=2)
    ax.tick_params(length=0, pad=1)
    if title:
        ax.set_title(title, fontsize=MACRO_MATRIX_LABEL_FONTSIZE, pad=4)

    for i in range(3):
        for j in range(3):
            val = m[i, j]
            if np.isfinite(val):
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=MACRO_MATRIX_ANNOT_FONTSIZE,
                    fontweight="bold" if i != j else "normal",
                    color=_macro_matrix_text_color(val),
                )
            elif i == j:
                ax.text(j, i, "—", ha="center", va="center", fontsize=MACRO_MATRIX_ANNOT_FONTSIZE, color="#9CA3AF")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("P(next | change)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    save_figure(fig, out_path)
    plt.close(fig)

    summary = {
        "npz": str(npz_path),
        "n_epochs": int(y_true.size),
        "transition_change_only": {
            MACRO_NAMES[i]: {
                MACRO_NAMES[j]: (None if not np.isfinite(m[i, j]) else float(m[i, j]))
                for j in range(3)
            }
            for i in range(3)
        },
    }
    out_path.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "docs/paper/figures/main/Fig_macro_transition_f3.pdf",
    )
    parser.add_argument("--title", type=str, default="")
    args = parser.parse_args()
    if not args.npz.is_file():
        raise FileNotFoundError(args.npz)
    summary = plot_macro_transition(args.npz, args.out, title=args.title)
    print(f"Wrote {args.out}")
    print(json.dumps(summary["transition_change_only"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
