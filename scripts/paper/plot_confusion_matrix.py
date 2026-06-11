#!/usr/bin/env python3
"""S3 Fig: macro confusion matrix from holdout results.npz (Hungarian-aligned substates)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from scripts.paper.plot_style import apply_paper_style, panel_label, save_figure
REPO = Path(__file__).resolve().parents[2]
MACRO_NAMES = ["Wake", "NREM", "REM"]


def _dominant_macro_map(y_true: np.ndarray, y_pred: np.ndarray, n_macro: int = 3) -> dict[int, int]:
    """Map each predicted state to its dominant expert macro label."""
    yt = np.clip(y_true.astype(int), 0, n_macro - 1)
    mapping: dict[int, int] = {}
    for p in np.unique(y_pred.astype(int)):
        mask = y_pred == p
        if not mask.any():
            continue
        counts = np.bincount(yt[mask], minlength=n_macro)
        mapping[int(p)] = int(np.argmax(counts))
    return mapping


def _macro_confusion(y_true: np.ndarray, y_pred: np.ndarray, n_macro: int = 3) -> np.ndarray:
    """Row-normalized: P(expert macro | predicted state)."""
    yt = np.clip(y_true.astype(int), 0, n_macro - 1)
    yp = y_pred.astype(int)
    pred_states = sorted(np.unique(yp))
    n_pred = len(pred_states)
    mat = np.zeros((n_pred, n_macro), dtype=np.float64)
    for row, p in enumerate(pred_states):
        mask = yp == p
        counts = np.bincount(yt[mask], minlength=n_macro).astype(float)
        mat[row] = counts / max(counts.sum(), 1.0)
    return mat


def _macro_recall(y_true: np.ndarray, y_pred: np.ndarray, n_macro: int = 3) -> np.ndarray:
    yt = np.clip(y_true.astype(int), 0, n_macro - 1)
    macro_map = _dominant_macro_map(yt, y_pred, n_macro)
    yp_macro = np.array([macro_map.get(int(p), -1) for p in y_pred.astype(int)])
    recall = np.zeros(n_macro, dtype=np.float64)
    for m in range(n_macro):
        mask = yt == m
        if mask.any():
            recall[m] = float((yp_macro[mask] == m).sum() / mask.sum())
    return recall


def plot_confusion(npz_path: Path, out_path: Path) -> dict:
    apply_paper_style()
    data = np.load(npz_path)
    y_true = data["y_true"].reshape(-1)
    y_pred = data["y_hat"].reshape(-1)
    # Drop artifact if present
    valid = y_true < 3
    y_true, y_pred = y_true[valid], y_pred[valid]

    conf = _macro_confusion(y_true, y_pred)
    recall = _macro_recall(y_true, y_pred)
    n_states = conf.shape[0]
    state_labels = [f"S{i + 1}" for i in range(n_states)]

    fig = plt.figure(figsize=(6.2, 3.2))
    gs = GridSpec(1, 2, width_ratios=[1.35, 0.65], wspace=0.35)
    ax_hm = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1])

    im = ax_hm.imshow(conf, aspect="auto", cmap="Blues", vmin=0, vmax=1.0)
    panel_label(ax_hm, "A", x=-0.12, y=1.08)
    ax_hm.set_xticks(np.arange(3))
    ax_hm.set_xticklabels(MACRO_NAMES, fontsize=8)
    ax_hm.set_yticks(np.arange(n_states))
    ax_hm.set_yticklabels(state_labels, fontsize=8)
    ax_hm.set_xlabel("Expert macro label")
    ax_hm.set_ylabel("Predicted state (aligned)")
    for i in range(n_states):
        for j in range(3):
            v = conf[i, j]
            if v > 0.08:
                ax_hm.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                           color="white" if v > 0.5 else "#1E3A5F")
    fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04, label="Fraction")

    panel_label(ax_bar, "B", x=-0.18, y=1.08)
    colors = ["#3B82F6", "#F59E0B", "#10B981"]
    bars = ax_bar.bar(np.arange(3), recall, color=colors, width=0.55, edgecolor="white")
    ax_bar.set_xticks(np.arange(3))
    ax_bar.set_xticklabels(MACRO_NAMES, fontsize=8)
    ax_bar.set_ylabel("Macro recall", fontsize=8)
    ax_bar.set_ylim(0, 1.0)
    for b, v in zip(bars, recall):
        ax_bar.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center", fontsize=7)

    save_figure(fig, out_path)
    plt.close(fig)

    summary = {
        "npz": str(npz_path),
        "n_states": n_states,
        "macro_recall": {MACRO_NAMES[i]: float(recall[i]) for i in range(3)},
        "dominant_macro_per_state": {
            state_labels[i]: MACRO_NAMES[int(np.argmax(conf[i]))] for i in range(n_states)
        },
    }
    out_path.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--npz",
        type=Path,
        default=REPO
        / "results/cv4fold/joint_holdout/fold_4/chmmgmvae_locked"
        / "joint_ho_f4_chmmgmvae_locked_20260609-042207/plots/1/results.npz",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "docs/paper/figures/curated/S3_confusion_exemplar.pdf",
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
