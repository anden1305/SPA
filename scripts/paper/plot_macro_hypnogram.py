#!/usr/bin/env python3
"""Dual-track macro hypnogram (expert vs Hungarian-aligned predicted Wake/NREM/REM)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from scripts.paper.plot_confusion_matrix import MACRO_NAMES, _predicted_macro
from scripts.paper.plot_figure27_compact import (
    LABEL_COLORS_TRUE,
    _draw_classic_hypnogram,
    _macro_y_values,
    _time_hours,
)
from scripts.paper.plot_style import apply_paper_style, save_figure

REPO = Path(__file__).resolve().parents[2]
DEFAULT_NPZ = (
    REPO
    / "results/cv4fold/joint_holdout/fold_3/chmmgmvae_locked"
    / "joint_ho_f3_chmmgmvae_locked_20260610-010029/plots/2/results.npz"
)
PRED_COLORS = ["#3B82F6", "#F59E0B", "#10B981"]  # Wake, NREM, REM


def _pick_excerpt(
    y_true,
    y_pred_macro,
    *,
    minutes: float,
    epoch_sec: float = 4.0,
    min_macro_frac: float = 0.05,
) -> int:
    """Start index of a window with expert and predicted REM and all predicted macros visible."""
    import numpy as np

    n_win = int(minutes * 60 / epoch_sec)
    if y_true.size <= n_win:
        return 0
    best_start = 0
    best_score = -1.0
    step = max(1, n_win // 40)
    for start in range(0, y_true.size - n_win, step):
        end = start + n_win
        true_sl = y_true[start:end]
        pred_sl = y_pred_macro[start:end]
        pred_fracs = [(pred_sl == i).mean() for i in range(3)]
        if min(pred_fracs) < min_macro_frac:
            continue
        true_rem = int((true_sl == 2).sum())
        pred_rem = int((pred_sl == 2).sum())
        if true_rem == 0 or pred_rem == 0:
            continue
        # Prefer balanced predicted macros; tie-break on REM count.
        score = min(pred_fracs) * 100.0 + 0.01 * (true_rem + pred_rem)
        if score > best_score:
            best_score = score
            best_start = start
    if best_score >= 0:
        return best_start
    # Fallback: any predicted REM with some Wake.
    for start in range(0, y_true.size - n_win, step):
        end = start + n_win
        pred_sl = y_pred_macro[start:end]
        if (pred_sl == 2).any() and (pred_sl == 0).any():
            return start
    return 0


def plot_macro_hypnogram(
    npz_path: Path,
    out_path: Path,
    *,
    minutes: float = 30.0,
    epoch_sec: float = 4.0,
) -> dict:
    import numpy as np

    apply_paper_style()
    data = np.load(npz_path)
    y_true = np.clip(data["y_true"].reshape(-1).astype(int), 0, 2)
    y_pred = data["y_hat"].reshape(-1)
    valid = y_true < 3
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    y_macro = _predicted_macro(y_true, y_pred)

    start = _pick_excerpt(y_true, y_macro, minutes=minutes, epoch_sec=epoch_sec)
    n_epochs = int(minutes * 60 / epoch_sec)
    end = min(start + n_epochs, y_true.size)
    true = y_true[start:end]
    pred = y_macro[start:end]
    n = true.size
    time_h = _time_hours(n, epoch_sec)
    hours = n * epoch_sec / 3600.0

    macro_yticks = np.array([0.0, 1.0, 2.0])
    macro_names = ["REM", "NREM", "Wake"]

    fig = plt.figure(figsize=(5.6, 2.35))
    gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.10)
    ax_exp = fig.add_subplot(gs[0])
    ax_pred = fig.add_subplot(gs[1], sharex=ax_exp)

    _draw_classic_hypnogram(
        ax_exp,
        true,
        _macro_y_values(true),
        LABEL_COLORS_TRUE,
        macro_yticks,
        macro_names,
        time_hours=time_h,
        ylabel="Expert",
        show_xlabel=False,
        hours=hours,
        epoch_sec=epoch_sec,
    )
    _draw_classic_hypnogram(
        ax_pred,
        pred,
        _macro_y_values(pred),
        PRED_COLORS,
        macro_yticks,
        macro_names,
        time_hours=time_h,
        ylabel="Predicted",
        show_xlabel=True,
        hours=hours,
        epoch_sec=epoch_sec,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path)
    plt.close(fig)

    summary = {
        "npz": str(npz_path),
        "start_epoch": int(start),
        "minutes": minutes,
        "n_epochs": int(n),
        "expert_rem_epochs": int((true == 2).sum()),
        "predicted_rem_epochs": int((pred == 2).sum()),
    }
    out_path.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "docs/paper/figures/main/Fig_macro_hypnogram_f3.pdf",
    )
    parser.add_argument("--minutes", type=float, default=30.0)
    args = parser.parse_args()
    if not args.npz.is_file():
        raise FileNotFoundError(args.npz)
    summary = plot_macro_hypnogram(args.npz, args.out, minutes=args.minutes)
    print(f"Wrote {args.out}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
