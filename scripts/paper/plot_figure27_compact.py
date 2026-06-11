#!/usr/bin/env python3
"""Fig 3–4: substage physiology grid + transitions + hypnogram (publication quality)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec

from scripts.paper.plot_style import SUBSTAGE_COLORS, apply_paper_style, panel_label, save_figure
from scripts.substage_analysis.frequency_plot import plot_label_channel_frequency_grid
from scripts.substage_analysis.transition_matrix import plot_transition_matrix

LABEL_COLORS_TRUE = ["#3B82F6", "#F59E0B", "#10B981", "#EF4444"]
MACRO_NAMES = ["Wake", "NREM", "REM"]


def remap_labels(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y)
    ranked_old = np.argsort(-counts)
    ranked_old = ranked_old[counts[ranked_old] > 0]
    mapping = np.empty(counts.shape[0], dtype=int)
    mapping[ranked_old] = np.arange(len(ranked_old))
    return mapping[y]


def _transition_matrix(y: np.ndarray, n_states: int) -> np.ndarray:
    src, dst = y[:-1], y[1:]
    ch = src != dst
    src, dst = src[ch], dst[ch]
    counts = np.zeros((n_states, n_states), dtype=np.float64)
    if src.size:
        np.add.at(counts, (src, dst), 1.0)
    row_sums = counts.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = counts / row_sums
    m[row_sums[:, 0] == 0, :] = np.nan
    np.fill_diagonal(m, np.nan)
    return m


def plot_hypnogram_dual(
    labels_pred: np.ndarray,
    labels_true: np.ndarray,
    label_colors: list[str],
    out_path: Path,
    *,
    epoch_sec: float = 4.0,
    hours: float = 2.0,
) -> None:
    apply_paper_style()
    n_epochs = min(int(hours * 3600 / epoch_sec), labels_pred.size)
    sub = labels_pred[:n_epochs]
    true = labels_true[:n_epochs]
    t_hours = np.arange(n_epochs) * epoch_sec / 3600.0

    k_pred = int(sub.max()) + 1
    cmap_sub = ListedColormap(label_colors[:k_pred])
    cmap_macro = ListedColormap(LABEL_COLORS_TRUE[:3])

    fig = plt.figure(figsize=(7.0, 2.0))
    gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.08)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)

    ax0.imshow(sub.reshape(1, -1), aspect="auto", cmap=cmap_sub, vmin=0, vmax=k_pred - 1)
    ax0.set_ylabel("Substates", fontsize=8)
    ax0.set_yticks([])
    panel_label(ax0, "B", x=-0.04, y=1.15)

    true_clip = np.clip(true, 0, 2)
    ax1.imshow(true_clip.reshape(1, -1), aspect="auto", cmap=cmap_macro, vmin=0, vmax=2)
    ax1.set_ylabel("Expert", fontsize=8)
    ax1.set_yticks([])
    ax1.set_xlabel("Time (h)")
    ax1.set_xticks(np.linspace(0, n_epochs - 1, 5))
    ax1.set_xticklabels([f"{t:.1f}" for t in np.linspace(0, hours, 5)])

    for ax in (ax0, ax1):
        ax.tick_params(axis="x", labelsize=7)
    save_figure(fig, out_path)
    plt.close(fig)


def plot_fig4_combined(
    labels_pred: np.ndarray,
    label_names: list[str],
    label_colors: list[str],
    labels_true: np.ndarray,
    out_path: Path,
    csv_path: Path,
) -> None:
    """Fig 4: transition matrix (A) + hypnogram with expert macro (B)."""
    from matplotlib.colors import LinearSegmentedColormap
    import pandas as pd

    apply_paper_style()
    n_states = len(label_names)
    m = _transition_matrix(labels_pred, n_states)
    if csv_path:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(m, index=label_names, columns=label_names).to_csv(csv_path)

    fig = plt.figure(figsize=(7.0, 4.5))
    gs = GridSpec(2, 1, height_ratios=[1.15, 0.65], hspace=0.42)
    ax_tm = fig.add_subplot(gs[0])

    cmap = LinearSegmentedColormap.from_list("w_blue", ["#ffffff", "#2563EB"]).copy()
    cmap.set_bad(alpha=0.0)
    im = ax_tm.imshow(np.ma.masked_invalid(m), aspect="auto", cmap=cmap, vmin=0, vmax=1.0)
    panel_label(ax_tm, "A", x=-0.18, y=1.08)
    ax_tm.set_xticks(np.arange(n_states))
    ax_tm.set_yticks(np.arange(n_states))
    short = [n.split(" (")[0] for n in label_names]
    ax_tm.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
    ax_tm.set_yticklabels(short, fontsize=7)
    ax_tm.set_xlabel("Next state")
    ax_tm.set_ylabel("Current state")
    for i in range(n_states):
        for j in range(n_states):
            v = m[i, j]
            if np.isfinite(v) and v > 0.05:
                ax_tm.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="#1E3A5F")
    cbar = fig.colorbar(im, ax=ax_tm, fraction=0.046, pad=0.04)
    cbar.set_label("P(next | change)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    ax_sub = fig.add_subplot(gs[1])
    n_epochs = min(1800, labels_pred.size)
    sub = labels_pred[:n_epochs]
    true = np.clip(labels_true[:n_epochs], 0, 2)
    k_pred = int(sub.max()) + 1
    panel_label(ax_sub, "B", x=-0.04, y=1.35)
    ax_sub.imshow(sub.reshape(1, -1), aspect="auto", cmap=ListedColormap(label_colors[:k_pred]), vmin=0, vmax=k_pred - 1)
    ax_sub.set_yticks([])
    ax_sub.set_ylabel("Sub", fontsize=7)
    ax_macro = ax_sub.inset_axes([0, -0.88, 1, 0.72])
    ax_macro.imshow(true.reshape(1, -1), aspect="auto", cmap=ListedColormap(LABEL_COLORS_TRUE), vmin=0, vmax=2)
    ax_macro.set_yticks([])
    ax_macro.set_ylabel("Exp", fontsize=7)
    ax_macro.set_xlabel("Time (2 h excerpt, 4 s epochs)")
    ax_macro.tick_params(axis="x", labelsize=7)

    save_figure(fig, out_path)
    plt.close(fig)


def run(
    npz_path: Path,
    out_panel: Path,
    out_transition: Path,
    out_hypno: Path | None,
    out_fig4: Path | None,
    *,
    max_states: int = 10,
) -> None:
    data = np.load(npz_path)
    raw = data["x"]
    n, t, c, f = raw.shape
    n_epochs = n * t
    labels_true = data["y_true"].reshape(n_epochs)
    labels_pred_full = remap_labels(data["y_hat"].reshape(n_epochs))

    counts = np.bincount(labels_pred_full)
    order = np.argsort(-counts)[:max_states]

    def _topk_remap(y: np.ndarray) -> tuple[np.ndarray, list[str], list[str]]:
        names_full = []
        names_short = []
        out = np.full_like(y, -1)
        for new_id, old_id in enumerate(order):
            out[y == old_id] = new_id
            pct = 100.0 * counts[old_id] / n_epochs
            names_full.append(f"S{new_id + 1} ({pct:.0f}%)")
            names_short.append(f"S{new_id + 1}")
        # Pool rare states into one bucket so transitions/hypnogram keep full timeline
        rare = out < 0
        if rare.any():
            out[rare] = len(order)
            names_full.append(f"Other ({100.0 * rare.sum() / n_epochs:.0f}%)")
            names_short.append("Other")
        return out, names_full, names_short

    labels_pred, names, names_short = _topk_remap(labels_pred_full)
    labels_true_v = labels_true
    raw_flat = raw.reshape(n_epochs, c, f)
    sub_ids = data["sub_ids"].reshape(n_epochs)

    n_pred = int(labels_pred.max()) + 1
    colors = SUBSTAGE_COLORS[:n_pred]

    out_panel.parent.mkdir(parents=True, exist_ok=True)
    n_macro = int(labels_true_v.max()) + 1
    macro_names = ["Wake", "NREM", "REM", "Artifact"][:n_macro]
    plot_label_channel_frequency_grid(
        raw_datapoints=raw_flat,
        sample_rate=128.0,
        y_pred=labels_pred,
        label_names=names,
        channel_names=["EEG1", "EEG2", "EMG"],
        plot_title="",
        label_colors=colors,
        labels_true=labels_true_v,
        label_names_true=macro_names,
        label_colors_true=LABEL_COLORS_TRUE[:n_macro],
        sub_ids=sub_ids,
        channel_freq_ranges=[(0.0, 30), (0.0, 30), (5, 60)],
        input_scale="linear",
        y_scale="linear",
        y_lim_mode="minmax",
        save_path=str(out_panel),
    )

    plot_transition_matrix(
        y=labels_pred,
        label_names=names,
        plot_name="Substage transitions",
        save_path=str(out_transition),
        csv_path=str(out_transition.with_suffix(".csv")),
        figsize=(4.5, 4.0),
        font_size=8,
    )

    if out_hypno is not None:
        plot_hypnogram_dual(labels_pred, labels_true_v, colors, out_hypno)

    if out_fig4 is not None:
        plot_fig4_combined(labels_pred, names_short, colors, labels_true_v, out_fig4, out_transition.with_suffix(".csv"))


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--out-panel", type=Path, default=repo / "docs/paper/figures/archive/Fig3_substage_panels.pdf")
    parser.add_argument("--out-transition", type=Path, default=repo / "docs/paper/figures/archive/Fig4_transition_matrix.pdf")
    parser.add_argument("--out-hypno", type=Path, default=repo / "docs/paper/figures/archive/Fig4_hypnogram.pdf")
    parser.add_argument("--out-fig4", type=Path, default=repo / "docs/paper/figures/main/Fig5.pdf")
    args = parser.parse_args()

    if not args.npz.exists():
        raise FileNotFoundError(args.npz)

    run(args.npz, args.out_panel, args.out_transition, args.out_hypno, args.out_fig4)

    import shutil
    for src, dst in [
        (args.out_panel, repo / "paper/overleaf/figures/figure3_substage_panels.pdf"),
        (args.out_fig4, repo / "paper/overleaf/figures/figure4_combined.pdf"),
    ]:
        if src.exists():
            shutil.copy(src, dst)
    print(f"Wrote {args.out_panel}")
    print(f"Wrote {args.out_fig4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
