#!/usr/bin/env python3
"""Fig 3–4: substage physiology grid + transitions + hypnogram (publication quality)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from scripts.paper.plot_style import SUBSTAGE_COLORS, apply_paper_style, panel_label, save_figure
from scripts.substage_analysis.fig27_from_npz import plot_fig27_gmm_predicted
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


def _time_hours(n_epochs: int, epoch_sec: float) -> np.ndarray:
    return (np.arange(n_epochs) + 0.5) * epoch_sec / 3600.0


def _macro_y_values(labels: np.ndarray) -> np.ndarray:
    """Wake=0, NREM=1, REM=2 → y with Wake on top (classic hypnogram order)."""
    return np.array([2.0, 1.0, 0.0], dtype=float)[labels]


def _draw_classic_hypnogram(
    ax,
    labels: np.ndarray,
    y_values: np.ndarray,
    colors: list[str],
    yticks: np.ndarray,
    yticklabels: list[str],
    *,
    time_hours: np.ndarray,
    ylabel: str = "",
    show_xlabel: bool = True,
    hours: float = 2.0,
    epoch_sec: float = 4.0,
) -> None:
    """Stepped-line hypnogram with stage labels on the left y-axis."""
    from matplotlib.collections import LineCollection

    n = len(labels)
    if n < 1:
        return
    dt = epoch_sec / 3600.0
    t_edges = np.concatenate([[time_hours[0] - dt / 2], time_hours[:-1] + dt / 2, [time_hours[-1] + dt / 2]])

    segments: list[list[tuple[float, float]]] = []
    seg_colors: list[str] = []
    for i in range(n):
        t0, t1 = t_edges[i], t_edges[i + 1]
        yi = float(y_values[i])
        c = colors[int(labels[i])]
        segments.append([(t0, yi), (t1, yi)])
        seg_colors.append(c)
        if i + 1 < n and y_values[i + 1] != yi:
            yj = float(y_values[i + 1])
            segments.append([(t1, yi), (t1, yj)])
            seg_colors.append(c)

    ax.add_collection(LineCollection(segments, colors=seg_colors, linewidths=1.3, capstyle="butt"))
    ax.set_xlim(t_edges[0], t_edges[-1])
    ax.set_ylim(-0.5, float(yticks.max()) + 0.5)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="0.88", linewidth=0.6, linestyle="-")
    if show_xlabel:
        ax.set_xlabel(f"Time (h, {hours:.0f} h excerpt, {int(epoch_sec)} s epochs)", fontsize=8)
        ax.set_xticks(np.linspace(0, hours, 5))
        ax.tick_params(axis="x", labelsize=7)
    else:
        ax.tick_params(axis="x", labelbottom=False)


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
    """Dual-track hypnogram: expert macro (top) and predicted substates (bottom), classic stepped lines."""
    apply_paper_style()
    n_epochs = min(int(hours * 3600 / epoch_sec), labels_pred.size)
    sub = labels_pred[:n_epochs]
    true = np.clip(labels_true[:n_epochs], 0, 2)

    k_pred = int(sub.max()) + 1
    macro_names = ["REM", "NREM", "Wake"]
    macro_yticks = np.array([0, 1, 2], dtype=float)
    sub_yticks = np.arange(k_pred, dtype=float)
    sub_names = [f"S{k + 1}" for k in range(k_pred)]
    time_h = _time_hours(n_epochs, epoch_sec)

    fig = plt.figure(figsize=(7.2, 3.4))
    gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.12)
    ax_exp = fig.add_subplot(gs[0])
    ax_sub = fig.add_subplot(gs[1], sharex=ax_exp)

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
        ax_sub,
        sub,
        sub.astype(float),
        label_colors[:k_pred],
        sub_yticks,
        sub_names,
        time_hours=time_h,
        ylabel="Predicted",
        show_xlabel=True,
        hours=hours,
        epoch_sec=epoch_sec,
    )

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

    fig = plt.figure(figsize=(7.0, 5.4))
    gs = GridSpec(2, 1, height_ratios=[1.0, 1.05], hspace=0.38)
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

    gs_hyp = gs[1].subgridspec(2, 1, height_ratios=[1, 1], hspace=0.1)
    ax_exp = fig.add_subplot(gs_hyp[0])
    ax_sub = fig.add_subplot(gs_hyp[1], sharex=ax_exp)
    n_epochs = min(1800, labels_pred.size)
    sub = labels_pred[:n_epochs]
    true = np.clip(labels_true[:n_epochs], 0, 2)
    k_pred = int(sub.max()) + 1
    hours = n_epochs * 4.0 / 3600.0
    time_h = _time_hours(n_epochs, 4.0)
    panel_label(ax_exp, "B", x=-0.12, y=1.12)
    _draw_classic_hypnogram(
        ax_exp,
        true,
        _macro_y_values(true),
        LABEL_COLORS_TRUE,
        np.array([0.0, 1.0, 2.0]),
        ["REM", "NREM", "Wake"],
        time_hours=time_h,
        ylabel="Expert",
        show_xlabel=False,
        hours=hours,
        epoch_sec=4.0,
    )
    _draw_classic_hypnogram(
        ax_sub,
        sub,
        sub.astype(float),
        label_colors[:k_pred],
        np.arange(k_pred, dtype=float),
        [f"S{k + 1}" for k in range(k_pred)],
        time_hours=time_h,
        ylabel="Predicted",
        show_xlabel=True,
        hours=hours,
        epoch_sec=4.0,
    )

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
    plot_fig27_gmm_predicted(
        npz_path,
        out_panel,
        k=int(labels_pred_full.max()) + 1,
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
        from scripts.paper.plot_thesis_substage_dynamics import plot_hypnogram_from_npz

        plot_hypnogram_from_npz(npz_path, out_hypno, minutes=30.0)

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
