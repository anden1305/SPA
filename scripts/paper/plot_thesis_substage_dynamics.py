#!/usr/bin/env python3
"""Thesis-style substage transition matrix (Fig 28) and hypnogram (Fig 29) from results.npz."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

from scripts.paper.plot_style import SUBSTAGE_COLORS, apply_paper_style, panel_label, save_figure
from scripts.substage_analysis.full_analysis_from_npz import remap_labels

MACRO_NAMES = ["Wake", "NREM", "REM"]
MACRO_ORDER = ["Wake", "NREM", "REM"]
EPOCH_SEC = 4.0

# Light fills for per-row / macro-band backgrounds (dominant expert macro per substage).
MACRO_BAND_STYLES = {
    "Wake": ("#D6EAF8", "Wake"),
    "NREM": ("#FDEBD0", "NREM"),
    "REM": ("#D5F5F5", "REM"),
}

GROUP_STYLES = {
    "Awake": ("#D6EAF8", "Awake"),
    "Awake - NREM Transitions": ("#ECEAE4", "Awake - NREM Transitions"),
    "NREM": ("#FDEBD0", "NREM"),
    "NREM - REM Transitions": ("#EEF5E0", "NREM - REM Transitions"),
    "REM": ("#D5F5F5", "REM"),
    "REM - Awake Transitions": ("#E8EAF6", "REM - Awake Transitions"),
}


@dataclass
class SubstageMeta:
    idx: int
    name: str
    group: str
    y_pos: float
    color: str
    dominant_macro: str
    occupancy: float
    median_bout_s: float


def _bout_lengths(y: np.ndarray, state: int) -> list[int]:
    idx = np.where(y == state)[0]
    if idx.size == 0:
        return []
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks + 1, [idx.size]])
    return [int(idx[e - 1] - idx[s] + 1) for s, e in zip(starts, ends)]


def _classify_substage(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    state: int,
) -> tuple[str, str, float]:
    mask = y_pred == state
    n = int(mask.sum())
    if n == 0:
        return "Awake", "Wake", 0.0
    comp = np.bincount(y_true[mask].astype(int), minlength=3) / n
    dom = int(comp.argmax())
    dom_pct = float(comp[dom])
    bouts = _bout_lengths(y_pred, state)
    med_s = float(np.median(bouts) * EPOCH_SEC) if bouts else 0.0

    if dom == 0:  # Wake
        if dom_pct >= 0.75 and med_s >= 20:
            return "Awake", "Wake", med_s
        return "Awake - NREM Transitions", "Wake", med_s
    if dom == 1:  # NREM
        if dom_pct >= 0.75 and med_s >= 20:
            return "NREM", "NREM", med_s
        if comp[2] >= 0.25:
            return "NREM - REM Transitions", "NREM", med_s
        return "Awake - NREM Transitions", "NREM", med_s
    # REM
    if dom_pct >= 0.55 and med_s >= 16:
        return "REM", "REM", med_s
    return "REM - Awake Transitions", "REM", med_s


def _build_substage_meta(y_pred: np.ndarray, y_true: np.ndarray, k: int) -> list[SubstageMeta]:
    raw: list[SubstageMeta] = []
    for s in range(k):
        group, dom, med_s = _classify_substage(y_pred, y_true, s)
        occ = 100.0 * (y_pred == s).mean()
        raw.append(
            SubstageMeta(
                idx=s,
                name=f"Substage {s + 1}",
                group=group,
                y_pos=0.0,
                color=SUBSTAGE_COLORS[s % len(SUBSTAGE_COLORS)],
                dominant_macro=dom,
                occupancy=occ,
                median_bout_s=med_s,
            )
        )

    ordered: list[SubstageMeta] = []
    y = float(k - 1)
    for macro in MACRO_ORDER:
        members = sorted([m for m in raw if m.dominant_macro == macro], key=lambda m: -m.occupancy)
        for m in members:
            ordered.append(
                SubstageMeta(
                    idx=m.idx,
                    name=m.name,
                    group=m.group,
                    y_pos=y,
                    color=m.color,
                    dominant_macro=m.dominant_macro,
                    occupancy=m.occupancy,
                    median_bout_s=m.median_bout_s,
                )
            )
            y -= 1.0
    return ordered


def _draw_macro_backgrounds(ax, meta: list[SubstageMeta]) -> None:
    """Contiguous macro bands (Wake / NREM / REM) behind substage rows."""
    sorted_meta = sorted(meta, key=lambda item: -item.y_pos)
    i = 0
    while i < len(sorted_meta):
        dom = sorted_meta[i].dominant_macro
        j = i + 1
        while j < len(sorted_meta) and sorted_meta[j].dominant_macro == dom:
            j += 1
        block = sorted_meta[i:j]
        y_bot = min(m.y_pos for m in block) - 0.5
        y_top = max(m.y_pos for m in block) + 0.5
        fill = MACRO_BAND_STYLES.get(dom, MACRO_BAND_STYLES["Wake"])[0]
        ax.axhspan(y_bot, y_top, facecolor=fill, edgecolor="none", zorder=0)
        i = j


def _pick_excerpt(y: np.ndarray, n_epochs: int) -> int:
    """Start index of n_epochs window with most state changes."""
    if y.size <= n_epochs:
        return 0
    best_start, best_score = 0, -1
    step = max(1, n_epochs // 20)
    for start in range(0, y.size - n_epochs, step):
        sub = y[start : start + n_epochs]
        score = int(np.sum(sub[1:] != sub[:-1]))
        if score > best_score:
            best_score, best_start = score, start
    return best_start


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


def plot_transition_matrix_thesis(
    y: np.ndarray,
    k: int,
    out_path: Path,
    *,
    csv_path: Path | None = None,
) -> np.ndarray:
    apply_paper_style()
    names = [f"Substage {i + 1}" for i in range(k)]
    m = _transition_matrix(y, k)

    if csv_path is not None:
        import pandas as pd

        csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(m, index=names, columns=names).to_csv(csv_path)

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    cmap = LinearSegmentedColormap.from_list("w_blue", ["#ffffff", "#1f77b4"]).copy()
    cmap.set_bad(alpha=0.0)
    im = ax.imshow(np.ma.masked_invalid(m), aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(k))
    ax.set_yticks(np.arange(k))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Next state")
    ax.set_ylabel("Current state")
    for i in range(k):
        for j in range(k):
            v = m[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$P(\mathrm{next} \mid \mathrm{current})$", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    save_figure(fig, out_path)
    plt.close(fig)
    return m


def plot_hypnogram_thesis(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    meta: list[SubstageMeta],
    out_path: Path,
    *,
    minutes: float = 30.0,
    start: int | None = None,
) -> int:
    apply_paper_style()
    n_epochs = int(minutes * 60 / EPOCH_SEC)
    if start is None:
        start = _pick_excerpt(y_pred, n_epochs)
    end = min(start + n_epochs, y_pred.size)
    sub = y_pred[start:end]
    n = sub.size

    idx_to_y = {m.idx: m.y_pos for m in meta}
    idx_to_color = {m.idx: m.color for m in meta}
    y_vals = np.array([idx_to_y[int(s)] for s in sub], dtype=float)

    time_min = (np.arange(n) + 0.5) * EPOCH_SEC / 60.0
    dt = EPOCH_SEC / 60.0
    t_edges = np.concatenate([[time_min[0] - dt / 2], time_min[:-1] + dt / 2, [time_min[-1] + dt / 2]])

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    fig.subplots_adjust(left=0.26)

    _draw_macro_backgrounds(ax, meta)

    segments: list[list[tuple[float, float]]] = []
    seg_colors: list[str] = []
    line_color = "#2563EB"
    for i in range(n):
        t0, t1 = t_edges[i], t_edges[i + 1]
        yi = y_vals[i]
        c = line_color
        segments.append([(t0, yi), (t1, yi)])
        seg_colors.append(c)
        if i + 1 < n and y_vals[i + 1] != yi:
            yj = y_vals[i + 1]
            segments.append([(t1, yi), (t1, yj)])
            seg_colors.append(c)

    ax.add_collection(LineCollection(segments, colors=seg_colors, linewidths=1.4, capstyle="butt", zorder=2))
    ax.set_xlim(t_edges[0], t_edges[-1])
    ax.set_ylim(min(m.y_pos for m in meta) - 0.6, max(m.y_pos for m in meta) + 0.6)
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Predicted substage")
    sorted_meta = sorted(meta, key=lambda m: -m.y_pos)
    ax.set_yticks([m.y_pos for m in sorted_meta])
    ax.set_yticklabels([f"{m.name} ({m.dominant_macro})" for m in sorted_meta], fontsize=8)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="0.88", linewidth=0.6, linestyle="-", zorder=0.5)
    ax.tick_params(axis="x", labelsize=8)

    save_figure(fig, out_path)
    plt.close(fig)
    return start


def plot_hypnogram_from_npz(
    npz_path: Path,
    out_path: Path,
    *,
    minutes: float = 30.0,
) -> None:
    """30 min substage hypnogram with per-row macro shading from results.npz."""
    data = np.load(npz_path)
    y_pred = remap_labels(data["y_hat"].reshape(-1))
    y_true = data["y_true"].reshape(-1)
    k = int(y_pred.max()) + 1
    meta = _build_substage_meta(y_pred, y_true, k)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_hypnogram_thesis(y_pred, y_true, meta, out_path, minutes=minutes)


def plot_combined(
    npz_path: Path,
    out_transition: Path,
    out_hypnogram: Path,
    *,
    minutes: float = 30.0,
    combined_path: Path | None = None,
) -> None:
    data = np.load(npz_path)
    y_pred = remap_labels(data["y_hat"].reshape(-1))
    y_true = data["y_true"].reshape(-1)
    k = int(y_pred.max()) + 1
    meta = _build_substage_meta(y_pred, y_true, k)

    out_transition.parent.mkdir(parents=True, exist_ok=True)
    plot_transition_matrix_thesis(
        y_pred,
        k,
        out_transition,
        csv_path=out_transition.with_suffix(".csv"),
    )
    start = plot_hypnogram_thesis(y_pred, y_true, meta, out_hypnogram, minutes=minutes)

    if combined_path is not None:
        from matplotlib.gridspec import GridSpec

        apply_paper_style()
        m = _transition_matrix(y_pred, k)
        names = [f"Substage {i + 1}" for i in range(k)]
        n_epochs = int(minutes * 60 / EPOCH_SEC)
        sub = y_pred[start : start + n_epochs]
        n = sub.size
        idx_to_y = {midx.idx: midx.y_pos for midx in meta}
        y_vals = np.array([idx_to_y[int(s)] for s in sub], dtype=float)
        time_min = (np.arange(n) + 0.5) * EPOCH_SEC / 60.0
        dt = EPOCH_SEC / 60.0
        t_edges = np.concatenate([[time_min[0] - dt / 2], time_min[:-1] + dt / 2, [time_min[-1] + dt / 2]])

        fig = plt.figure(figsize=(7.4, 7.8))
        gs = GridSpec(2, 1, height_ratios=[1.05, 1.0], hspace=0.34)
        ax_tm = fig.add_subplot(gs[0])
        ax_h = fig.add_subplot(gs[1])

        cmap = LinearSegmentedColormap.from_list("w_blue", ["#ffffff", "#1f77b4"]).copy()
        cmap.set_bad(alpha=0.0)
        im = ax_tm.imshow(np.ma.masked_invalid(m), aspect="auto", cmap=cmap, vmin=0, vmax=1.0)
        panel_label(ax_tm, "A", x=-0.14, y=1.06)
        ax_tm.set_xticks(np.arange(k))
        ax_tm.set_yticks(np.arange(k))
        ax_tm.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax_tm.set_yticklabels(names, fontsize=8)
        ax_tm.set_xlabel("Next state")
        ax_tm.set_ylabel("Current state")
        for i in range(k):
            for j in range(k):
                v = m[i, j]
                if np.isfinite(v):
                    ax_tm.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, fontweight="bold")
        cbar = fig.colorbar(im, ax=ax_tm, fraction=0.046, pad=0.04)
        cbar.set_label(r"$P(\mathrm{next} \mid \mathrm{current})$", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        _draw_macro_backgrounds(ax_h, meta)

        segments: list[list[tuple[float, float]]] = []
        seg_colors: list[str] = []
        lc = "#2563EB"
        for i in range(n):
            t0, t1 = t_edges[i], t_edges[i + 1]
            yi = y_vals[i]
            segments.append([(t0, yi), (t1, yi)])
            seg_colors.append(lc)
            if i + 1 < n and y_vals[i + 1] != yi:
                segments.append([(t1, yi), (t1, y_vals[i + 1])])
                seg_colors.append(lc)
        ax_h.add_collection(LineCollection(segments, colors=seg_colors, linewidths=1.3, capstyle="butt", zorder=2))
        panel_label(ax_h, "B", x=-0.12, y=1.06)
        ax_h.set_xlim(t_edges[0], t_edges[-1])
        ax_h.set_ylim(min(mx.y_pos for mx in meta) - 0.6, max(mx.y_pos for mx in meta) + 0.6)
        ax_h.set_xlabel("Time (minutes)")
        ax_h.set_ylabel("Predicted substage")
        ax_h.set_yticks([mx.y_pos for mx in meta])
        ax_h.set_yticklabels([f"{mx.name} ({mx.dominant_macro})" for mx in meta], fontsize=7)
        ax_h.yaxis.grid(True, color="0.88", linewidth=0.6, zorder=1)

        save_figure(fig, combined_path)
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--out-transition", type=Path, required=True)
    parser.add_argument("--out-hypnogram", type=Path, required=True)
    parser.add_argument("--out-combined", type=Path, default=None)
    parser.add_argument("--minutes", type=float, default=30.0)
    args = parser.parse_args()
    if not args.npz.is_file():
        raise FileNotFoundError(args.npz)
    plot_combined(
        args.npz,
        args.out_transition,
        args.out_hypnogram,
        minutes=args.minutes,
        combined_path=args.out_combined,
    )
    print(f"Wrote {args.out_transition}")
    print(f"Wrote {args.out_hypnogram}")
    if args.out_combined:
        print(f"Wrote {args.out_combined}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
