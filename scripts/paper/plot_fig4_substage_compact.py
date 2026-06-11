#!/usr/bin/env python3
"""Fig 4: compact 2×2 substage physiology (main-text biology panel)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from scripts.paper.plot_style import SUBSTAGE_COLORS, apply_paper_style, panel_label, save_figure

REPO = Path(__file__).resolve().parents[2]
MACRO_NAMES = ["Wake", "NREM", "REM"]
DEFAULT_NPZ = (
    REPO
    / "results/cv4fold/paper_k_sweep/fold_4/K4/joint_k_sweep_f4_K4_20260609-183717/plots/3/results.npz"
)


def remap_labels(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y)
    ranked = np.argsort(-counts)
    ranked = ranked[counts[ranked] > 0]
    mapping = np.empty(counts.shape[0], dtype=int)
    mapping[ranked] = np.arange(len(ranked))
    return mapping[y]


def _bout_lengths(y: np.ndarray) -> dict[int, list[float]]:
    out: dict[int, list[float]] = {}
    if y.size == 0:
        return out
    start = 0
    for i in range(1, y.size + 1):
        if i == y.size or y[i] != y[start]:
            lbl = int(y[start])
            out.setdefault(lbl, []).append(float(i - start))
            start = i
    return out


def _state_names(y: np.ndarray, y_true: np.ndarray, n_states: int) -> list[str]:
    """Provisional names from dominant expert macro + physiology."""
    names = []
    for s in range(n_states):
        mask = y == s
        if not mask.any():
            names.append(f"S{s + 1}")
            continue
        macro_counts = np.bincount(np.clip(y_true[mask], 0, 2), minlength=3)
        dom = MACRO_NAMES[int(np.argmax(macro_counts))]
        names.append(dom)
    # Disambiguate duplicate macro names
    seen: dict[str, int] = {}
    out = []
    for n in names:
        if n in seen:
            seen[n] += 1
            out.append(f"{n}′")
        else:
            seen[n] = 1
            out.append(n)
    return out


def plot_compact(npz_path: Path, out_path: Path, sample_rate: float = 128.0) -> None:
    apply_paper_style()
    data = np.load(npz_path)
    raw = data["x"].reshape(-1, data["x"].shape[2], data["x"].shape[3])
    y_true = np.clip(data["y_true"].reshape(-1), 0, 2)
    y_pred = remap_labels(data["y_hat"].reshape(-1))
    n_states = int(y_pred.max()) + 1
    names = _state_names(y_pred, y_true, n_states)
    colors = SUBSTAGE_COLORS[:n_states]

    n_bins = raw.shape[2]
    freqs = np.linspace(0, sample_rate / 2, n_bins)
    theta_m = (freqs >= 4.0) & (freqs < 8.0)
    delta_m = (freqs >= 0.5) & (freqs < 4.0)

    fig = plt.figure(figsize=(7.0, 5.2))
    gs = GridSpec(2, 2, hspace=0.38, wspace=0.32)
    ax_psd = fig.add_subplot(gs[0, 0])
    ax_td = fig.add_subplot(gs[0, 1])
    ax_bout = fig.add_subplot(gs[1, 0])
    ax_mix = fig.add_subplot(gs[1, 1])

    panel_label(ax_psd, "A", x=-0.14, y=1.08)
    panel_label(ax_td, "B", x=-0.14, y=1.08)
    panel_label(ax_bout, "C", x=-0.14, y=1.08)
    panel_label(ax_mix, "D", x=-0.14, y=1.08)

    # A: mean EEG PSD (channel 0)
    for s in range(n_states):
        mask = y_pred == s
        if not mask.any():
            continue
        psd = raw[mask, 0, :].mean(axis=0)
        ax_psd.plot(freqs, psd, color=colors[s], lw=1.5, label=names[s])
    ax_psd.set_xlim(0, 30)
    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_ylabel("Mean EEG power")
    ax_psd.set_title("EEG spectrum", fontsize=9)
    ax_psd.legend(frameon=False, fontsize=7, loc="upper right")

    # B: θ/δ ratio
    ratios = []
    for s in range(n_states):
        mask = y_pred == s
        p = raw[mask, 0, :]
        th = p[:, theta_m].sum(axis=1)
        de = p[:, delta_m].sum(axis=1)
        ratios.append(th / np.maximum(de, 1e-12))
    bp = ax_td.boxplot(ratios, patch_artist=True, widths=0.55, showfliers=False)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    ax_td.set_xticklabels(names, rotation=25, ha="right", fontsize=7)
    ax_td.set_ylabel("θ/δ ratio")
    ax_td.set_title("NREM depth proxy", fontsize=9)

    # C: bout length
    bouts = _bout_lengths(y_pred)
    bout_data = [bouts.get(s, [1.0]) for s in range(n_states)]
    bp2 = ax_bout.boxplot(bout_data, patch_artist=True, widths=0.55, showfliers=False)
    for patch, c in zip(bp2["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    ax_bout.set_xticklabels(names, rotation=25, ha="right", fontsize=7)
    ax_bout.set_ylabel("Bout length (epochs)")
    ax_bout.set_title("Temporal stability", fontsize=9)

    # D: expert macro composition (stacked bar)
    comp = np.zeros((n_states, 3))
    for s in range(n_states):
        mask = y_pred == s
        if mask.any():
            comp[s] = np.bincount(y_true[mask], minlength=3) / mask.sum()
    bottom = np.zeros(n_states)
    macro_colors = ["#3B82F6", "#F59E0B", "#10B981"]
    for m, (mn, mc) in enumerate(zip(MACRO_NAMES, macro_colors)):
        ax_mix.bar(np.arange(n_states), comp[:, m], bottom=bottom, color=mc, label=mn, width=0.6)
        bottom += comp[:, m]
    ax_mix.set_xticks(np.arange(n_states))
    ax_mix.set_xticklabels(names, rotation=25, ha="right", fontsize=7)
    ax_mix.set_ylabel("Expert macro fraction")
    ax_mix.set_ylim(0, 1.0)
    ax_mix.set_title("Macro alignment", fontsize=9)
    ax_mix.legend(frameon=False, fontsize=6, ncol=3, loc="upper right")

    save_figure(fig, out_path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--out", type=Path, default=REPO / "docs/paper/figures/curated/Fig4_substage_biology.pdf")
    args = parser.parse_args()
    if not args.npz.is_file():
        raise FileNotFoundError(args.npz)
    plot_compact(args.npz, args.out)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
