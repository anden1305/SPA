#!/usr/bin/env python3
"""Generate compact Fig-27-style substage panel + transition matrix (paper Fig 3–4)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from scripts.substage_analysis.frequency_plot import plot_label_channel_frequency_grid
from scripts.substage_analysis.transition_matrix import plot_transition_matrix

LABEL_COLORS_PRED = [
    "#6A3D9A", "#1B9E77", "#D95F02", "#7570B3", "#E7298A",
    "#66A61E", "#E6AB02", "#A6761D", "#666666", "#8DD3C7",
    "#BEBADA", "#FB8072", "#80B1D3", "#FDB462", "#B3B3B3",
]
LABEL_COLORS_TRUE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#ff0000"]


def remap_labels(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y)
    ranked_old = np.argsort(-counts)
    ranked_old = ranked_old[counts[ranked_old] > 0]
    mapping = np.empty(counts.shape[0], dtype=int)
    mapping[ranked_old] = np.arange(len(ranked_old))
    return mapping[y]


def run(npz_path: Path, out_panel: Path, out_transition: Path, out_hypno: Path | None) -> None:
    data = np.load(npz_path)
    raw = data["x"]
    n, t, c, f = raw.shape
    n_epochs = n * t
    x_latent = data["x_latent"]
    if x_latent.ndim == 2:
        if x_latent.shape[0] != n_epochs:
            raise ValueError(f"x_latent rows {x_latent.shape[0]} != n*t {n_epochs}")
        x_latent = x_latent.reshape(n, t, -1)
    elif x_latent.ndim != 3:
        raise ValueError(f"Expected x_latent (N,T,L) or (N*T,L), got shape {x_latent.shape}")
    labels_true = data["y_true"].reshape(n_epochs)
    labels_pred = remap_labels(data["y_hat"].reshape(n_epochs))
    raw_flat = raw.reshape(n_epochs, c, f)
    sub_ids = data["sub_ids"].reshape(n_epochs)

    n_pred = int(labels_pred.max()) + 1
    label_names = [f"Substage {i + 1}" for i in range(n_pred)]
    label_names_true = ["Awake", "NREM", "REM", "Artifact"][: int(labels_true.max()) + 1]

    out_panel.parent.mkdir(parents=True, exist_ok=True)
    plot_label_channel_frequency_grid(
        raw_datapoints=raw_flat,
        sample_rate=128.0,
        y_pred=labels_pred,
        label_names=label_names,
        channel_names=["EEG1", "EEG2", "EMG"],
        plot_title="Substage-specific physiology (cHMMGMVAE prior)",
        label_colors=LABEL_COLORS_PRED[:n_pred],
        labels_true=labels_true,
        label_names_true=label_names_true,
        label_colors_true=LABEL_COLORS_TRUE[: len(label_names_true)],
        sub_ids=sub_ids,
        channel_freq_ranges=[(0.0, 30), (0.0, 30), (5, 60)],
        save_path=str(out_panel),
    )

    out_transition.parent.mkdir(parents=True, exist_ok=True)
    plot_transition_matrix(
        y=labels_pred,
        label_names=label_names,
        plot_name="Substage transition matrix (cHMMGMVAE)",
        save_path=str(out_transition),
        csv_path=str(out_transition.with_suffix(".csv")),
    )

    if out_hypno is not None:
        import matplotlib.pyplot as plt

        # First ~2 h at 4 s/epoch → 1800 epochs
        max_epochs = min(1800, labels_pred.size)
        fig, ax = plt.subplots(figsize=(10, 2.5))
        ax.imshow(labels_pred[:max_epochs].reshape(1, -1), aspect="auto", cmap="tab20")
        ax.set_yticks([])
        ax.set_xlabel("Epoch (4 s)")
        ax.set_title("Example substage hypnogram (excerpt)")
        out_hypno.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_hypno, dpi=150)
        plt.close(fig)
        print(f"Wrote {out_hypno}")


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, required=True, help="results.npz from best holdout run")
    parser.add_argument(
        "--out-panel",
        type=Path,
        default=repo / "paper/overleaf/figures/figure3_substage_panels.pdf",
    )
    parser.add_argument(
        "--out-transition",
        type=Path,
        default=repo / "paper/overleaf/figures/figure4_transition_matrix.pdf",
    )
    parser.add_argument(
        "--out-hypno",
        type=Path,
        default=repo / "paper/overleaf/figures/figure4_hypnogram_excerpt.pdf",
    )
    args = parser.parse_args()

    if not args.npz.exists():
        raise FileNotFoundError(
            f"{args.npz} not found. Run holdout/K-sweep with save_results_npz: true first."
        )

    run(args.npz, args.out_panel, args.out_transition, args.out_hypno)
    print(f"Wrote {args.out_panel}")
    print(f"Wrote {args.out_transition}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
