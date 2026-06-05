"""Shared PCA tripanel plots (run-level and per-mouse)."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from src.helpers.align_labels import align_labels_hungarian

TRIPANEL_PALETTE = np.array(
    [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
)


def _label_colors(arr: NDArray) -> NDArray:
    unique = np.unique(arr)
    lut = {v: TRIPANEL_PALETTE[i % len(TRIPANEL_PALETTE)] for i, v in enumerate(unique)}
    return np.array([lut[v] for v in arr])


def _subsample_by_class(
    x: NDArray,
    init_arr: NDArray,
    trained_arr: NDArray,
    y: NDArray,
    max_per_class: int = 2000,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    indices: list[int] = []
    for cls in np.unique(y):
        cls_indices = np.where(y == cls)[0]
        if len(cls_indices) > max_per_class:
            sampled = np.random.choice(cls_indices, size=max_per_class, replace=False)
        else:
            sampled = cls_indices
        indices.extend(sampled.tolist())
    idx = np.array(indices)
    return x[idx], init_arr[idx], trained_arr[idx], y[idx]


def save_pca_tripanel_arrays(
    x_latent: NDArray,
    y_true: NDArray,
    *,
    out_dir: Path | str,
    display_name: str,
    y_pred: NDArray | None = None,
    y_init: NDArray | None = None,
    y_trained: NDArray | None = None,
    state_names: list[str] | None = None,
    filename_prefix: str = "hmm_tripanel",
) -> list[Path]:
    """Save tripanel PCA PNGs matching run-level visualizer style.

    Provide either (y_init, y_trained) for training plots or y_pred for validation/per-mouse.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    y = np.asarray(y_true).reshape(-1)
    x = np.asarray(x_latent, dtype=float)
    if x.ndim == 3:
        x = x.reshape(-1, x.shape[-1])
    if x.ndim != 2:
        raise ValueError(f"x_latent must be 2D after flatten; got {x.shape}")

    if y_pred is not None:
        init_arr = np.asarray(y_pred).reshape(-1)
        trained_arr = init_arr.copy()
        titles = [f"{display_name} pred", f"{display_name} pred", "True"]
    elif y_init is not None and y_trained is not None:
        init_arr = np.asarray(y_init).reshape(-1)
        trained_arr = np.asarray(y_trained).reshape(-1)
        titles = [f"{display_name} init", f"{display_name} trained", "True"]
    else:
        raise ValueError("Provide y_pred or both y_init and y_trained")

    try:
        init_arr = align_labels_hungarian(y, init_arr)
        trained_arr = align_labels_hungarian(y, trained_arr)
    except Exception as e:
        print(f"Warning: Could not align labels for PCA tripanel due to: {e}")

    if not (len(y) == len(init_arr) == len(trained_arr) == x.shape[0]):
        raise ValueError(
            f"Length mismatch: y={len(y)}, init={len(init_arr)}, "
            f"trained={len(trained_arr)}, x={x.shape[0]}"
        )

    x, init_arr, trained_arr, y = _subsample_by_class(x, init_arr, trained_arr, y)

    k = min(4, max(2, x.shape[1]))
    u, s, _ = np.linalg.svd(x - x.mean(0, keepdims=True), full_matrices=False)
    proj = u[:, :k] * s[:k]

    arrays = [init_arr, trained_arr, y]
    cols = [_label_colors(a) for a in arrays]
    saved: list[Path] = []

    for a, b in combinations(range(proj.shape[1]), 2):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)
        for ax, arr, c, title in zip(axes, arrays, cols, titles):
            ax.scatter(proj[:, a], proj[:, b], c=c, s=6, alpha=0.85, edgecolors="none")
            ax.set_title(f"{title} (PC{a + 1} vs PC{b + 1})")
            ax.set_xlabel(f"PC{a + 1}")
            unique_labels = np.unique(arr)
            if state_names is not None and len(state_names) >= unique_labels.size:
                label_texts = [state_names[int(label)] for label in unique_labels]
            else:
                label_texts = [f"State {label}" for label in unique_labels]
            legend_elements = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=TRIPANEL_PALETTE[i % len(TRIPANEL_PALETTE)],
                    label=label_texts[i],
                    markersize=8,
                )
                for i, label in enumerate(unique_labels)
            ]
            ax.legend(handles=legend_elements, loc="best", title="States")
        axes[0].set_ylabel(f"PC{b + 1}")
        fig.tight_layout()
        out_path = out / f"{filename_prefix}_pc{a + 1}_pc{b + 1}.png"
        fig.savefig(out_path.as_posix(), dpi=160)
        plt.close(fig)
        saved.append(out_path)
    return saved
