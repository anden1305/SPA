from __future__ import annotations

import os
from typing import Sequence, Optional

import numpy as np
import matplotlib.pyplot as plt


def plot_label_distribution(
    y_pred: np.ndarray,                 # (N,) int labels
    label_names: Sequence[str],         # length L, labels 0..L-1
    plot_title: str,
    save_path: Optional[str] = None,
    *,
    sort_by_count: bool = False,
    show_percent_on_bars: bool = True,
    figsize: tuple = (10, 5),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Plots distribution of labels: counts + percentage contribution per label.

    Returns (counts, percentages) in the plotted order.
    """
    if not isinstance(y_pred, np.ndarray) or y_pred.ndim != 1:
        raise ValueError(f"y_pred must be a numpy array of shape (N,). Got {getattr(y_pred,'shape',None)}.")
    if len(label_names) == 0:
        raise ValueError("label_names must be non-empty.")
    if not isinstance(plot_title, str) or not plot_title.strip():
        raise ValueError("plot_title must be a non-empty string.")

    # Ensure integer labels
    if not np.issubdtype(y_pred.dtype, np.integer):
        if np.all(np.isfinite(y_pred)) and np.all(np.equal(y_pred, np.round(y_pred))):
            y_pred = y_pred.astype(int)
        else:
            raise ValueError("y_pred must be integer dtype (or safely castable to integers).")

    L = len(label_names)
    if y_pred.size == 0:
        raise ValueError("y_pred is empty.")
    if y_pred.min() < 0 or y_pred.max() >= L:
        raise ValueError(f"y_pred must be in [0, {L-1}]. Got min={y_pred.min()}, max={y_pred.max()}.")

    counts = np.bincount(y_pred, minlength=L).astype(np.int64)
    total = counts.sum()
    perc = counts / total * 100.0

    order = np.arange(L)
    if sort_by_count:
        order = np.argsort(-counts)  # descending

    counts_p = counts[order]
    perc_p = perc[order]
    names_p = [label_names[i] for i in order]

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(L)
    bars = ax.bar(x, counts_p)

    ax.set_title(plot_title)
    ax.set_ylabel("Count")
    ax.set_xticks(x)
    ax.set_xticklabels(names_p, rotation=45, ha="right")

    # Annotate: "count (xx.x%)"
    if show_percent_on_bars:
        ymax = counts_p.max() if counts_p.size else 1
        pad = max(1.0, 0.02 * ymax)
        for b, c, p in zip(bars, counts_p, perc_p):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + pad,
                f"{int(c)} ({p:.1f}%)",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    ax.grid(True, axis="y", linewidth=0.5, alpha=0.35)
    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return counts_p, perc_p
