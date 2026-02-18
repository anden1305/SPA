from __future__ import annotations

import os
import pandas as pd
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def plot_transition_matrix(
    y: np.ndarray,                      # (N,) sequence of state labels over time/order
    label_names: Sequence[str],          # length C (labels 0..C-1)
    plot_name: str,
    save_path: str,
    csv_path: str = None,
    *,
    annotate: bool = True,
    fmt: str = ".2f",
    figsize: tuple = (9, 8),
    font_size: int = 10,
) -> np.ndarray:
    """
    Transition matrix plot that IGNORE self-transitions.

    - Counts transitions only when y[t] != y[t+1].
    - Row-normalizes over *changes of state*:
        P(next=j | current=i, next!=i)
    - Diagonal is empty (NaN) and not annotated.
    - Colors go from white (0) -> blue (1).
    - Annotations are larger and bold.
    - If a state i never transitions to a different state, its row will be all NaN.

    Saves to save_path, returns the plotted matrix M (NaNs on diagonal and possibly whole NaN rows).
    """
    if not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array.")
    if y.ndim != 1:
        raise ValueError(f"y must have shape (N,). Got {y.shape}.")
    if y.size < 2:
        raise ValueError("y must contain at least 2 elements to compute transitions.")
    if not isinstance(plot_name, str) or not plot_name.strip():
        raise ValueError("plot_name must be a non-empty string.")
    if not isinstance(save_path, str) or not save_path.strip():
        raise ValueError("save_path must be a non-empty string.")
    if len(label_names) == 0:
        raise ValueError("label_names must be non-empty.")

    # Ensure integer labels
    if not np.issubdtype(y.dtype, np.integer):
        if np.all(np.isfinite(y)) and np.all(np.equal(y, np.round(y))):
            y = y.astype(int)
        else:
            raise ValueError("y must be integer dtype (or safely castable to integers).")

    C = len(label_names)
    if y.min() < 0:
        raise ValueError("Labels must be >= 0.")
    if y.max() >= C:
        raise ValueError(
            f"Found label {y.max()} but label_names has length {C}. "
            "Expected labels in [0, C-1]."
        )

    # Only transitions that change state
    src_all = y[:-1]
    dst_all = y[1:]
    change_mask = src_all != dst_all
    src = src_all[change_mask]
    dst = dst_all[change_mask]

    counts = np.zeros((C, C), dtype=np.float64)
    if src.size > 0:
        np.add.at(counts, (src, dst), 1.0)

    row_sums = counts.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        M = counts / row_sums

    # Rows with no outgoing change transitions -> all NaN
    no_outgoing = (row_sums[:, 0] == 0)
    M[no_outgoing, :] = np.nan

    # Diagonal empty
    np.fill_diagonal(M, np.nan)

    # White -> Blue colormap
    cmap = LinearSegmentedColormap.from_list(
        "white_to_science_blue",
        ["#ffffff", "#1f77b4"]  # matplotlib default scientific blue
    ).copy()
    cmap.set_bad(alpha=0.0)  # NaNs (diagonal / empty rows) transparent

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    masked = np.ma.masked_invalid(M)
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("P(next | current, change)")

    ax.set_title(plot_name)
    ax.set_xlabel("Next state")
    ax.set_ylabel("Current state")

    ax.set_xticks(np.arange(C))
    ax.set_yticks(np.arange(C))
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.set_yticklabels(label_names)

    if annotate:
        for i in range(C):
            for j in range(C):
                val = M[i, j]
                if np.isfinite(val):
                    ax.text(
                        j,
                        i,
                        f"{val:{fmt}}",
                        ha="center",
                        va="center",
                        fontsize=font_size,
                        fontweight="bold",
                    )

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    
    if csv_path is not None:
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        df = pd.DataFrame(M, index=label_names, columns=label_names)
        df.to_csv(csv_path)

    return M
