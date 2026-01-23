from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union
import os

import numpy as np
import matplotlib.pyplot as plt


def pca_scatter_random_samples(
    datapoints: np.ndarray,                # (N, F)
    labels: np.ndarray,                    # (N,)
    label_names: Sequence[str],            # length C (for labels 0..C-1)
    analysis_name: str,
    seed: int,
    n_samples: int,
    save_path: str,
    *,
    sample_without_replacement: bool = True,
    point_size: float = 18.0,
    alpha: float = 0.75,
    center: bool = True,
    standardize: bool = False,
    return_details: bool = False,
) -> Union[None, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    1) Fits PCA on ALL datapoints (N,F).
    2) Randomly selects n_samples datapoints and projects them into PCA space.
    3) Scatter plot of selected points in (PC1, PC2) colored by their labels.
    4) Legend uses label_names.
    5) Title uses analysis_name.
    6) Saves figure to save_path.

    Notes:
    - PCA is implemented with SVD (no sklearn dependency).
    - If standardize=True, features are z-scored before PCA (after optional centering).
    - Returns (selected_indices, pc_scores_selected, explained_variance_ratio) if return_details=True.
    """
    # -----------------
    # Input validation
    # -----------------
    if not isinstance(datapoints, np.ndarray):
        raise TypeError("datapoints must be a numpy array.")
    if not isinstance(labels, np.ndarray):
        raise TypeError("labels must be a numpy array.")
    if datapoints.ndim != 2:
        raise ValueError(f"datapoints must have shape (N,F). Got {datapoints.shape}.")
    if labels.ndim != 1:
        raise ValueError(f"labels must have shape (N,). Got {labels.shape}.")
    N, F = datapoints.shape
    if labels.shape[0] != N:
        raise ValueError(f"labels length must equal N={N}. Got {labels.shape[0]}.")
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0.")
    if sample_without_replacement and n_samples > N:
        raise ValueError(f"n_samples={n_samples} cannot exceed N={N} without replacement.")
    if len(label_names) == 0:
        raise ValueError("label_names must be non-empty.")
    if not isinstance(analysis_name, str) or not analysis_name.strip():
        raise ValueError("analysis_name must be a non-empty string.")
    if not isinstance(save_path, str) or not save_path.strip():
        raise ValueError("save_path must be a non-empty string.")

    # Ensure labels are integer-like
    if not np.issubdtype(labels.dtype, np.integer):
        # try to safely cast if they look integer-like
        if np.all(np.isfinite(labels)) and np.all(np.equal(labels, np.round(labels))):
            labels = labels.astype(int)
        else:
            raise ValueError("labels must be integer dtype (or safely castable to integers).")

    # Decide number of classes from label_names (assumes labels are 0..C-1)
    C = len(label_names)
    if labels.min() < 0:
        raise ValueError("labels must be >= 0.")
    if labels.max() >= C:
        raise ValueError(
            f"labels contain value {labels.max()} but label_names has length {C}. "
            "Expected labels in [0, C-1]."
        )

    # -----------------
    # PCA fit on ALL data
    # -----------------
    X = datapoints.astype(np.float64, copy=False)

    # Center / standardize
    mu = X.mean(axis=0) if center else np.zeros(F, dtype=np.float64)
    Xc = X - mu

    if standardize:
        sigma = Xc.std(axis=0, ddof=1)
        sigma[sigma == 0.0] = 1.0
        Xc = Xc / sigma

    # SVD: Xc = U S Vt, principal axes are rows of Vt
    # PCs scores for a sample x: (x_centered) @ V, where V = Vt.T
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt.T  # (F, F)
    # explained variance ratio
    # eigenvalues of covariance: (S^2) / (N-1)
    if N > 1:
        eigvals = (S ** 2) / (N - 1)
        explained_variance_ratio = eigvals / eigvals.sum()
    else:
        explained_variance_ratio = np.zeros_like(S)

    # -----------------
    # Sample + project
    # -----------------
    rng = np.random.default_rng(seed)
    if sample_without_replacement:
        idx = rng.choice(N, size=n_samples, replace=False)
    else:
        idx = rng.choice(N, size=n_samples, replace=True)

    X_sel = X[idx]
    X_sel_c = X_sel - mu
    if standardize:
        X_sel_c = X_sel_c / sigma

    # scores in PC space (n_samples, F)
    scores_sel = X_sel_c @ V
    pc1 = scores_sel[:, 0]
    pc2 = scores_sel[:, 1]
    y_sel = labels[idx]

    # -----------------
    # Plot
    # -----------------
    fig, ax = plt.subplots(figsize=(9, 7))

    # Use discrete colormap for C classes
    cmap = plt.get_cmap("tab10" if C <= 10 else "tab20")
    colors = [cmap(i % cmap.N) for i in range(C)]

    # Plot per class to get clean legend entries
    for c in range(C):
        mask = (y_sel == c)
        if not np.any(mask):
            continue
        ax.scatter(
            pc1[mask],
            pc2[mask],
            s=point_size,
            alpha=alpha,
            label=label_names[c],
            color=colors[c],
            edgecolors="none",
        )

    evr1 = explained_variance_ratio[0] if explained_variance_ratio.size > 0 else 0.0
    evr2 = explained_variance_ratio[1] if explained_variance_ratio.size > 1 else 0.0

    ax.set_xlabel(f"PC1 ({evr1*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({evr2*100:.1f}% var)")
    ax.set_title(analysis_name)
    ax.legend(loc="best", frameon=True)
    ax.grid(True, linewidth=0.5, alpha=0.35)

    # -----------------
    # Save
    # -----------------
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    if return_details:
        return idx, scores_sel, explained_variance_ratio
    return None
