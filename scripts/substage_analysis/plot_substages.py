from __future__ import annotations

from typing import Sequence, Tuple, Union
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE

from src.visuals.layered_scatter import scatter_layered

sns.set_theme(style="whitegrid", context="paper")

ColorLike = Union[str, Tuple[float, float, float], Tuple[float, float, float, float]]


def pca_scatter_random_samples(
    datapoints: np.ndarray,                # (N, F)
    labels: np.ndarray,                    # (N,)
    label_names: Sequence[str],            # length C (for labels 0..C-1)
    analysis_name: str,
    seed: int,
    n_samples: int,
    save_path: str,
    *,
    label_colors: Sequence[ColorLike],     # length C
    sample_without_replacement: bool = True,
    point_size: float = 18.0,
    alpha: float = 0.75,
    center: bool = True,
    standardize: bool = False,
    return_details: bool = False,
    # --- NEW ---
    trim_outliers: bool = True,
    outlier_percent: float = 0.5,          # total trimmed per axis (e.g., 1.0 -> 0.5% low + 0.5% high)
) -> Union[None, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    ...
    - If trim_outliers=True, removes outliers in (PC1, PC2) by trimming
      `outlier_percent` total per axis using percentile bounds, then plots.
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

    C = len(label_names)
    if len(label_colors) != C:
        raise ValueError(
            f"label_colors must have same length as label_names "
            f"(expected {C}, got {len(label_colors)})."
        )

    # Ensure labels are integer-like
    if not np.issubdtype(labels.dtype, np.integer):
        if np.all(np.isfinite(labels)) and np.all(np.equal(labels, np.round(labels))):
            labels = labels.astype(int)
        else:
            raise ValueError("labels must be integer dtype (or safely castable to integers).")

    if labels.min() < 0:
        raise ValueError("labels must be >= 0.")
    if labels.max() >= C:
        raise ValueError(
            f"labels contain value {labels.max()} but label_names has length {C}. "
            "Expected labels in [0, C-1]."
        )

    if trim_outliers:
        if not (0.0 <= outlier_percent < 100.0):
            raise ValueError("outlier_percent must be in [0, 100).")
        if outlier_percent > 0 and n_samples < 20:
            # not strictly required, but percentiles get silly with tiny samples
            pass

    # -----------------
    # PCA fit on ALL data
    # -----------------
    X = datapoints.astype(np.float64, copy=False)

    mu = X.mean(axis=0) if center else np.zeros(F, dtype=np.float64)
    Xc = X - mu

    if standardize:
        sigma = Xc.std(axis=0, ddof=1)
        sigma[sigma == 0.0] = 1.0
        Xc = Xc / sigma

    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt.T

    if N > 1:
        eigvals = (S ** 2) / (N - 1)
        explained_variance_ratio = eigvals / eigvals.sum()
    else:
        explained_variance_ratio = np.zeros_like(S)

    # -----------------
    # Sample + project
    # -----------------
    rng = np.random.default_rng(seed)
    idx = rng.choice(N, size=n_samples, replace=not sample_without_replacement)

    X_sel = X[idx]
    X_sel_c = X_sel - mu
    if standardize:
        X_sel_c = X_sel_c / sigma

    scores_sel = X_sel_c @ V
    pc1 = scores_sel[:, 0]
    pc2 = scores_sel[:, 1]
    y_sel = labels[idx]

    # -----------------
    # NEW: Trim outliers in PC1/PC2 before plotting
    # -----------------
    if trim_outliers and outlier_percent > 0:
        tail = outlier_percent / 2.0
        lo, hi = tail, 100.0 - tail

        pc1_lo, pc1_hi = np.percentile(pc1, [lo, hi])
        pc2_lo, pc2_hi = np.percentile(pc2, [lo, hi])

        keep = (pc1 >= pc1_lo) & (pc1 <= pc1_hi) & (pc2 >= pc2_lo) & (pc2 <= pc2_hi)

        # filter everything consistently
        pc1 = pc1[keep]
        pc2 = pc2[keep]
        y_sel = y_sel[keep]
        scores_sel = scores_sel[keep]
        idx = idx[keep]

    # -----------------
    # Plot
    # -----------------
    fig, ax = plt.subplots(figsize=(9, 7))

    colors_lut = {c: label_colors[c] for c in range(C)}
    scatter_layered(
        ax,
        pc1,
        pc2,
        y_sel,
        label_names=label_names if "Artifact" in label_names else None,
        colors=colors_lut,
        point_size=point_size,
    )

    evr1 = explained_variance_ratio[0] if explained_variance_ratio.size > 0 else 0.0
    evr2 = explained_variance_ratio[1] if explained_variance_ratio.size > 1 else 0.0

    ax.set_xlabel(f"PC1 ({evr1*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({evr2*100:.1f}% var)")
    ax.set_title(analysis_name)
    ax.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=label_colors[c], label=label_names[c], markersize=8)
            for c in range(C)
            if np.any(y_sel == c)
        ],
        loc="best",
        frameon=True,
    )
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


def _scatter_tsne_panel(
    ax: plt.Axes,
    emb: np.ndarray,
    labels: np.ndarray,
    label_names: Sequence[str],
    label_colors: Sequence[ColorLike],
    *,
    title: str,
    point_size: float = 14.0,
    alpha: float = 0.75,
) -> None:
    colors_lut = {c: label_colors[c] for c in range(len(label_names))}
    scatter_layered(
        ax,
        emb[:, 0],
        emb[:, 1],
        labels,
        label_names=label_names if "Artifact" in label_names else None,
        colors=colors_lut,
        point_size=point_size,
    )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(title)
    ax.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=label_colors[c], label=label_names[c], markersize=8)
            for c in range(len(label_names))
            if np.any(labels == c)
        ],
        loc="best",
        frameon=True,
        fontsize=8,
    )
    ax.grid(True, linewidth=0.4, alpha=0.35)


def tsne_scatter_pair(
    datapoints: np.ndarray,
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
    label_names_true: Sequence[str],
    label_names_pred: Sequence[str],
    label_colors_true: Sequence[ColorLike],
    label_colors_pred: Sequence[ColorLike],
    analysis_name: str,
    save_path_true: str,
    save_path_pred: str,
    *,
    seed: int,
    n_samples: int,
    perplexity: float = 30.0,
    point_size: float = 14.0,
    alpha: float = 0.75,
) -> None:
    """Fit t-SNE on a latent subsample; save expert-label and prediction-label scatters."""
    if datapoints.ndim != 2 or labels_true.ndim != 1 or labels_pred.ndim != 1:
        raise ValueError("datapoints (N,F), labels_true (N,), labels_pred (N,) required.")
    n = datapoints.shape[0]
    if labels_true.shape[0] != n or labels_pred.shape[0] != n:
        raise ValueError("label arrays must match datapoints row count.")
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0.")
    if n_samples > n:
        raise ValueError(f"n_samples={n_samples} cannot exceed N={n}.")

    n_use = min(n_samples, n)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=n_use, replace=False)
    x = datapoints[idx].astype(np.float64, copy=False)
    y_true = labels_true[idx].astype(int, copy=False)
    y_pred = labels_pred[idx].astype(int, copy=False)

    if n_use > 15_000:
        point_size = min(point_size, 8.0)
        alpha = min(alpha, 0.55)

    perp = float(min(perplexity, max(5.0, (n_use - 1) / 3.0)))
    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    emb = tsne.fit_transform(x)

    for path, labels, names, colors, subtitle in [
        (save_path_true, y_true, label_names_true, label_colors_true, "Expert labels"),
        (save_path_pred, y_pred, label_names_pred, label_colors_pred, "GM substages"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 7))
        _scatter_tsne_panel(
            ax,
            emb,
            labels,
            names,
            colors,
            title=f"{analysis_name} — {subtitle}",
            point_size=point_size,
            alpha=alpha,
        )
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def tsne_true_vs_predicted(
    datapoints: np.ndarray,
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
    label_names_true: Sequence[str],
    label_names_pred: Sequence[str],
    label_colors_true: Sequence[ColorLike],
    label_colors_pred: Sequence[ColorLike],
    save_path: str,
    *,
    seed: int,
    n_samples: int,
    title: str = "Latent t-SNE: expert labels vs substages",
    perplexity: float = 30.0,
) -> None:
    """Side-by-side t-SNE with one shared embedding (same subsample as PCA)."""
    n = datapoints.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(n_samples, n), replace=False)
    x = datapoints[idx].astype(np.float64, copy=False)
    y_true = labels_true[idx].astype(int, copy=False)
    y_pred = labels_pred[idx].astype(int, copy=False)
    perp = float(min(perplexity, max(5.0, (len(idx) - 1) / 3.0)))
    emb = TSNE(
        n_components=2,
        perplexity=perp,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(x)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, y, names, colors, subtitle in [
        (axes[0], y_true, label_names_true, label_colors_true, "Expert labels"),
        (axes[1], y_pred, label_names_pred, label_colors_pred, "GM substages"),
    ]:
        _scatter_tsne_panel(ax, emb, y, names, colors, title=subtitle)
    fig.suptitle(title, fontsize=12, y=1.02)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
