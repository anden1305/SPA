"""Thesis-style substage figure stack from a single results.npz (holdout validation export)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from scripts.substage_analysis.distribution_plot import plot_label_distribution
from scripts.substage_analysis.fig27_from_npz import plot_fig27_gmm_predicted
from scripts.substage_analysis.kmeans import kmeans_predict_labels
from scripts.substage_analysis.labels import PREDICTED_SUBSTAGE_LABEL
from scripts.substage_analysis.plot_substages import pca_scatter_random_samples, tsne_scatter_pair
from scripts.substage_analysis.transition_matrix import plot_transition_matrix

LABEL_COLORS_TRUE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#e41a1c"]
LABEL_COLORS_PRED = [
    "#6A3D9A", "#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E",
    "#E6AB02", "#A6761D", "#666666", "#8DD3C7", "#BEBADA", "#FB8072",
    "#80B1D3", "#FDB462", "#B3B3B3", "#17BECF", "#BCBD22",
]
SEED = 124
N_SAMPLES = 7500
TSNE_N_SAMPLES = 25_000


def remap_labels(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y.astype(int))
    ranked_old = np.argsort(-counts)
    ranked_old = ranked_old[counts[ranked_old] > 0]
    mapping = np.empty(counts.shape[0], dtype=int)
    mapping[ranked_old] = np.arange(len(ranked_old))
    return mapping[y.astype(int)]


def _fit_pca_scores(datapoints: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = datapoints.astype(np.float64)
    mu = x.mean(axis=0)
    xc = x - mu
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    v = vt.T
    n = x.shape[0]
    if n > 1:
        evr = (s ** 2) / (n - 1)
        evr = evr / evr.sum()
    else:
        evr = np.zeros_like(s)
    scores = xc @ v
    return scores, evr, mu


def plot_pca_true_vs_predicted(
    datapoints: np.ndarray,
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
    label_names_true: Sequence[str],
    label_names_pred: Sequence[str],
    save_path: str,
    *,
    seed: int = SEED,
    n_samples: int = N_SAMPLES,
    title: str = "Latent PCA: expert labels vs substages",
) -> None:
    """Thesis Fig 25/26 style side-by-side PCA."""
    n = datapoints.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(n_samples, n), replace=False)
    scores, evr, mu = _fit_pca_scores(datapoints)
    pc1 = scores[idx, 0]
    pc2 = scores[idx, 1]
    y_true = labels_true[idx]
    y_pred = labels_pred[idx]

    from src.visuals.layered_scatter import scatter_layered

    title_kw = {"fontweight": "normal", "family": "serif"}
    axis_kw = {"fontweight": "normal", "family": "serif", "fontsize": 10}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, y, names, colors, subtitle in [
        (axes[0], y_true, label_names_true, LABEL_COLORS_TRUE[: len(label_names_true)], "Expert labels"),
        (axes[1], y_pred, label_names_pred, LABEL_COLORS_PRED[: len(label_names_pred)], PREDICTED_SUBSTAGE_LABEL),
    ]:
        colors_lut = {c: colors[c] for c in range(len(names))}
        scatter_layered(
            ax,
            pc1,
            pc2,
            y,
            label_names=names if "Artifact" in names else None,
            colors=colors_lut,
            point_size=14,
        )
        ax.set_xlabel(f"PC1 ({evr[0] * 100:.1f}% var)", **axis_kw)
        ax.set_ylabel(f"PC2 ({evr[1] * 100:.1f}% var)", **axis_kw)
        ax.set_title(subtitle, fontsize=11, **title_kw)
        ax.legend(
            handles=[
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[c], label=names[c], markersize=8)
                for c in range(len(names))
                if np.any(y == c)
            ],
            loc="best",
            fontsize=8,
            frameon=True,
            prop={"weight": "normal", "family": "serif", "size": 8},
        )
        ax.grid(True, linewidth=0.4, alpha=0.35)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("normal")
            label.set_family("serif")
    fig.suptitle(title, fontsize=11, y=1.02, fontweight="normal", family="serif")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_latent_feature_boxplots(
    datapoints: np.ndarray,
    labels_pred: np.ndarray,
    label_names_pred: Sequence[str],
    save_path: str,
    *,
    title: str = "Latent dimensions by substage",
) -> None:
    """Boxplots of each latent dim per predicted substage."""
    n_states = len(label_names_pred)
    n_feat = datapoints.shape[1]
    fig, axes = plt.subplots(1, n_feat, figsize=(2.2 * n_feat, 4.0), squeeze=False)
    for f in range(n_feat):
        ax = axes[0, f]
        data = [datapoints[labels_pred == c, f] for c in range(n_states)]
        ax.boxplot(data, tick_labels=[n.split()[-1] for n in label_names_pred], showfliers=False)
        ax.set_title(f"z{f + 1}", fontsize=9)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(title, fontsize=11)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_full_thesis_analysis(
    npz_path: Path,
    out_dir: Path,
    *,
    k: int | None = None,
    nmi: float | None = None,
) -> dict[str, str]:
    """
    Generate thesis-style substage plots (better organized than legacy substages_analysis/).

    Naming mirrors the old stack: pca_*, transition_*, frequency_grid, label_distribution,
    plus pca_comparison_true_vs_predicted and latent_feature_boxplots.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(npz_path)
    datapoints = data["x_latent"].reshape(-1, data["x_latent"].shape[-1])
    labels_true = data["y_true"].reshape(-1)
    labels_pred = remap_labels(data["y_hat"].reshape(-1))
    raw = data["x"].reshape(-1, data["x"].shape[2], data["x"].shape[3])
    sub_ids = data["sub_ids"].reshape(-1)

    n_true = int(labels_true.max()) + 1
    n_pred = int(labels_pred.max()) + 1
    label_names_true = ["Awake", "NREM", "REM", "Artifact"][:n_true]
    label_names_pred = [f"Substage {i + 1}" for i in range(n_pred)]
    n_samples = min(N_SAMPLES, datapoints.shape[0])

    prefix = f"K{k}_" if k is not None else ""
    title_k = f"K={k}" if k is not None else ""
    nmi_s = f", NMI={nmi:.3f}" if nmi is not None else ""

    outputs: dict[str, str] = {}

    def _save(name: str, path: Path) -> Path:
        outputs[name] = str(path)
        return path

    plot_pca_true_vs_predicted(
        datapoints,
        labels_true,
        labels_pred,
        label_names_true,
        label_names_pred,
        str(_save("pca_comparison", out_dir / f"{prefix}pca_comparison_true_vs_predicted.png")),
        n_samples=n_samples,
        title=f"{title_k} latent PCA{nmi_s}",
    )

    for tag, labels, names, colors in [
        ("pca_true", labels_true, label_names_true, LABEL_COLORS_TRUE[:n_true]),
        ("pca_predicted", labels_pred, label_names_pred, LABEL_COLORS_PRED[:n_pred]),
    ]:
        pca_scatter_random_samples(
            datapoints=datapoints,
            labels=labels,
            label_names=names,
            analysis_name=f"{title_k} PCA ({tag.split('_', 1)[1]})",
            seed=SEED,
            n_samples=n_samples,
            save_path=str(_save(tag, out_dir / f"{prefix}pca_scatter_{tag.split('_', 1)[1]}.png")),
            label_colors=colors,
        )

    labels_kmeans = kmeans_predict_labels(datapoints=datapoints, c=n_pred, seed=SEED)
    pca_scatter_random_samples(
        datapoints=datapoints,
        labels=labels_kmeans,
        label_names=label_names_pred,
        analysis_name=f"{title_k} PCA (kmeans)",
        seed=SEED,
        n_samples=n_samples,
        save_path=str(_save("pca_kmeans", out_dir / f"{prefix}pca_scatter_kmeans.png")),
        label_colors=LABEL_COLORS_PRED[:n_pred],
    )

    for tag, y, names in [
        ("transition_true", labels_true, label_names_true),
        ("transition_predicted", labels_pred, label_names_pred),
        ("transition_kmeans", labels_kmeans, label_names_pred),
    ]:
        stem = tag.replace("transition_", "")
        plot_transition_matrix(
            y=y,
            label_names=names,
            plot_name=f"{title_k} transitions ({stem})",
            save_path=str(_save(tag, out_dir / f"{prefix}transition_matrix_{stem}.png")),
            csv_path=str(out_dir / f"{prefix}transition_matrix_{stem}.csv")
            if tag == "transition_predicted"
            else None,
        )

    fig27_k = k if k is not None else n_pred
    plot_fig27_gmm_predicted(
        npz_path,
        out_dir / f"{prefix}frequency_plot_gmm_predicted.png",
        k=fig27_k,
        nmi=nmi,
    )
    outputs["frequency_plot_gmm_predicted"] = str(out_dir / f"{prefix}frequency_plot_gmm_predicted.png")

    plot_label_distribution(
        y_pred=labels_pred,
        label_names=label_names_pred,
        plot_title=f"{title_k} substage occupancy{nmi_s}",
        save_path=str(_save("label_distribution", out_dir / f"{prefix}label_distribution.png")),
    )

    plot_latent_feature_boxplots(
        datapoints,
        labels_pred,
        label_names_pred,
        str(_save("latent_boxplots", out_dir / f"{prefix}latent_feature_boxplots.png")),
        title=f"{title_k} latent dimensions by substage{nmi_s}",
    )

    tsne_n = min(TSNE_N_SAMPLES, datapoints.shape[0])
    tsne_scatter_pair(
        datapoints=datapoints,
        labels_true=labels_true,
        labels_pred=labels_pred,
        label_names_true=label_names_true,
        label_names_pred=label_names_pred,
        label_colors_true=LABEL_COLORS_TRUE[:n_true],
        label_colors_pred=LABEL_COLORS_PRED[:n_pred],
        analysis_name=f"{title_k} latent t-SNE{nmi_s} (n={tsne_n:,})",
        save_path_true=str(_save("tsne_true", out_dir / f"{prefix}tsne_scatter_true.png")),
        save_path_pred=str(_save("tsne_predicted", out_dir / f"{prefix}tsne_scatter_predicted.png")),
        seed=SEED,
        n_samples=tsne_n,
    )

    return outputs


def run_tsne_from_npz(
    npz_path: Path,
    out_dir: Path,
    *,
    k: int | None = None,
    nmi: float | None = None,
    n_samples: int | None = None,
) -> dict[str, str]:
    """t-SNE only: expert vs GM substage scatters (shared embedding)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(npz_path)
    datapoints = data["x_latent"].reshape(-1, data["x_latent"].shape[-1])
    labels_true = data["y_true"].reshape(-1)
    labels_pred = remap_labels(data["y_hat"].reshape(-1))

    n_true = int(labels_true.max()) + 1
    n_pred = int(labels_pred.max()) + 1
    label_names_true = ["Awake", "NREM", "REM", "Artifact"][:n_true]
    label_names_pred = [f"Substage {i + 1}" for i in range(n_pred)]
    tsne_n = min(n_samples if n_samples is not None else TSNE_N_SAMPLES, datapoints.shape[0])

    prefix = f"K{k}_" if k is not None else ""
    title_k = f"K={k}" if k is not None else ""
    nmi_s = f", NMI={nmi:.3f}" if nmi is not None else ""

    outputs: dict[str, str] = {}
    true_path = out_dir / f"{prefix}tsne_scatter_true.png"
    pred_path = out_dir / f"{prefix}tsne_scatter_predicted.png"
    tsne_scatter_pair(
        datapoints=datapoints,
        labels_true=labels_true,
        labels_pred=labels_pred,
        label_names_true=label_names_true,
        label_names_pred=label_names_pred,
        label_colors_true=LABEL_COLORS_TRUE[:n_true],
        label_colors_pred=LABEL_COLORS_PRED[:n_pred],
        analysis_name=f"{title_k} latent t-SNE{nmi_s} (n={tsne_n:,})",
        save_path_true=str(true_path),
        save_path_pred=str(pred_path),
        seed=SEED,
        n_samples=tsne_n,
    )
    outputs["tsne_scatter_true"] = str(true_path)
    outputs["tsne_scatter_predicted"] = str(pred_path)
    return outputs
