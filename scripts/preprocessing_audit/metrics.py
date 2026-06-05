"""Separability metrics on VAE feature tensors."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def flatten_features_and_labels(
    x: np.ndarray,
    y: np.ndarray,
    max_samples: int = 8000,
    seed: int = 123,
) -> tuple[np.ndarray, np.ndarray]:
    n_seq, S, C, F = x.shape
    seq_labels = []
    for i in range(n_seq):
        row = y[i].ravel()
        vals, counts = np.unique(row, return_counts=True)
        seq_labels.append(int(vals[counts.argmax()]))
    labels = np.array(seq_labels, dtype=np.int64)
    feats = x.reshape(n_seq, S * C * F)
    if len(feats) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(feats), size=max_samples, replace=False)
        return feats[idx], labels[idx]
    return feats, labels


def compute_metrics(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    feats, labels = flatten_features_and_labels(x, y)
    unique = np.unique(labels)
    out: dict[str, float] = {
        "n_sequences": float(x.shape[0]),
        "n_features": float(feats.shape[1]),
    }
    if len(unique) < 2 or len(feats) < 10:
        out.update(
            silhouette_k3=np.nan,
            pc12_var_explained=np.nan,
            mean_mahalanobis_between_stages=np.nan,
            emg_rem_wake_auc_proxy=np.nan,
        )
        return out

    try:
        sil = silhouette_score(feats, labels, metric="euclidean")
    except Exception:
        sil = np.nan
    out["silhouette_k3"] = float(sil) if len(unique) >= 2 else np.nan

    pca = PCA(n_components=min(2, feats.shape[1], len(feats) - 1))
    pca.fit(feats)
    out["pc12_var_explained"] = float(pca.explained_variance_ratio_.sum())

    # Mahalanobis between stage centroids
    dists = []
    for i, a in enumerate(unique):
        for b in unique[i + 1 :]:
            ma = feats[labels == a]
            mb = feats[labels == b]
            if len(ma) < 2 or len(mb) < 2:
                continue
            ca, cb = ma.mean(0), mb.mean(0)
            dists.append(float(np.linalg.norm(ca - cb)))
    out["mean_mahalanobis_between_stages"] = float(np.mean(dists)) if dists else np.nan

    # EMG channel proxy: last channel mean feature vs wake(0) vs rem(2)
    if x.ndim != 4:
        out["emg_rem_wake_auc_proxy"] = np.nan
        return out
    C = x.shape[2]
    emg_idx = C - 1
    emg_power = x[:, :, emg_idx, :].mean(axis=(1, 2))
    wake_mask = labels == 0
    rem_mask = labels == 2
    if wake_mask.sum() > 0 and rem_mask.sum() > 0:
        wake_m = emg_power[wake_mask].mean()
        rem_m = emg_power[rem_mask].mean()
        out["emg_rem_wake_auc_proxy"] = float(rem_m > wake_m)
    else:
        out["emg_rem_wake_auc_proxy"] = np.nan

    return out


def counterfactual_post_normalize(
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    return ((x - mean) / (std + 1e-8)).astype(np.float32)


def post_norm_stats_run_level(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=(0, 1), keepdims=True)
    std = np.std(x, axis=(0, 1), keepdims=True) + 1e-8
    return mean, std
