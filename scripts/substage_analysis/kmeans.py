from __future__ import annotations

import numpy as np
from typing import Tuple


def kmeans_predict_labels(
    datapoints: np.ndarray,
    c: int,
    seed: int = 0,
    max_iter: int = 300,
    tol: float = 1e-4,
    n_init: int = 10,
) -> np.ndarray:
    """
    Runs K-Means on datapoints (N,F) with c clusters and returns predicted labels y (N,).

    - Uses k-means++ initialization.
    - Runs n_init restarts and returns the best (lowest inertia) solution.
    - Pure NumPy (no sklearn dependency).
    """
    if not isinstance(datapoints, np.ndarray):
        raise TypeError("datapoints must be a numpy array.")
    if datapoints.ndim != 2:
        raise ValueError(f"datapoints must have shape (N,F). Got {datapoints.shape}.")
    N, F = datapoints.shape
    if c <= 0:
        raise ValueError("c must be > 0.")
    if c > N:
        raise ValueError(f"c={c} cannot exceed N={N}.")
    if max_iter <= 0:
        raise ValueError("max_iter must be > 0.")
    if n_init <= 0:
        raise ValueError("n_init must be > 0.")
    if tol < 0:
        raise ValueError("tol must be >= 0.")

    X = datapoints.astype(np.float64, copy=False)

    def pairwise_sq_dists_to_centroids(X_: np.ndarray, centroids_: np.ndarray) -> np.ndarray:
        # returns (N, c): ||x||^2 + ||mu||^2 - 2 x·mu
        x2 = np.sum(X_ * X_, axis=1, keepdims=True)              # (N,1)
        m2 = np.sum(centroids_ * centroids_, axis=1)[None, :]    # (1,c)
        return x2 + m2 - 2.0 * (X_ @ centroids_.T)

    def kmeanspp_init(rng: np.random.Generator) -> np.ndarray:
        centroids_ = np.empty((c, F), dtype=np.float64)

        # pick first centroid uniformly at random
        i0 = rng.integers(0, N)
        centroids_[0] = X[i0]

        # distances to nearest centroid so far
        d2 = np.sum((X - centroids_[0]) ** 2, axis=1)  # (N,)

        for k in range(1, c):
            # if all points are identical (or already perfectly covered)
            total = d2.sum()
            if not np.isfinite(total) or total <= 0:
                # fall back to random pick among points
                centroids_[k] = X[rng.integers(0, N)]
                d2 = np.minimum(d2, np.sum((X - centroids_[k]) ** 2, axis=1))
                continue

            probs = d2 / total
            idx = rng.choice(N, p=probs)
            centroids_[k] = X[idx]

            # update nearest distances
            d2 = np.minimum(d2, np.sum((X - centroids_[k]) ** 2, axis=1))

        return centroids_

    best_inertia = np.inf
    best_labels = None

    base_rng = np.random.default_rng(seed)

    for _ in range(n_init):
        # fresh RNG per init for determinism across inits
        init_seed = int(base_rng.integers(0, 2**32 - 1))
        rng = np.random.default_rng(init_seed)

        centroids = kmeanspp_init(rng)

        for _it in range(max_iter):
            dists = pairwise_sq_dists_to_centroids(X, centroids)  # (N,c)
            labels = np.argmin(dists, axis=1)

            # recompute centroids; handle empty clusters by re-seeding to farthest point
            new_centroids = centroids.copy()
            for k in range(c):
                mask = labels == k
                if np.any(mask):
                    new_centroids[k] = X[mask].mean(axis=0)
                else:
                    # pick point farthest from its assigned centroid (or overall farthest)
                    farthest_idx = np.argmax(np.min(dists, axis=1))
                    new_centroids[k] = X[farthest_idx]

            shift = np.sqrt(np.sum((new_centroids - centroids) ** 2, axis=1)).max()
            centroids = new_centroids
            if shift <= tol:
                break

        # inertia = sum of squared distances to assigned centroid
        final_dists = pairwise_sq_dists_to_centroids(X, centroids)
        inertia = float(np.sum(final_dists[np.arange(N), labels]))

        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()

    return best_labels