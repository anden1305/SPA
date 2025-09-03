from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np
import torch

def _to_2d_features(x: Any) -> torch.Tensor:
    # Accept torch or numpy, flatten per-sample, handle complex by magnitude
    if isinstance(x, np.ndarray):
        if np.iscomplexobj(x):
            x = np.abs(x)
        x = torch.from_numpy(x)
    elif not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    if torch.is_complex(x):
        x = x.abs()
    x = x.float().contiguous()
    if x.ndim == 1:
        return x[:, None]
    # Keep the last dim as features; flatten preceding dims into samples
    # e.g. [E, W, F] -> [E*W, F]
    return x.reshape(-1, x.shape[-1])

def _to_labels(y: Any) -> torch.Tensor:
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y)
    elif not isinstance(y, torch.Tensor):
        y = torch.as_tensor(y)
    return y.view(-1).long()

def _align_samples_and_labels(X: torch.Tensor, Y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    n_x, n_y = X.shape[0], Y.shape[0]
    if n_x == n_y:
        return X, Y
    # Case 1: more samples than labels -> repeat labels
    if n_x % n_y == 0:
        r = n_x // n_y
        return X, Y.repeat_interleave(r)
    # Case 2: more labels than samples -> reduce labels by block mode
    if n_y % n_x == 0:
        r = n_y // n_x
        Yb = Y[: n_x * r].view(n_x, r)
        Y_mode, _ = torch.mode(Yb, dim=1)
        return X, Y_mode
    # Neither divides cleanly -> cannot align exactly
    raise ValueError(
        f"Cannot align samples and labels: X has {n_x} samples, y has {n_y}. "
        "Ensure counts are compatible or expose a mapping from transforms."
    )

def _mean_within_distance(A: torch.Tensor) -> float:
    m = A.shape[0]
    if m <= 1:
        return 0.0
    # Unbiased estimator of E||X - X'|| with i != j
    pw = torch.pdist(A, p=2)
    return float((2.0 / (m * (m - 1))) * pw.sum())

def _mean_cross_distance(A: torch.Tensor, B: torch.Tensor) -> float:
    if A.numel() == 0 or B.numel() == 0:
        return 0.0
    cd = torch.cdist(A, B, p=2)
    return float(cd.mean())

def _energy_distance(A: torch.Tensor, B: torch.Tensor) -> float:
    # ED = 2 E||A-B|| - E||A-A'|| - E||B-B'||
    return 2.0 * _mean_cross_distance(A, B) - _mean_within_distance(A) - _mean_within_distance(B)

def _fisher_trace(X: torch.Tensor, y: torch.Tensor, reg: float = 1e-4, max_dim: int = 4096) -> float | None:
    N, D = X.shape
    classes = torch.unique(y)
    if D > max_dim or classes.numel() < 2:
        return None
    device = X.device
    Xd = X.to(dtype=torch.float64, device=device)
    mu = Xd.mean(dim=0, keepdim=True)
    Sw = torch.zeros(D, D, dtype=torch.float64, device=device)
    Sb = torch.zeros(D, D, dtype=torch.float64, device=device)
    for c in classes:
        idx = (y == c)
        Xc = Xd[idx]
        if Xc.shape[0] == 0:
            continue
        mc = Xc.mean(dim=0, keepdim=True)
        Xcz = Xc - mc
        Sw += Xcz.t().mm(Xcz)
        diff = (mc - mu)
        Sb += Xc.shape[0] * (diff.t().mm(diff))
    Deye = torch.eye(D, dtype=torch.float64, device=device)
    Sw_reg = Sw + reg * Deye
    try:
        M = torch.linalg.solve(Sw_reg, Sb)
        return float(torch.trace(M).item())
    except Exception:
        return None

def compute_state_distinctness(x: Any, y: Any) -> Dict[str, Any]:
    X = _to_2d_features(x)
    Y = _to_labels(y).to(device=X.device)
    X, Y = _align_samples_and_labels(X, Y)

    classes = torch.unique(Y)
    classes_sorted = [int(c.item()) for c in classes]
    
    grouped: Dict[int, torch.Tensor] = {cid: X[Y == c] for cid, c in zip(classes_sorted, classes)}
    counts = {cid: int(grouped[cid].shape[0]) for cid in classes_sorted}

    k = len(classes_sorted)
    ed_mat = [[0.0 for _ in range(k)] for _ in range(k)]
    total_weight = 0.0
    weighted_sum = 0.0
    for i, ci in enumerate(classes_sorted):
        Ai = grouped[ci]
        for j in range(i + 1, k):
            cj = classes_sorted[j]
            Bj = grouped[cj]
            ed = _energy_distance(Ai, Bj)
            ed_mat[i][j] = ed
            ed_mat[j][i] = ed
            w = counts[ci] * counts[cj]
            weighted_sum += w * ed
            total_weight += w

    n_pairs = k * (k - 1) // 2
    mean_pairwise = (
        sum(ed_mat[i][j] for i in range(k) for j in range(i + 1, k)) / n_pairs
        if n_pairs > 0 else 0.0
    )
    weighted_mean = (weighted_sum / total_weight) if total_weight > 0 else mean_pairwise

    fisher_val = _fisher_trace(X, Y)

    print(f"State Distinctness - Mean Pairwise ED: {mean_pairwise}, \nWeighted Mean ED: {weighted_mean}, \nFisher Trace: {fisher_val}")
    
    return {
        "pairwise_energy": ed_mat,
        "mean_pairwise_energy": float(mean_pairwise),
        "weighted_mean_pairwise_energy": float(weighted_mean),
        "fisher_trace": None if fisher_val is None else float(fisher_val),
    }

