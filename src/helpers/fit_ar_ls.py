from __future__ import annotations

from typing import Sequence, Tuple
import torch
from torch import Tensor

__all__ = ["fit_ar_ls"]

def fit_ar_ls(x: Tensor, lags: Sequence[int], ridge: float = 0.0) -> Tuple[Tensor, Tensor]:
    """Closed-form ridge least-squares AR fit for a single 1D sequence.

    Parameters
    ----------
    x : (T,) tensor
    lags : list/sequence of positive integers
    ridge : λ (adds λI to normal equations)
    Returns
    -------
    coeffs : (L,)
    var : () noise variance estimate
    """
    if x.dim() != 1:
        raise ValueError("x must be 1D tensor")
    Ls = list(lags)
    if len(Ls) == 0:
        raise ValueError("lags must be non-empty")
    if any(l <= 0 for l in Ls):
        raise ValueError("All lags must be positive")
    mL = max(Ls)
    if x.numel() <= mL:
        raise ValueError("Sequence too short for largest lag")
    rows = []
    ys = []
    for t in range(mL, x.numel()):
        rows.append([x[t - l].item() for l in Ls])
        ys.append(x[t].item())
    X = torch.tensor(rows, dtype=x.dtype, device=x.device)  # (N,L)
    y = torch.tensor(ys, dtype=x.dtype, device=x.device)    # (N,)
    XtX = X.T @ X
    if ridge > 0:
        XtX = XtX + ridge * torch.eye(XtX.size(0), device=X.device, dtype=X.dtype)
    Xty = X.T @ y
    coeffs = torch.linalg.solve(XtX, Xty)  # (L,)
    resid = y - X @ coeffs
    var = resid.pow(2).mean().clamp_min(1e-12)
    return coeffs, var
