## Normalized Mutual Information (NMI)
## input predicted tensors and ground truth tensors (ints)

from torch import Tensor
import torch
import numpy as np
from typing import Any


def _to_tensor(x: Any) -> Tensor:
    """Convert numpy arrays, lists/tuples, or torch tensors to a torch.Tensor.

    Returns a CPU tensor for non-torch inputs.
    """
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    if isinstance(x, (list, tuple)):
        return torch.as_tensor(x)
    # Fallback: try to construct a tensor (may raise)
    return torch.as_tensor(x)


def calculate_nmi(pred: Any, target: Any) -> float:
    """Compute Normalized Mutual Information (NMI) between integer label tensors.

    Args:
        pred: Tensor of predicted labels (ints), any shape.
        target: Tensor of target/true labels (ints), same shape as pred.

    Returns:
        float: NMI in [0, 1], using the arithmetic normalization 2*I(X;Y)/(H(X)+H(Y)).
    """
    # Accept numpy arrays, python lists/tuples or torch tensors
    pred = _to_tensor(pred)
    target = _to_tensor(target)

    if pred.numel() == 0:
        return 0.0

    # Flatten and ensure integer type
    pred = pred.view(-1).to(dtype=torch.long)
    target = target.view(-1).to(dtype=torch.long)

    if pred.shape != target.shape:
        raise ValueError("pred and target must have the same number of elements")

    device = pred.device
    eps = 1e-12
    n = pred.numel()

    # Map labels to contiguous indices for stability
    _, inv_pred = torch.unique(pred, return_inverse=True)
    _, inv_tgt = torch.unique(target, return_inverse=True)

    kx = int(inv_pred.max().item() + 1)
    ky = int(inv_tgt.max().item() + 1)

    # Joint counts via linearized indices
    lin_idx = inv_pred * ky + inv_tgt
    joint_counts = torch.bincount(lin_idx, minlength=kx * ky).to(dtype=torch.float64, device=device)
    joint = joint_counts.view(kx, ky)

    # Convert to probabilities
    joint_prob = joint / float(n)
    px = joint_prob.sum(dim=1)  # (kx,)
    py = joint_prob.sum(dim=0)  # (ky,)

    # Entropies
    Hx = -(px * (px + eps).log()).sum()
    Hy = -(py * (py + eps).log()).sum()
    Hxy = -(joint_prob * (joint_prob + eps).log()).sum()

    # Mutual information and normalization (arithmetic average)
    Ixy = Hx + Hy - Hxy
    denom = (Hx + Hy).clamp_min(eps)

    # Degenerate case: both partitions have (near) zero entropy (single cluster)
    if Hx.item() < 1e-10 and Hy.item() < 1e-10:
        return 1.0

    nmi = (2.0 * Ixy / denom).clamp(min=0.0, max=1.0)
    return round(float(nmi.item()), 4)