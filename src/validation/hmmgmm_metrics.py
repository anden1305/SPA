"""Trajectory metrics for HMM-GMM-VAE latent sequences."""
from __future__ import annotations

import torch


def state_switch_rate(labels: torch.Tensor, per: int = 100) -> float:
    """Mean number of state flips per ``per`` steps across a batch.

    Parameters
    ----------
    labels : (B, T) integer tensor.
    per : Scale for the rate (default: per 100 steps).

    Returns
    -------
    float
        ``per * E_b[ #{t : labels[b,t+1] != labels[b,t]} / (T - 1) ]``.
    """
    if labels.dim() != 2:
        raise ValueError(f"labels must be (B,T); got {tuple(labels.shape)}")
    B, T = labels.shape
    if T < 2:
        return 0.0
    flips = (labels[:, 1:] != labels[:, :-1]).float().sum(dim=1)
    rate_per_step = flips / float(T - 1)
    return float(rate_per_step.mean().item()) * float(per)


def latent_autocorr(mu: torch.Tensor, lag: int = 1) -> float:
    """Mean lag-``lag`` autocorrelation across batch and latent dims.

    Parameters
    ----------
    mu : (B, T, L) latent means.
    lag : positive integer.
    """
    if mu.dim() != 3:
        raise ValueError(f"mu must be (B,T,L); got {tuple(mu.shape)}")
    if lag < 1:
        raise ValueError(f"lag must be >= 1; got {lag}")
    _, T, _ = mu.shape
    if T <= lag:
        return 0.0

    x = mu - mu.mean(dim=1, keepdim=True)
    num = (x[:, lag:, :] * x[:, :-lag, :]).sum(dim=1)
    denom = (x * x).sum(dim=1).clamp_min(1e-12)
    rho = (num / denom).mean()
    return float(rho.item())


__all__ = ["state_switch_rate", "latent_autocorr"]
