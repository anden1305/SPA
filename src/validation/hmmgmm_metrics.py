"""Trajectory and collapse metrics for HMM-GMM-VAE."""
from __future__ import annotations

import numpy as np
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


def state_entropy(predictions: np.ndarray, num_states: int) -> float:
    """Shannon entropy (nats) of the empirical state histogram."""
    counts = np.bincount(predictions.astype(int), minlength=num_states)
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * np.log(p)
    return float(entropy)


def entropy_norm(predictions: np.ndarray, k_pred: int) -> float:
    """H(Y_hat) / log(K_pred) in [0, 1]; 0 = single-state collapse."""
    if k_pred <= 1:
        return 0.0
    h_max = np.log(k_pred)
    if h_max <= 0:
        return 0.0
    return state_entropy(predictions, k_pred) / h_max


def checkpoint_score(
    log_likelihood: float,
    entropy_norm_val: float,
    beta: float = 0.9,
) -> float:
    """Thesis composite S(e) = beta * omega + (1 - beta) * H_norm."""
    return float(beta * log_likelihood + (1.0 - beta) * entropy_norm_val)


__all__ = [
    "state_switch_rate",
    "latent_autocorr",
    "state_entropy",
    "entropy_norm",
    "checkpoint_score",
]
