"""HMM-GMM latent prior for sequence VAEs.

Pure-tensor utilities for using a hidden Markov model with diagonal-Gaussian
emissions as a prior over latent sequences ``z_{1:T}``. Reused by
``ConditionalVAE`` (``prior='hmm_gmm'`` / ``'warm_hmm_gmm'``) and by tests.

Conventions
-----------
- ``z``           shape ``(B, T, L)``  -- continuous latent samples.
- ``log_emit``    shape ``(B, T, K)``  -- per-state diagonal-Gaussian log-density of ``z_t``.
- ``log_pi``      shape ``(K,)``       -- log of initial state probabilities.
- ``log_A``       shape ``(K, K)``     -- log of row-stochastic transition matrix.
"""
from __future__ import annotations

import math

import torch


_LOG_2PI = math.log(2.0 * math.pi)


def gaussian_log_prob_diag(
    z: torch.Tensor,
    means: torch.Tensor,
    logvars: torch.Tensor,
) -> torch.Tensor:
    """Diagonal-Gaussian per-state log-density.

    Parameters
    ----------
    z : (B, T, L)
        Latent samples.
    means : (K, L)
    logvars : (K, L)

    Returns
    -------
    log_emit : (B, T, K)
    """
    if z.dim() != 3:
        raise ValueError(f"z must be (B,T,L); got {tuple(z.shape)}")
    if means.dim() != 2 or logvars.shape != means.shape:
        raise ValueError(
            f"means/logvars must be (K,L) with matching shapes; got "
            f"{tuple(means.shape)} / {tuple(logvars.shape)}"
        )

    z_ex = z.unsqueeze(2)             # (B,T,1,L)
    means_ex = means.view(1, 1, *means.shape)      # (1,1,K,L)
    logvars_ex = logvars.view(1, 1, *logvars.shape)  # (1,1,K,L)

    log_emit = -0.5 * (
        _LOG_2PI
        + logvars_ex
        + (z_ex - means_ex).pow(2) / logvars_ex.exp()
    ).sum(dim=-1)                     # (B,T,K)
    return log_emit


def forward_log_marginal(
    log_emit: torch.Tensor,
    log_pi: torch.Tensor,
    log_A: torch.Tensor,
) -> torch.Tensor:
    """Forward algorithm for marginal sequence log-likelihood.

    Computes ``log p(z_{1:T}) = log sum_{s_{1:T}} pi_{s_1} prod_t A_{s_{t-1},s_t} p(z_t | s_t)``.

    Parameters
    ----------
    log_emit : (B, T, K)
    log_pi : (K,)
    log_A : (K, K)

    Returns
    -------
    log_p_seq : (B,)
    """
    if log_emit.dim() != 3:
        raise ValueError(f"log_emit must be (B,T,K); got {tuple(log_emit.shape)}")
    K = log_emit.shape[-1]
    if log_pi.shape != (K,):
        raise ValueError(f"log_pi must be (K,); got {tuple(log_pi.shape)}")
    if log_A.shape != (K, K):
        raise ValueError(f"log_A must be (K,K); got {tuple(log_A.shape)}")

    _, T, _ = log_emit.shape
    eps = torch.finfo(log_emit.dtype).tiny

    A_prob = torch.exp(log_A)                              # (K,K)
    alpha = log_pi.unsqueeze(0) + log_emit[:, 0, :]        # (B,K)

    for t in range(1, T):
        m = alpha.max(dim=1, keepdim=True).values          # (B,1)
        v = torch.exp(alpha - m)                           # (B,K)
        u = v @ A_prob                                     # (B,K)
        alpha = log_emit[:, t, :] + m + torch.log(u.clamp_min(eps))

    return torch.logsumexp(alpha, dim=1)                   # (B,)


@torch.no_grad()
def viterbi_decode(
    log_emit: torch.Tensor,
    log_pi: torch.Tensor,
    log_A: torch.Tensor,
) -> torch.Tensor:
    """Most-likely state path per sequence (Viterbi).

    Parameters
    ----------
    log_emit : (B, T, K)
    log_pi : (K,)
    log_A : (K, K)

    Returns
    -------
    path : (B, T) int64
    """
    if log_emit.dim() != 3:
        raise ValueError(f"log_emit must be (B,T,K); got {tuple(log_emit.shape)}")
    B, T, K = log_emit.shape
    if log_pi.shape != (K,):
        raise ValueError(f"log_pi must be (K,); got {tuple(log_pi.shape)}")
    if log_A.shape != (K, K):
        raise ValueError(f"log_A must be (K,K); got {tuple(log_A.shape)}")

    device = log_emit.device
    backptr = log_emit.new_zeros((B, T, K), dtype=torch.long)
    log_A_T = log_A.transpose(0, 1).contiguous()           # (K,K)

    delta = log_pi.unsqueeze(0) + log_emit[:, 0, :]        # (B,K)
    for t in range(1, T):
        scores = delta.unsqueeze(1) + log_A_T              # (B,K,K)
        delta, idx = torch.max(scores, dim=2)              # (B,K)
        delta = delta + log_emit[:, t, :]
        backptr[:, t, :] = idx

    last = torch.argmax(delta, dim=1)                      # (B,)
    path = log_emit.new_zeros((B, T), dtype=torch.long)
    path[:, -1] = last
    arange_B = torch.arange(B, device=device)
    for t in range(T - 2, -1, -1):
        path[:, t] = backptr[arange_B, t + 1, path[:, t + 1]]
    return path


def init_sticky_transition_logits(
    K: int,
    kappa: float,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Logits whose softmax has ``kappa`` on the diagonal, ``(1-kappa)/(K-1)`` off-diagonal.

    Returns a ``(K, K)`` tensor suitable for copying into ``transition_logits``.
    """
    if not 0.0 < kappa < 1.0:
        raise ValueError(f"kappa must be in (0,1); got {kappa}")
    if K < 2:
        raise ValueError(f"K must be >= 2; got {K}")

    off = (1.0 - kappa) / (K - 1)
    A = torch.full((K, K), off, device=device, dtype=dtype)
    A.fill_diagonal_(kappa)
    return A.clamp_min(1e-12).log()


def estimate_transition_logits_from_paths(
    assign: torch.Tensor,
    K: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate (initial_logits, transition_logits) from observed state paths.

    Mirrors ``set_transition_params`` in ``src/initializations/kmeans.py`` but
    returns the parameters instead of writing them, and works without a model.

    Parameters
    ----------
    assign : (B, T) integer tensor of state indices in ``[0, K)``.
    K : number of states.

    Returns
    -------
    log_pi : (K,)
    log_A : (K, K)
    """
    if assign.dim() != 2:
        raise ValueError(f"assign must be (B,T); got {tuple(assign.shape)}")
    if K < 1:
        raise ValueError(f"K must be >= 1; got {K}")

    device = assign.device
    z = assign.long()
    B, T = z.shape

    pi_counts = torch.bincount(z[:, 0], minlength=K).float() + 1e-3
    pi = pi_counts / pi_counts.sum()

    if T >= 2:
        prev = z[:, :-1].reshape(-1)
        nxt = z[:, 1:].reshape(-1)
        joint_idx = prev * K + nxt
        trans_counts = (
            torch.bincount(joint_idx, minlength=K * K).float().reshape(K, K) + 1e-3
        )
        A = trans_counts / trans_counts.sum(-1, keepdim=True)
    else:
        A = torch.full((K, K), 1.0 / K, device=device)

    log_pi = pi.clamp_min(1e-12).log().to(device)
    log_A = A.clamp_min(1e-12).log().to(device)
    return log_pi, log_A


__all__ = [
    "gaussian_log_prob_diag",
    "forward_log_marginal",
    "viterbi_decode",
    "init_sticky_transition_logits",
    "estimate_transition_logits_from_paths",
]
