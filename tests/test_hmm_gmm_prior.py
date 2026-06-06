"""Unit tests for the HMM-GMM prior math (``src/models/hmm_gmm_prior.py``)."""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from src.models.hmm_gmm_prior import (
    estimate_transition_logits_from_paths,
    forward_log_marginal,
    gaussian_log_prob_diag,
    init_sticky_transition_logits,
    viterbi_decode,
)


# --------------------------------------------------------------------------- #
# Emissions                                                                    #
# --------------------------------------------------------------------------- #

def test_gaussian_log_prob_diag_shapes_and_values():
    B, T, L, K = 3, 5, 4, 6
    z = torch.randn(B, T, L)
    means = torch.randn(K, L)
    logvars = torch.randn(K, L) * 0.1

    log_emit = gaussian_log_prob_diag(z, means, logvars)
    assert log_emit.shape == (B, T, K)
    assert torch.isfinite(log_emit).all()

    # Per-state, the log-density should be the sum over L dims of a
    # univariate normal log-pdf.
    var = logvars[0].exp()
    ref = -0.5 * (
        math.log(2.0 * math.pi)
        + logvars[0]
        + (z[0, 0] - means[0]).pow(2) / var
    ).sum()
    assert torch.allclose(log_emit[0, 0, 0], ref, atol=1e-6)


# --------------------------------------------------------------------------- #
# Forward algorithm                                                            #
# --------------------------------------------------------------------------- #

def _brute_force_log_marginal(
    log_emit: torch.Tensor,
    log_pi: torch.Tensor,
    log_A: torch.Tensor,
) -> torch.Tensor:
    """Exhaustive enumeration of state paths (O(K^T) -- tiny T only)."""
    B, T, K = log_emit.shape
    out = torch.full((B,), float("-inf"))
    for b in range(B):
        for path in torch.cartesian_prod(*[torch.arange(K)] * T):
            if T == 1:
                path = path.view(1)
            score = log_pi[path[0]] + log_emit[b, 0, path[0]]
            for t in range(1, T):
                score = score + log_A[path[t - 1], path[t]] + log_emit[b, t, path[t]]
            out[b] = torch.logaddexp(out[b], score)
    return out


def test_forward_log_marginal_matches_brute_force():
    B, T, K = 2, 4, 3
    log_emit = torch.randn(B, T, K) * 0.5
    log_pi = F.log_softmax(torch.randn(K), dim=0)
    log_A = F.log_softmax(torch.randn(K, K), dim=-1)

    log_p = forward_log_marginal(log_emit, log_pi, log_A)
    ref = _brute_force_log_marginal(log_emit, log_pi, log_A)
    assert log_p.shape == (B,)
    assert torch.allclose(log_p, ref, atol=1e-5)


def test_forward_log_marginal_t1_equals_logsumexp():
    B, K = 4, 5
    log_emit = torch.randn(B, 1, K)
    log_pi = F.log_softmax(torch.randn(K), dim=0)
    log_A = F.log_softmax(torch.randn(K, K), dim=-1)

    log_p = forward_log_marginal(log_emit, log_pi, log_A)
    ref = torch.logsumexp(log_pi.unsqueeze(0) + log_emit[:, 0, :], dim=1)
    assert torch.allclose(log_p, ref, atol=1e-6)


def test_forward_log_marginal_gradients_flow():
    B, T, K = 2, 6, 3
    z = torch.randn(B, T, K, requires_grad=True)
    transition_logits = torch.randn(K, K, requires_grad=True)
    initial_logits = torch.randn(K, requires_grad=True)
    means = torch.randn(K, K, requires_grad=True)
    logvars = torch.zeros(K, K, requires_grad=True)

    log_emit = gaussian_log_prob_diag(z, means, logvars)
    log_pi = F.log_softmax(initial_logits, dim=0)
    log_A = F.log_softmax(transition_logits, dim=-1)
    log_p = forward_log_marginal(log_emit, log_pi, log_A).sum()
    log_p.backward()

    for name, t in {
        "z": z,
        "transition_logits": transition_logits,
        "initial_logits": initial_logits,
        "means": means,
        "logvars": logvars,
    }.items():
        assert t.grad is not None and torch.isfinite(t.grad).all(), name


# --------------------------------------------------------------------------- #
# Sticky init                                                                  #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("K,kappa", [(3, 0.9), (5, 0.5), (2, 0.8)])
def test_init_sticky_transition_logits_diagonal(K: int, kappa: float):
    logits = init_sticky_transition_logits(K, kappa)
    assert logits.shape == (K, K)
    A = logits.exp()
    diag = torch.diagonal(A)
    assert torch.allclose(diag, torch.full((K,), kappa), atol=1e-6)
    row_sums = A.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(K), atol=1e-6)


def test_init_sticky_transition_logits_rejects_bad_args():
    with pytest.raises(ValueError):
        init_sticky_transition_logits(3, 0.0)
    with pytest.raises(ValueError):
        init_sticky_transition_logits(3, 1.0)
    with pytest.raises(ValueError):
        init_sticky_transition_logits(1, 0.5)


# --------------------------------------------------------------------------- #
# Viterbi                                                                      #
# --------------------------------------------------------------------------- #

def test_viterbi_constant_emissions_returns_argmax_state():
    B, T, K = 2, 5, 4
    log_emit = torch.zeros(B, T, K)
    log_emit[:, :, 2] = 5.0  # state 2 dominates everywhere
    log_pi = torch.full((K,), -math.log(K))
    log_A = torch.full((K, K), -math.log(K))

    path = viterbi_decode(log_emit, log_pi, log_A)
    assert path.shape == (B, T)
    assert (path == 2).all()


def test_viterbi_follows_sticky_transitions():
    # Start prefers state 0; emissions are flat; transitions are sticky.
    # Expect: stay in state 0 for the whole sequence.
    B, T, K = 1, 6, 3
    log_emit = torch.zeros(B, T, K)
    log_pi = torch.tensor([10.0, 0.0, 0.0]).log_softmax(0)
    log_A = init_sticky_transition_logits(K, 0.95)

    path = viterbi_decode(log_emit, log_pi, log_A)
    assert (path == 0).all()


# --------------------------------------------------------------------------- #
# Path-based transition estimation                                             #
# --------------------------------------------------------------------------- #

def test_estimate_transition_logits_recovers_known_paths():
    # 4 sequences alternating 0 -> 1 -> 0 -> 1 ...
    K = 2
    assign = torch.tensor([
        [0, 1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0, 1],
    ])
    log_pi, log_A = estimate_transition_logits_from_paths(assign, K)
    A = log_A.exp()

    # A[0, 1] should be ~1, A[1, 0] should be ~1.
    assert A[0, 1] > 0.99
    assert A[1, 0] > 0.99
    # All sequences start in state 0
    assert log_pi.exp()[0] > 0.99


def test_estimate_transition_logits_t1_falls_back_to_uniform():
    K = 3
    assign = torch.zeros(5, 1, dtype=torch.long)
    log_pi, log_A = estimate_transition_logits_from_paths(assign, K)
    # log_pi can be peaked on state 0; transitions are uniform fallback.
    A = log_A.exp()
    assert torch.allclose(A, torch.full((K, K), 1.0 / K), atol=1e-6)
