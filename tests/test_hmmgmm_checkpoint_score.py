"""Unit tests for thesis checkpoint score helpers."""
from __future__ import annotations

import numpy as np

from src.validation.hmmgmm_metrics import checkpoint_score, entropy_norm, state_entropy


def test_entropy_norm_collapsed_single_state():
    preds = np.zeros(1000, dtype=int)
    assert state_entropy(preds, 3) == 0.0
    assert entropy_norm(preds, 3) == 0.0


def test_entropy_norm_uniform_three_states():
    preds = np.tile(np.arange(3), 100)
    h = state_entropy(preds, 3)
    assert np.isclose(h, np.log(3), rtol=1e-5)
    assert np.isclose(entropy_norm(preds, 3), 1.0, rtol=1e-5)


def test_checkpoint_score_weights():
    ll = -100.0
    h_norm = 0.5
    s = checkpoint_score(ll, h_norm, beta=0.9)
    assert np.isclose(s, 0.9 * ll + 0.1 * h_norm)


def test_entropy_norm_k_pred_one():
    preds = np.array([0, 0, 0])
    assert entropy_norm(preds, 1) == 0.0
