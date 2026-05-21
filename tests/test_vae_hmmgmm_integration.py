"""Integration tests for the HMM-GMM prior wired into ``ConditionalVAE``."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from src.models.vae import ConditionalVAE


def _make_vae(prior: str, *, T: int = 6, K: int = 3, L: int = 4) -> ConditionalVAE:
    """Build a minimal ``ConditionalVAE`` without touching real datasets."""
    dl = MagicMock()
    dl.get_num_subjects.return_value = 2
    dl.get_num_labs.return_value = 2
    dl.get_vae_dims.return_value = (1, 8, T, K)  # (C, F, S=T, num_states)

    cfg = SimpleNamespace(
        seed=0,
        num_states=K,
        model=SimpleNamespace(
            type='cvae_marhmm',
            conditioning_source='subject',
            params=dict(
                latent_dim=L,
                enc_hidden_dims=[16, 8],
                dec_hidden_dims=[8, 16],
                conv_channels=[4, 4],
                kernel_sizes=[3, 3],
                strides=[1, 1],
                paddings=[1, 1],
                emb_dim=0,
                decoder_only_conditioning=True,
                use_rms=False,
                prior=prior,
                num_gmm_states=K,
                gmm_warmup_epochs=1,
                hmm_warmup_epochs=2,
                hmm_transition_ramp_epochs=0,
                hmm_sticky_kappa=0.9,
                hmm_estimate_transitions=True,
                min_beta=0.01,
                max_beta=0.5,
                beta_warmup_epochs=1,
                beta_slowdown_epochs=0,
                free_nats_per_dim=0.0,
                no_beta_epochs=0,
            ),
        ),
    )

    device = torch.device('cpu')
    vae = ConditionalVAE(data_loader=dl, config=cfg, device=device)
    return vae


@pytest.fixture
def fake_batch():
    B, T, C, F = 2, 6, 1, 8
    x = torch.randn(B, T, C, F)
    sub_ids = torch.zeros(B, dtype=torch.long)
    return x, sub_ids


def test_hmm_gmm_forward_and_backward(fake_batch):
    x, sub_ids = fake_batch
    vae = _make_vae('hmm_gmm', T=x.shape[1], K=3, L=4)

    x_recon, mu, logvar, z = vae.forward(x, sub_ids)
    assert mu.shape == z.shape == (x.shape[0], x.shape[1], 4)
    assert x_recon.shape == x.shape

    recon_loss, reg_loss = vae.calculate_loss(x, x_recon, mu, logvar, z, epoch=0)
    total = recon_loss + reg_loss
    assert torch.isfinite(total)
    total.backward()

    # Gradients should reach transition logits and emission parameters.
    assert vae.prior_transition_logits.grad is not None
    assert torch.isfinite(vae.prior_transition_logits.grad).all()
    assert vae.prior_means.grad is not None
    assert torch.isfinite(vae.prior_means.grad).all()


def test_warm_hmm_gmm_phase_transitions(monkeypatch, fake_batch):
    x, sub_ids = fake_batch
    vae = _make_vae('warm_hmm_gmm', T=x.shape[1], K=3, L=4)

    # Avoid touching the mocked data loader inside __get_kmeans_centroids
    # and __init_hmm_transitions: patch them to no-ops that match the expected
    # tensor layouts.
    K, L = vae.num_gmm_states, vae.latent_dim
    fake_centroids = (torch.randn(K, L), torch.zeros(K, L), torch.zeros(K))
    monkeypatch.setattr(
        vae, '_ConditionalVAE__get_kmeans_centroids', lambda *_, **__: fake_centroids
    )
    monkeypatch.setattr(
        vae, '_ConditionalVAE__init_hmm_transitions', lambda *_, **__: None
    )

    epochs_and_branches = [
        (0, 'standard'),          # epoch < gmm_warmup_epochs
        (1, 'gmm_sequence'),      # gmm_warmup_epochs <= epoch < hmm_warmup_epochs
        (5, 'hmm'),               # epoch >= hmm_warmup_epochs
    ]
    for epoch, expected in epochs_and_branches:
        x_recon, mu, logvar, z = vae.forward(x, sub_ids)
        recon_loss, reg_loss = vae.calculate_loss(x, x_recon, mu, logvar, z, epoch=epoch)
        assert torch.isfinite(recon_loss + reg_loss), f"epoch {epoch} ({expected})"

    # After the loop the GMM init flag must be set.
    assert vae.gmm_warmup_initialized


def test_predict_hmm_labels_shape(fake_batch):
    x, sub_ids = fake_batch
    vae = _make_vae('hmm_gmm', T=x.shape[1], K=3, L=4)
    y, log_pz, mu = vae.predict_hmm_labels(x, sub_ids)
    assert y.shape == (x.shape[0], x.shape[1])
    assert mu.shape == (x.shape[0], x.shape[1], 4)
    assert torch.isfinite(log_pz)
