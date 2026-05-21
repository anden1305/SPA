"""Regression: standard / GMM / warm_gmm priors unchanged by HMM-GMM additions."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from src.models.vae import ConditionalVAE


def _make_vae(prior: str, *, T: int = 1, K: int = 4, L: int = 4) -> ConditionalVAE:
    dl = MagicMock()
    dl.get_num_subjects.return_value = 2
    dl.get_num_labs.return_value = 2
    dl.get_vae_dims.return_value = (1, 8, T, K)

    cfg = SimpleNamespace(
        seed=0,
        num_states=K,
        model=SimpleNamespace(
            type="cvae_marhmm",
            conditioning_source="subject",
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
                gmm_warmup_epochs=2,
                hmm_warmup_epochs=99,
                min_beta=0.1,
                max_beta=1.0,
                beta_warmup_epochs=0,
                beta_slowdown_epochs=0,
                free_nats_per_dim=0.0,
                no_beta_epochs=0,
            ),
        ),
    )
    return ConditionalVAE(data_loader=dl, config=cfg, device=torch.device("cpu"))


@pytest.mark.parametrize("prior", ["standard", "gmm", "warm_gmm"])
def test_legacy_priors_no_transition_params(prior: str):
    vae = _make_vae(prior, T=1)
    assert not hasattr(vae, "prior_transition_logits")


@pytest.mark.parametrize("prior", ["standard", "gmm", "warm_gmm"])
def test_legacy_priors_forward_and_loss(prior: str, monkeypatch):
    vae = _make_vae(prior, T=1)
    if prior == "warm_gmm":
        K, L = vae.num_gmm_states, vae.latent_dim
        fake = (torch.randn(K, L), torch.zeros(K, L), torch.zeros(K))
        monkeypatch.setattr(
            vae, "_ConditionalVAE__get_kmeans_centroids", lambda *_, **__: fake
        )
    x = torch.randn(2, 1, 1, 8)
    sub_ids = torch.zeros(2, dtype=torch.long)
    x_recon, mu, logvar, z = vae.forward(x, sub_ids)
    recon, reg = vae.calculate_loss(x, x_recon, mu, logvar, z, epoch=5)
    assert torch.isfinite(recon + reg)


def test_gmm_prior_sequence_length_one_matches_flattened():
    """gmm_prior with (B,1,L) should match behavior on (B*1,L)."""
    vae = _make_vae("gmm", T=1, K=3, L=4)
    B = 2
    mu = torch.randn(B, 1, 4)
    logvar = torch.zeros(B, 1, 4)
    z = mu + 0.1 * torch.randn_like(mu)
    kl_seq = vae.gmm_prior(logvar, mu, z)
    kl_flat = vae.gmm_prior(logvar.view(B, 4), mu.view(B, 4), z.view(B, 4))
    assert torch.allclose(kl_seq, kl_flat, atol=1e-5)


def test_hmm_priors_have_transition_logits_only():
    for prior in ("hmm_gmm", "warm_hmm_gmm"):
        vae = _make_vae(prior, T=4)
        assert hasattr(vae, "prior_transition_logits")
        assert vae.prior_transition_logits.shape == (4, 4)


def test_predict_gmm_labels_accepts_warm_hmm_after_init():
    vae = _make_vae("warm_hmm_gmm", T=4)
    vae.gmm_warmup_initialized = True
    x = torch.randn(2, 4, 1, 8)
    sub_ids = torch.zeros(2, dtype=torch.long)
    y, log_pz, mu = vae.predict_hmm_labels(x, sub_ids)
    assert y.shape == (2, 4)
    y_gmm, _, _ = vae.predict_gmm_labels(x, sub_ids)
    assert y_gmm.shape == (2, 4)
