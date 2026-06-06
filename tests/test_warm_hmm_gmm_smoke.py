"""Smoke test: warm_hmm_gmm prior with seq_len=1 forward + loss."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from src.models.vae import ConditionalVAE


def _mock_config() -> SimpleNamespace:
    return SimpleNamespace(
        seed=0,
        model=SimpleNamespace(
            params={
                "conv_channels": [8, 16],
                "kernel_sizes": [3, 3],
                "strides": [2, 2],
                "paddings": [1, 1],
                "latent_dim": 4,
                "enc_hidden_dims": [32],
                "dec_hidden_dims": [32],
                "emb_dim": 2,
                "decoder_only_conditioning": True,
                "min_beta": 0.01,
                "max_beta": 1.0,
                "beta_warmup_epochs": 0,
                "beta_slowdown_epochs": 0,
                "no_beta_epochs": 0,
                "prior": "warm_hmm_gmm",
                "gmm_warmup_epochs": 1,
                "hmm_warmup_epochs": 2,
                "hmm_transition_ramp_epochs": 1,
                "hmm_sticky_kappa": 0.9,
                "hmm_estimate_transitions": True,
                "free_nats_per_dim": 0.0,
                "num_gmm_states": 3,
            }
        ),
    )


def test_warm_hmm_gmm_seq_len_one_forward():
    B, S, C, F = 8, 1, 3, 32
    x_pool = torch.randn(B, S, C, F)
    subject_ids_pool = torch.arange(B)

    dl = MagicMock()
    dl.get_num_subjects.return_value = 4
    dl.get_vae_dims.return_value = (C, F, S, 3)
    dl.get_all_data.return_value = (x_pool, None, subject_ids_pool)

    model = ConditionalVAE(dl, _mock_config(), torch.device("cpu"))
    x = torch.randn(2, S, C, F)
    subject_ids = torch.tensor([0, 1])

    x_recon, mu, logvar, z = model(x, subject_ids)
    assert x_recon.shape == x.shape
    assert z.shape == (2, S, model.latent_dim)

    for epoch in (0, 1, 3):
        recon_loss, reg_loss = model.calculate_loss(x, x_recon, mu, logvar, z, epoch)
        assert torch.isfinite(recon_loss)
        assert torch.isfinite(reg_loss)
