"""Tests for raw_cnn temporal front-end in ConditionalVAE."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from src.models.temporal_front import build_temporal_front, build_temporal_front_decoder
from src.models.vae import ConditionalVAE


def test_temporal_front_roundtrip_shape():
    C, L_in = 3, 512
    channels = [32, 32]
    kernels = [15, 7]
    strides = [2, 2]
    paddings = [7, 3]
    pool = 2

    front, lens, out_ch = build_temporal_front(
        C, L_in, channels, kernels, strides, paddings, pool_kernel=pool
    )
    dec = build_temporal_front_decoder(C, channels, kernels, strides, paddings, pool, lens)

    x = torch.randn(4, C, L_in)
    h = front(x)
    x_hat = dec(h)
    assert x_hat.shape == x.shape


def test_conditional_vae_raw_cnn_forward_shape():
    dl = MagicMock()
    dl.get_num_subjects.return_value = 2
    dl.get_num_labs.return_value = 1
    dl.get_vae_dims.return_value = (3, 512, 4, 4)

    cfg = SimpleNamespace(
        seed=0,
        num_states=4,
        model=SimpleNamespace(
            conditioning_source="subject",
            params={
                "latent_dim": 4,
                "enc_hidden_dims": [16, 8],
                "dec_hidden_dims": [8, 16],
                "conv_channels": [8, 8],
                "kernel_sizes": [3, 3],
                "strides": [2, 2],
                "paddings": [1, 1],
                "emb_dim": 0,
                "decoder_only_conditioning": True,
                "prior": "standard",
                "min_beta": 0.0,
                "max_beta": 0.0,
                "no_beta_epochs": 10,
                "front_channels": [8, 8],
                "front_kernels": [15, 7],
                "front_strides": [2, 2],
                "front_paddings": [7, 3],
                "front_pool_kernel": 2,
                "raw_cnn_recon_domain": "spectral",
                "raw_cnn_fft_anchor_weight": 0.1,
                "raw_cnn_fft_anchor_decay_epochs": 10,
            },
        ),
        cvae=SimpleNamespace(feature_pipeline="raw_cnn"),
    )

    vae = ConditionalVAE(dl, cfg, torch.device("cpu"))
    B, S, C, T = 2, 4, 3, 512
    x = torch.randn(B, S, C, T)
    sub = torch.zeros(B, S, dtype=torch.long)

    x_recon, mu, logvar, z = vae(x, sub)
    assert x_recon.shape[0] == B and x_recon.shape[1] == S
    assert x_recon.shape[2] == vae.front_feature_channels
    assert x_recon.shape[3] == vae.front_feature_length
    assert mu.shape == (B, S, vae.latent_dim)

    loss, reg = vae.calculate_loss(x, x_recon, mu, logvar, z, epoch=0)
    assert torch.isfinite(loss)
    assert torch.isfinite(reg)
    loss.backward()
    assert vae.front_end is not None
    assert vae.front_decoder is None
