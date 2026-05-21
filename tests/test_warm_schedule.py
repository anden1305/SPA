"""Phase gating tests for ``ConditionalVAE.warm_hmm_gmm_prior``."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from src.models.vae import ConditionalVAE


def _make_vae(K: int = 3, L: int = 4, T: int = 5):
    dl = MagicMock()
    dl.get_num_subjects.return_value = 1
    dl.get_num_labs.return_value = 1
    dl.get_vae_dims.return_value = (1, 8, T, K)

    cfg = SimpleNamespace(
        seed=0,
        num_states=K,
        model=SimpleNamespace(
            type='cvae_marhmm',
            conditioning_source='subject',
            params=dict(
                latent_dim=L,
                enc_hidden_dims=[8, 8],
                dec_hidden_dims=[8, 8],
                conv_channels=[4],
                kernel_sizes=[3],
                strides=[1],
                paddings=[1],
                emb_dim=0,
                decoder_only_conditioning=True,
                use_rms=False,
                prior='warm_hmm_gmm',
                num_gmm_states=K,
                gmm_warmup_epochs=3,
                hmm_warmup_epochs=6,
                hmm_transition_ramp_epochs=0,
                hmm_sticky_kappa=0.9,
                hmm_estimate_transitions=False,
                min_beta=0.0,
                max_beta=1.0,
                beta_warmup_epochs=0,
                beta_slowdown_epochs=0,
                free_nats_per_dim=0.0,
                no_beta_epochs=0,
            ),
        ),
    )
    return ConditionalVAE(data_loader=dl, config=cfg, device=torch.device('cpu'))


@pytest.mark.parametrize(
    "epoch,branch",
    [
        (0, 'standard'),
        (2, 'standard'),
        (3, 'gmm_sequence'),
        (5, 'gmm_sequence'),
        (6, 'hmm'),
        (10, 'hmm'),
    ],
)
def test_warm_schedule_branch_dispatch(monkeypatch, epoch: int, branch: str):
    vae = _make_vae()
    K, L = vae.num_gmm_states, vae.latent_dim

    # Avoid calling KMeans / data loader: pre-init centroids manually.
    fake_centroids = (torch.randn(K, L), torch.zeros(K, L), torch.zeros(K))
    monkeypatch.setattr(
        vae, '_ConditionalVAE__get_kmeans_centroids', lambda *_, **__: fake_centroids
    )

    called: dict[str, int] = {'standard': 0, 'gmm_sequence': 0, 'hmm': 0}
    orig_standard = vae.standard_prior
    orig_gmm_seq = vae.gmm_sequence_prior
    orig_hmm = vae.hmm_gmm_prior

    def wrap(key, fn):
        def inner(*args, **kwargs):
            called[key] += 1
            return fn(*args, **kwargs)
        return inner

    monkeypatch.setattr(vae, 'standard_prior', wrap('standard', orig_standard))
    monkeypatch.setattr(vae, 'gmm_sequence_prior', wrap('gmm_sequence', orig_gmm_seq))
    monkeypatch.setattr(vae, 'hmm_gmm_prior', wrap('hmm', orig_hmm))

    B, T = 2, 5
    mu = torch.randn(B, T, L, requires_grad=True)
    logvar = torch.zeros(B, T, L, requires_grad=True)
    z = mu + 0.0  # keep deterministic

    out = vae.warm_hmm_gmm_prior(logvar, mu, z, epoch=epoch)
    assert torch.isfinite(out)
    assert called[branch] >= 1, f"expected branch {branch} at epoch {epoch}, got {called}"
