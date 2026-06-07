"""Parity tests for training-path performance optimizations (same outputs)."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from src.helpers.nmi import calculate_nmi
from src.models.cvae_mar_hmm import CVAEMARHMM
from src.models.vae import ConditionalVAE


def _mock_config(*, prior: str = "warm_hmm_gmm") -> SimpleNamespace:
    return SimpleNamespace(
        seed=0,
        model=SimpleNamespace(
            type="cvae_marhmm",
            covariance_type="full",
            init_strategy="random_uniform",
            init_noisy=True,
            features=False,
            params={
                "lags": [1, 2, 4],
                "ridge": 0.1,
                "var_reg": 0.05,
                "sticky_coef": 0.1,
                "sticky_kappa": 0.9,
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
                "prior": prior,
                "num_gmm_states": 3,
                "gmm_warmup_epochs": 1,
                "hmm_warmup_epochs": 2,
                "hmm_transition_ramp_epochs": 1,
                "hmm_sticky_kappa": 0.9,
                "hmm_estimate_transitions": True,
                "free_nats_per_dim": 0.0,
            },
        ),
        cvae=SimpleNamespace(
            normalize_global=False,
            pre_normalize=False,
            post_normalize=False,
            band_pass_filter_fft=False,
            traning_pipeline="cvae",
        ),
        dataloader=SimpleNamespace(
            batch_size=4,
            validation_batch_size=4,
            num_batches=4,
            window_size=512,
            stride=512,
            sequence_length=1,
            shuffle=False,
            normalize=False,
            use_legacy=False,
            transforms=[],
        ),
        trainer=SimpleNamespace(
            epochs=5,
            learning_rate=1e-3,
            optimizer="adam",
            grad_clip=0.5,
            validate_per_epoch=1,
            checkpoint_score=SimpleNamespace(enabled=False, beta=0.9, warmup_frac=0.05),
            early_stopping=SimpleNamespace(enabled=False, patience=10, min_delta=0.001),
            scheduler=SimpleNamespace(enabled=False, type="exponential", step_size=None, gamma=0.99),
        ),
        validator=SimpleNamespace(
            nmi=True,
            accuracy=True,
            cross_nmi=True,
            learning_rate=True,
            state_distinctness=True,
            summary_statistics=True,
            log_likelihood=True,
        ),
        visualizer=SimpleNamespace(
            losses=True,
            learning_rate=True,
            pca_tripanel=True,
            confusion_matrix=True,
            state_distinctness=True,
            summary_statistics=True,
            historic_values=True,
        ),
        runs=1,
        verbose=False,
        validate_data=False,
        num_states=3,
        results_dir="results/test",
        run_name="test",
    )


def _mock_dataloader(B: int = 16, S: int = 1, C: int = 3, F: int = 32) -> MagicMock:
    x_pool = torch.randn(B, S, C, F)
    sub_ids_pool = torch.arange(B)
    dl = MagicMock()
    dl.get_num_subjects.return_value = 8
    dl.get_vae_dims.return_value = (C, F, S, 3)
    dl.get_num_states.return_value = 3
    dl.get_all_data.return_value = (x_pool, torch.zeros(B, S, dtype=torch.long), sub_ids_pool)
    dl.get_subject_map.return_value = {"sub-001": 1}
    dl.get_channel_signal_names.return_value = ["EEG1", "EEG3", "EMG"]
    dl.get_state_names.return_value = ["W", "N", "R"]
    dl.get_feature_names.return_value = []
    dl.get_feature_dim.return_value = 3
    dl.get_transforms.return_value = []
    dl.has_features_enabled.return_value = False
    dl.device = torch.device("cpu")
    return dl


def test_predict_hmm_labels_mu_matches_reencode():
    torch.manual_seed(42)
    dl = _mock_dataloader()
    cfg = _mock_config()
    model = ConditionalVAE(dl, cfg, torch.device("cpu"))
    x = torch.randn(4, 1, 3, 32)
    sub_ids = torch.tensor([0, 1, 2, 3])

    for epoch in (2, 3):
        model.gmm_warmup_initialized = True
        model.hmm_transitions_initialized_at_hmm = True
        _, _, _, z = model.forward(x, sub_ids)
        mu = model.encode_to_latent(x, sub_ids)
        y_full, log_pz_full, _ = model.predict_hmm_labels(x, sub_ids)
        y_cached, log_pz_cached, _ = model.predict_hmm_labels(x, sub_ids, mu=mu)
        assert torch.equal(y_full, y_cached)
        assert torch.allclose(log_pz_full, log_pz_cached, rtol=0, atol=1e-6)


def test_predict_gmm_labels_mu_matches_reencode():
    torch.manual_seed(7)
    dl = _mock_dataloader()
    cfg = _mock_config(prior="gmm")
    model = ConditionalVAE(dl, cfg, torch.device("cpu"))
    model.gmm_warmup_initialized = True
    x = torch.randn(4, 1, 3, 32)
    sub_ids = torch.tensor([0, 1, 2, 3])
    mu = model.encode_to_latent(x, sub_ids)
    y_full, _, _ = model.predict_gmm_labels(x, sub_ids, initialize_if_needed=True)
    y_cached, _, _ = model.predict_gmm_labels(x, sub_ids, initialize_if_needed=True, mu=mu)
    assert torch.equal(y_full, y_cached)


def test_marhmm_forward_encode_only_matches_full_cvae_path():
    """MARHMM phase: encode-only forward must match loss from full CVAE forward (frozen encoder)."""
    torch.manual_seed(99)
    dl = _mock_dataloader(B=8)
    cfg = _mock_config(prior="gmm")
    device = torch.device("cpu")
    model = CVAEMARHMM(dl, cfg, device, training_pipeline="marhmm")
    model.prepare_for_training()
    x = torch.randn(4, 1, 3, 32)
    sub_ids = torch.tensor([0, 1, 2, 3])
    epoch = 0

    mu = model.cvae.encode_to_latent(x, sub_ids)
    loss_fast, reg_fast = model.forward(x, sub_ids, epoch)

    x_recon, mu_full, logvar, z = model.cvae.forward(x, sub_ids)
    loss_marhmm = model.marhmm(mu_full)
    reg_marhmm = model.marhmm.regularization_loss()

    assert torch.allclose(mu, mu_full, rtol=0, atol=1e-6)
    assert torch.allclose(loss_fast, loss_marhmm, rtol=0, atol=1e-6)
    assert torch.allclose(reg_fast, reg_marhmm, rtol=0, atol=1e-6)


def test_calculate_nmi_tensor_matches_numpy():
    pred = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1])
    target = torch.tensor([0, 1, 1, 1, 2, 0, 0, 1])
    nmi_tensor = calculate_nmi(pred, target)
    nmi_numpy = calculate_nmi(pred.cpu().numpy(), target.cpu().numpy())
    assert nmi_tensor == nmi_numpy
