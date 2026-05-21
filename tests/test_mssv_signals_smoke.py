"""Smoke tests: legacy MSSV loading (no signals) and lab_2 EEG1+EEG3+EMG selection."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.config.config import DatasetConfig, GlobalConfig
from src.data.mssv_dataset import MSSVDataset
from src.orchestrator.orchestrator import Orchestrator

DECODER_ONLY_SMOKE = (
    "src/config/run/cvaeprior/decoder_only/generalization_subject/"
    "generalization_subject_cgmvae_decoder_only_sub039_smoke.yaml"
)
LOLO_SMOKE = (
    "src/config/run/cvaeprior/lab_conditioning/lab_and_subject_conditioning_LOLO/"
    "generalization_lab_holdout_lab2_smoke.yaml"
)


def test_legacy_sub039_default_signals():
    """sub-039 without signals: lab_3 layout (EEG1, EEG2, EMG)."""
    ds = MSSVDataset(
        config=DatasetConfig(type="mssv", id="sub-039", run=1, remove_artifact=True)
    )
    assert ds.config["signals"] == ["EEG1", "EEG2", "EMG"]
    assert ds.config["n_channels"] == 3


def test_lab2_explicit_parietal_frontal_signals():
    """lab_2 with EEG1 (parietal) + EEG3 (frontal) + EMG."""
    ds = MSSVDataset(
        config=DatasetConfig(
            type="mssv",
            id="sub-071",
            run=1,
            remove_artifact=True,
            signals=["EEG1", "EEG3", "EMG"],
        )
    )
    assert ds.config["signals"] == ["EEG1", "EEG3", "EMG"]
    assert ds.config["n_channels"] == 3
    assert ds.data.shape[0] == 3


def test_lolo_smoke_config_unified_feature_dim():
    orch = Orchestrator(LOLO_SMOKE)
    assert orch.train_loader.get_feature_dim() == orch.val_loader.get_feature_dim()
    assert orch.train_loader.data_loaders[0].dataset.config["n_channels"] == 3


@pytest.mark.integration
def test_train_vae_decoder_only_smoke():
    """2-epoch train_vae on sub-039 only (no signals field)."""
    orch = Orchestrator(DECODER_ONLY_SMOKE)
    orch.train_cvae()


@pytest.mark.integration
def test_train_lolo_lab2_holdout_smoke():
    """2-epoch train on lab_3+lab_5, val lab_2 with mixed signal names."""
    orch = Orchestrator(LOLO_SMOKE)
    orch.train_cvae()
