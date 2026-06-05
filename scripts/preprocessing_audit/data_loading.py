"""Load cohort recordings the same way training does."""

from __future__ import annotations

import numpy as np

from src.config.config import DatasetConfig, GlobalConfig
from src.data.mssv_dataset import MSSVDataset
from src.preprocessing.vae_preprocessing import VAEPreprocessing

from scripts.preprocessing_audit.manifest import RunRecord

STAGE_NAMES_4 = ["Awake", "NREM", "REM", "Artifact"]
STAGE_NAMES_3 = ["Awake", "NREM", "REM"]


def load_mssv_array(rec: RunRecord, remove_artifact: bool = True) -> tuple[np.ndarray, np.ndarray, dict]:
    ds = MSSVDataset(
        config=DatasetConfig(
            type="mssv",
            id=rec.participant_id,
            run=rec.run,
            remove_artifact=remove_artifact,
            signals=rec.signals,
        )
    )
    x, y = ds[:]
    meta = {
        "lab": ds.config["lab"],
        "stage_names": list(ds.config["stage_names"]),
        "n_timesteps_after": x.shape[1],
        "signals": list(ds.config["signals"]),
        "sampling_rate": ds.get_sampling_rate(),
    }
    return x, y, meta


def run_vae_preprocessing(
    cfg: GlobalConfig,
    x: np.ndarray,
    y: np.ndarray,
    *,
    for_raw: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    dl = cfg.dataloader
    prep = VAEPreprocessing(
        cfg,
        window_size=dl.window_size,
        stride=dl.stride,
        sequence_length=dl.sequence_length,
        normalize=dl.normalize,
        sampling_rate=128,
    )
    return prep(x.copy(), y.copy(), for_raw=for_raw)


def subsample_sequences(
    x: np.ndarray,
    y: np.ndarray,
    max_sequences: int,
    seed: int = 123,
) -> tuple[np.ndarray, np.ndarray]:
    n = x.shape[0]
    if n <= max_sequences:
        return x, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_sequences, replace=False)
    idx.sort()
    return x[idx], y[idx]
