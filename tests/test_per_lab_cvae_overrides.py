"""Per-lab cvae_overrides for paper-line holdout configs."""

from __future__ import annotations

from pathlib import Path

import yaml

from scripts.cv4fold.locked_recipes import (
    UNIFIED_EMB_DIM,
    UNIFIED_JOINT_HMM_SEQUENCE_LENGTH,
    UNIFIED_LATENT_DIM,
    apply_unified_recipe,
    build_unified_holdout_skeleton,
    extract_lab_cvae_overrides,
)
from scripts.cv4fold.manifest_utils import (
    dataset_entries_with_lab_prepro,
    load_manifest,
    train_mice,
)
from src.config.config import GlobalConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
JOINT_PILOT = (
    REPO_ROOT
    / "src/config/run/cvaemarhmm/cv4fold/joint_holdout/fold_4/chmmgmvae.yaml"
)


def test_extract_lab_cvae_overrides_differ_by_lab() -> None:
    lab_2 = extract_lab_cvae_overrides("lab_2")
    lab_3 = extract_lab_cvae_overrides("lab_3")
    assert lab_2["post_normalize"] is False
    assert lab_3["post_normalize"] is True
    assert lab_2["band_pass_freqs"] != lab_3["band_pass_freqs"]


def test_unified_chmm_recipe_locked_dims() -> None:
    cfg = build_unified_holdout_skeleton("chmmgmvae")
    params = cfg["model"]["params"]
    assert params["latent_dim"] == UNIFIED_LATENT_DIM == 6
    assert params["emb_dim"] == UNIFIED_EMB_DIM == 4
    assert cfg["dataloader"]["sequence_length"] == UNIFIED_JOINT_HMM_SEQUENCE_LENGTH
    assert params["prior"] == "hmm_gmm"
    assert params["enc_hidden_dims"] == [512, 256, 128]


def test_hmmgmvae_no_embedding() -> None:
    cfg = apply_unified_recipe({}, "hmmgmvae")
    assert cfg["model"]["params"]["emb_dim"] == 0


def test_dataset_entries_with_lab_prepro() -> None:
    manifest = load_manifest()
    mice = train_mice(manifest, fold=4, scope="joint")[:3]
    entries = dataset_entries_with_lab_prepro(
        manifest, mice, extract_cvae_overrides=extract_lab_cvae_overrides
    )
    assert all("cvae_overrides" in e for e in entries)
    assert all("post_normalize" in e["cvae_overrides"] for e in entries)


def test_joint_pilot_yaml_roundtrip() -> None:
    if not JOINT_PILOT.exists():
        return
    cfg = GlobalConfig.from_yaml(str(JOINT_PILOT))
    assert cfg.dataloader.sequence_length == 64
    assert cfg.model.params["latent_dim"] == 6
    assert cfg.train_datasets[0].cvae_overrides is not None
    raw = yaml.safe_load(JOINT_PILOT.read_text(encoding="utf-8"))
    assert raw["train_datasets"][0]["cvae_overrides"]["post_normalize"] in (True, False)
