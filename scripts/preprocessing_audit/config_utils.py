"""Config loading and ablation overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from src.config.config import GlobalConfig

REPO_ROOT = Path(__file__).resolve().parents[2]

ABLATION_VARIANTS: dict[str, dict[str, Any]] = {
    "baseline": {},
    "+percentile_clip": {"cvae.percentile_clip_channels": True},
    "+pre_normalize": {"cvae.pre_normalize": True},
    "hanning_on": {"cvae.perform_hanning_window": True},
    "bp_time_domain": {
        "cvae.band_pass_filter_type": "time_domain",
        "cvae.band_pass_freqs": [[0.5, 30.0], [0.5, 30.0], [5.0, 60.0]],
    },
    "bp_placement_aware": {
        "cvae.band_pass_freqs": [[None, 20.0], [None, 20.0], [5.0, 60.0]],
    },
    "no_post_normalize": {"cvae.post_normalize": False},
    "global_norm": {"cvae.normalize_global": True},
}


def load_global_config(config_path: Path) -> GlobalConfig:
    import yaml

    with config_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    data.setdefault("run_name", "preprocessing_audit")
    data.setdefault("train_datasets", [{"type": "mssv", "id": "sub-038", "run": 1}])
    data.setdefault("val_datasets", [{"type": "mssv", "id": "sub-039", "run": 1}])
    return GlobalConfig.model_validate(data)


def snapshot_baseline_config(config_path: Path, out_path: Path) -> None:
    cfg = load_global_config(config_path)
    snapshot = {
        "dataloader": cfg.dataloader.model_dump(),
        "cvae": cfg.cvae.model_dump(),
        "source_config": str(config_path),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(snapshot, f, default_flow_style=False, sort_keys=False)


def apply_overrides(cfg: GlobalConfig, overrides: dict[str, Any]) -> GlobalConfig:
    data = cfg.model_dump()
    for dotted, value in overrides.items():
        parts = dotted.split(".")
        node = data
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return GlobalConfig.model_validate(data)


def config_for_variant(base: GlobalConfig, variant: str) -> GlobalConfig:
    overrides = ABLATION_VARIANTS.get(variant, {})
    if variant == "artifact_harmonized":
        return copy.deepcopy(base)
    return apply_overrides(base, overrides)
