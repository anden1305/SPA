#!/usr/bin/env python3
"""Generate cv4fold HMM-on-raw holdout configs (joint + within-lab, 1 run per fold).

Matches completed fold-4 joint recipe under results/cv4fold/hmm_raw/joint/fold_4/.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml

from scripts.cv4fold.manifest_utils import (
    all_mice,
    dataset_entries,
    holdout_for_fold,
    load_manifest,
    train_mice,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "src/config/run/hmm/cv4fold"
HQ_LABS = ("lab_2", "lab_3", "lab_5")
ALL_FOLDS = (1, 2, 3, 4)


def _hmm_raw_skeleton() -> dict[str, Any]:
    return {
        "trainer": {
            "epochs": 150,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "grad_clip": None,
            "validate_per_epoch": 50,
            "early_stopping": {"enabled": False, "patience": 100, "min_delta": 0.001},
            "scheduler": {"enabled": True, "type": "exponential", "step_size": None, "gamma": 0.99993},
        },
        "validator": {
            "nmi": True,
            "accuracy": True,
            "cross_nmi": True,
            "learning_rate": True,
            "state_distinctness": True,
            "summary_statistics": True,
            "log_likelihood": True,
        },
        "visualizer": {
            "losses": True,
            "learning_rate": True,
            "pca_tripanel": True,
            "confusion_matrix": True,
            "state_distinctness": True,
            "summary_statistics": True,
            "historic_values": True,
        },
        "dataloader": {
            "num_batches": 150,
            "batch_size": 120,
            "validation_batch_size": 4096,
            "window_size": None,
            "sequence_length": 1,
            "stride": None,
            "transforms": [
                {"type": "percentile_clipping", "channel": "EEG", "params": {"percentile": 0.5}},
                {"type": "percentile_clipping", "channel": "EMG", "params": {"percentile": 0.1}},
            ],
            "shuffle": True,
            "normalize": True,
            "use_legacy": True,
        },
        "model": {
            "type": "hmm",
            "covariance_type": "full",
            "init_strategy": "random_uniform",
            "init_noisy": True,
            "features": False,
            "conditioning_source": "subject",
            "n_states": 3,
            "params": {},
        },
        "verbose": True,
        "seed": 123,
        "runs": 1,
        "validate_data": True,
        "wandb": {"enabled": True, "group": None, "tags": None},
        "cvae": {
            "feature_pipeline": "fft",
            "normalize_global": False,
            "pre_normalize": False,
            "post_normalize": False,
            "percentile_clip_channels": False,
            "perform_hanning_window": False,
            "band_pass_filter_fft": False,
            "band_pass_freqs": None,
            "band_pass_filter_type": "frequency_domain",
            "traning_pipeline": "marhmm",
            "model_checkpoint_path": None,
            "reinit_marhmm": False,
        },
        "num_states": None,
    }


def _assert_hq_mice(manifest: dict, mouse_ids: list[str]) -> None:
    allowed = set(all_mice(manifest))
    extra = set(mouse_ids) - allowed
    if extra:
        raise ValueError(f"Mice not in HQ cohort: {sorted(extra)}")


def _build_cfg(manifest: dict, *, fold: int, scope: str, lab: str | None) -> dict[str, Any]:
    if scope == "joint":
        holdout = holdout_for_fold(manifest, fold, "joint")
        train = train_mice(manifest, fold, "joint")
    else:
        assert lab is not None
        holdout = holdout_for_fold(manifest, fold, "per_lab", lab)
        train = train_mice(manifest, fold, "per_lab", lab)
    _assert_hq_mice(manifest, holdout + train)
    cfg = copy.deepcopy(_hmm_raw_skeleton())
    cfg["train_datasets"] = dataset_entries(manifest, train)
    cfg["val_datasets"] = dataset_entries(manifest, holdout)
    return cfg


def _write(path: Path, cfg: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return path


def generate(manifest: dict, *, folds: list[int], labs: list[str]) -> list[Path]:
    written: list[Path] = []
    for fold in folds:
        cfg = _build_cfg(manifest, fold=fold, scope="joint", lab=None)
        cfg["results_dir"] = f"results/cv4fold/hmm_raw/joint/fold_{fold}"
        cfg["run_name"] = f"cv4fold_hmm_raw_joint_fold{fold}"
        path = _write(CONFIG_ROOT / "joint_holdout" / f"fold_{fold}" / "hmm_raw.yaml", cfg)
        written.append(path)
        print(f"Wrote {path} | joint fold {fold} | train={len(cfg['train_datasets'])} val={len(cfg['val_datasets'])}")

        for lab in labs:
            cfg_w = _build_cfg(manifest, fold=fold, scope="per_lab", lab=lab)
            cfg_w["results_dir"] = f"results/cv4fold/hmm_raw/unified_holdout/{lab}/fold_{fold}"
            cfg_w["run_name"] = f"cv4fold_hmm_raw_within_{lab}_fold{fold}"
            path_w = _write(
                CONFIG_ROOT / "unified_holdout" / lab / f"fold_{fold}" / "hmm_raw.yaml",
                cfg_w,
            )
            written.append(path_w)
            print(
                f"Wrote {path_w} | {lab} fold {fold} | "
                f"train={len(cfg_w['train_datasets'])} val={len(cfg_w['val_datasets'])}"
            )
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--folds", type=int, nargs="*", default=list(ALL_FOLDS))
    parser.add_argument("--labs", default=",".join(HQ_LABS))
    args = parser.parse_args()
    labs = [x.strip() for x in args.labs.split(",") if x.strip()]
    manifest = load_manifest(args.manifest)
    generate(manifest, folds=args.folds, labs=labs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
