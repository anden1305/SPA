#!/usr/bin/env python3
"""Within-lab holdout — locked fair-comparison recipes (all labs × folds).

Train/val scoped to one lab; holdout mice from manifest fold split.
Same locks as joint: cGMVAE emb8 lr=1.3e-3; cHMM warm_emb8_sticky92 lr=1.3e-3.

See docs/cv4fold/unified_holdout_paper_line.md
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml

from scripts.cv4fold.generate_ablation_joint_chmm_fold4 import (
    _patch_emb,
    _patch_lr,
    _patch_sticky,
    _patch_warm_prior,
)
from scripts.cv4fold.locked_recipes import (
    build_unified_holdout_skeleton,
    extract_lab_cvae_overrides,
)
from scripts.cv4fold.manifest_utils import (
    all_mice,
    dataset_entries_with_lab_prepro,
    holdout_for_fold,
    load_manifest,
    train_mice,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "src/config/run/cvaemarhmm/cv4fold"
LOCKED_LR = 0.0013
HQ_LABS = ("lab_2", "lab_3", "lab_5")
ALL_FOLDS = (1, 2, 3, 4)


def _assert_hq_mice(manifest: dict, mouse_ids: list[str]) -> None:
    allowed = set(all_mice(manifest))
    extra = set(mouse_ids) - allowed
    if extra:
        raise ValueError(f"Mice not in HQ cohort: {sorted(extra)}")


def _patch_warm_emb8_sticky92(cfg: dict[str, Any]) -> dict[str, Any]:
    return _patch_sticky(_patch_emb(_patch_warm_prior(cfg), 8), 0.92)


def build_within_lab_base(manifest: dict, *, fold: int, lab: str, model: str) -> dict[str, Any]:
    holdout = holdout_for_fold(manifest, fold, "per_lab", lab)
    train = train_mice(manifest, fold, "per_lab", lab)
    _assert_hq_mice(manifest, holdout + train)
    train_entries = dataset_entries_with_lab_prepro(
        manifest, train, extract_cvae_overrides=extract_lab_cvae_overrides
    )
    val_entries = dataset_entries_with_lab_prepro(
        manifest, holdout, extract_cvae_overrides=extract_lab_cvae_overrides
    )
    cfg = build_unified_holdout_skeleton(model)
    cfg["train_datasets"] = copy.deepcopy(train_entries)
    cfg["val_datasets"] = copy.deepcopy(val_entries)
    return cfg


def _write_cfg(path: Path, cfg: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return path


def generate(
    manifest: dict,
    *,
    folds: list[int],
    labs: list[str],
) -> list[Path]:
    written: list[Path] = []
    for fold in folds:
        for lab in labs:
            chmm_cfg = _patch_lr(
                _patch_warm_emb8_sticky92(
                    build_within_lab_base(manifest, fold=fold, lab=lab, model="chmmgmvae")
                ),
                LOCKED_LR,
            )
            chmm_cfg["results_dir"] = (
                f"results/cv4fold/unified_holdout/{lab}/fold_{fold}/chmmgmvae_locked"
            )
            chmm_cfg["run_name"] = f"within_ho_f{fold}_{lab}_chmmgmvae_locked"
            chmm_cfg["runs"] = 3
            chmm_cfg["seed"] = 123
            chmm_path = _write_cfg(
                CONFIG_ROOT / "unified_holdout" / lab / f"fold_{fold}" / "chmmgmvae_locked.yaml",
                chmm_cfg,
            )
            written.append(chmm_path)
            print(
                f"Wrote {chmm_path} | {lab} fold {fold} | "
                f"train={len(chmm_cfg['train_datasets'])} val={len(chmm_cfg['val_datasets'])}"
            )

            cgmvae_cfg = _patch_lr(
                _patch_emb(
                    build_within_lab_base(manifest, fold=fold, lab=lab, model="cgmvae"),
                    8,
                ),
                LOCKED_LR,
            )
            cgmvae_cfg["results_dir"] = (
                f"results/cv4fold/unified_holdout/{lab}/fold_{fold}/cgmvae_locked"
            )
            cgmvae_cfg["run_name"] = f"within_ho_f{fold}_{lab}_cgmvae_locked"
            cgmvae_cfg["runs"] = 3
            cgmvae_cfg["seed"] = 123
            cgmvae_path = _write_cfg(
                CONFIG_ROOT / "unified_holdout" / lab / f"fold_{fold}" / "cgmvae_locked.yaml",
                cgmvae_cfg,
            )
            written.append(cgmvae_path)
            print(
                f"Wrote {cgmvae_path} | {lab} fold {fold} | "
                f"train={len(cgmvae_cfg['train_datasets'])} val={len(cgmvae_cfg['val_datasets'])}"
            )

            hmm_cfg = _patch_lr(
                build_within_lab_base(manifest, fold=fold, lab=lab, model="hmmgmvae"),
                LOCKED_LR,
            )
            hmm_cfg["results_dir"] = (
                f"results/cv4fold/unified_holdout/{lab}/fold_{fold}/hmmgmvae_locked"
            )
            hmm_cfg["run_name"] = f"within_ho_f{fold}_{lab}_hmmgmvae_locked"
            hmm_cfg["runs"] = 3
            hmm_cfg["seed"] = 123
            hmm_path = _write_cfg(
                CONFIG_ROOT / "unified_holdout" / lab / f"fold_{fold}" / "hmmgmvae_locked.yaml",
                hmm_cfg,
            )
            written.append(hmm_path)
            print(
                f"Wrote {hmm_path} | {lab} fold {fold} | "
                f"train={len(hmm_cfg['train_datasets'])} val={len(hmm_cfg['val_datasets'])}"
            )
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fold",
        nargs="+",
        type=int,
        default=list(ALL_FOLDS),
    )
    parser.add_argument(
        "--labs",
        nargs="+",
        default=list(HQ_LABS),
        choices=HQ_LABS,
    )
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    generate(manifest, folds=args.fold, labs=args.labs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
