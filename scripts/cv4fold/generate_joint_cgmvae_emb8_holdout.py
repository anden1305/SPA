#!/usr/bin/env python3
"""Joint holdout cGMVAE with emb8 for architecture-fair comparison to tuned cHMM.

Shared with cHMM lock: latent_dim=6, emb_dim=8, wide_mlp, per-lab cvae_overrides.
cGMVAE-specific: T=1, prior=gmm, lr=3e-4.

See docs/cv4fold/ablations/ablation_joint_chmm_fold4.md.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml

from scripts.cv4fold.generate_ablation_joint_chmm_fold4 import _patch_emb
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


def _assert_hq_mice(manifest: dict, mouse_ids: list[str]) -> None:
    allowed = set(all_mice(manifest))
    extra = set(mouse_ids) - allowed
    if extra:
        raise ValueError(f"Mice not in HQ cohort: {sorted(extra)}")


def build_joint_cgmvae_base(manifest: dict, *, fold: int) -> dict[str, Any]:
    holdout = holdout_for_fold(manifest, fold, "joint")
    train = train_mice(manifest, fold, "joint")
    _assert_hq_mice(manifest, holdout + train)
    train_entries = dataset_entries_with_lab_prepro(
        manifest, train, extract_cvae_overrides=extract_lab_cvae_overrides
    )
    val_entries = dataset_entries_with_lab_prepro(
        manifest, holdout, extract_cvae_overrides=extract_lab_cvae_overrides
    )
    cfg = build_unified_holdout_skeleton("cgmvae")
    cfg["train_datasets"] = copy.deepcopy(train_entries)
    cfg["val_datasets"] = copy.deepcopy(val_entries)
    return cfg


def generate(manifest: dict, *, folds: list[int]) -> list[Path]:
    written: list[Path] = []
    for fold in folds:
        base = build_joint_cgmvae_base(manifest, fold=fold)
        cfg = _patch_emb(base, 8)
        cfg["results_dir"] = f"results/cv4fold/joint_holdout/fold_{fold}/cgmvae_emb8"
        cfg["run_name"] = f"joint_ho_f{fold}_cgmvae_emb8"
        cfg["runs"] = 3
        cfg["seed"] = 123

        out_dir = CONFIG_ROOT / "joint_holdout" / f"fold_{fold}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "cgmvae_emb8.yaml"
        with out_path.open("w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        written.append(out_path)
        print(f"Wrote {out_path} (latent=6, emb=8, T=1, gmm, lr=3e-4)")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fold",
        nargs="+",
        type=int,
        default=[4],
        help="Holdout folds (default: 4).",
    )
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    generate(manifest, folds=args.fold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
