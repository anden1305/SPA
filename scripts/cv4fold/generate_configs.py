#!/usr/bin/env python3
"""Generate cv4fold YAML configs from the HQ quality-cohort manifest."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

from scripts.cv4fold.locked_recipes import (
    LOCKED_CHMM_SEQUENCE_LENGTH,
    apply_locked_sequence_length,
    apply_model_variant,
    load_locked_recipe,
)
from scripts.cv4fold.manifest_utils import (
    all_mice,
    dataset_entries,
    holdout_for_fold,
    load_manifest,
    train_mice,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "src/config/run/cvaemarhmm/cv4fold"

HQ_LABS = ("lab_2", "lab_3", "lab_5")
HOLDOUT_MODELS = ("cgmvae", "chmmgmvae")


def _assert_hq_mice(manifest: dict, mouse_ids: list[str]) -> None:
    allowed = set(all_mice(manifest))
    extra = set(mouse_ids) - allowed
    if extra:
        raise ValueError(f"Mice not in HQ cohort: {sorted(extra)}")


def _mice_for_lab(manifest: dict, lab: str) -> list[str]:
    if lab not in manifest["cohort"]:
        raise KeyError(f"Unknown lab {lab!r}; expected one of {HQ_LABS}")
    return list(manifest["cohort"][lab])


def generate_per_lab_incohort(manifest: dict, labs: list[str]) -> list[Path]:
    """All HQ mice in lab for train and val (no holdout); uses locked ablation recipes."""
    written: list[Path] = []
    for lab in labs:
        mice = _mice_for_lab(manifest, lab)
        _assert_hq_mice(manifest, mice)
        entries = dataset_entries(manifest, mice)

        cfg = apply_model_variant(load_locked_recipe(lab), "cgmvae")
        cfg["train_datasets"] = copy.deepcopy(entries)
        cfg["val_datasets"] = copy.deepcopy(entries)
        cfg["results_dir"] = f"results/cv4fold/per_lab_incohort/{lab}"
        cfg["run_name"] = f"per_lab_incohort_{lab}_cgmvae"

        out_dir = CONFIG_ROOT / "per_lab_incohort" / lab
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "cgmvae.yaml"
        with out_path.open("w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        written.append(out_path)
        print(f"Wrote {out_path} ({len(mice)} HQ mice: {', '.join(mice)})")
    return written


def generate_per_lab_holdout(
    manifest: dict,
    *,
    fold: int,
    labs: list[str],
    models: list[str],
) -> list[Path]:
    """Within-lab train/val split: holdout mice from manifest fold_k; locked ablation recipes."""
    written: list[Path] = []
    for lab in labs:
        holdout = holdout_for_fold(manifest, fold, "per_lab", lab)
        train = train_mice(manifest, fold, "per_lab", lab)
        _assert_hq_mice(manifest, holdout + train)

        train_entries = dataset_entries(manifest, train)
        val_entries = dataset_entries(manifest, holdout)

        for model in models:
            if model == "chmmgmvae" and lab not in LOCKED_CHMM_SEQUENCE_LENGTH:
                print(f"Skip {lab} chmmgmvae (no locked seq T; use cgmvae holdout)")
                continue

            cfg = load_locked_recipe(lab)
            cfg = apply_model_variant(cfg, model, lab=lab)
            cfg = apply_locked_sequence_length(cfg, model, lab)
            cfg.setdefault("visualizer", {})["save_results_npz"] = True
            cfg["train_datasets"] = copy.deepcopy(train_entries)
            cfg["val_datasets"] = copy.deepcopy(val_entries)
            cfg["results_dir"] = f"results/cv4fold/per_lab_holdout/{lab}/fold_{fold}/{model}"
            cfg["run_name"] = f"holdout_f{fold}_{lab}_{model}"

            out_dir = CONFIG_ROOT / "per_lab_holdout" / lab / f"fold_{fold}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{model}.yaml"
            with out_path.open("w", encoding="utf-8") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
            written.append(out_path)
            print(
                f"Wrote {out_path} | train={len(train)} mice | holdout={', '.join(holdout)}"
            )
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("per_lab_incohort", "per_lab_holdout"),
        required=True,
    )
    parser.add_argument(
        "--fold",
        type=int,
        choices=(1, 2, 3, 4),
        default=4,
        help="Holdout fold (manifest splits); default 4 for pilot",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(HOLDOUT_MODELS),
        choices=HOLDOUT_MODELS,
        help="VAE model variants for per_lab_holdout",
    )
    parser.add_argument(
        "--labs",
        nargs="+",
        default=list(HQ_LABS),
        help=f"HQ labs to generate. Choices: {HQ_LABS}",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to cv_quality_cohort_v1.yaml",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    for lab in args.labs:
        if lab not in HQ_LABS:
            raise ValueError(f"Invalid lab {lab!r}; must be one of {HQ_LABS}")

    if args.phase == "per_lab_incohort":
        generate_per_lab_incohort(manifest, args.labs)
    elif args.phase == "per_lab_holdout":
        generate_per_lab_holdout(
            manifest,
            fold=args.fold,
            labs=args.labs,
            models=args.models,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
