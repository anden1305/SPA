#!/usr/bin/env python3
"""Generate cv4fold YAML configs from the HQ quality-cohort manifest."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

from scripts.cv4fold.locked_recipes import (
    LOCKED_CHMM_SEQUENCE_LENGTH,
    PAPER_MODELS,
    apply_locked_sequence_length,
    apply_model_variant,
    build_unified_holdout_skeleton,
    extract_lab_cvae_overrides,
    load_locked_recipe,
)
from scripts.cv4fold.manifest_utils import (
    all_mice,
    dataset_entries,
    dataset_entries_with_lab_prepro,
    holdout_for_fold,
    load_manifest,
    train_mice,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "src/config/run/cvaemarhmm/cv4fold"

HQ_LABS = ("lab_2", "lab_3", "lab_5")
HOLDOUT_MODELS = ("cgmvae", "chmmgmvae")
ALL_FOLDS = (1, 2, 3, 4)


def _assert_hq_mice(manifest: dict, mouse_ids: list[str]) -> None:
    allowed = set(all_mice(manifest))
    extra = set(mouse_ids) - allowed
    if extra:
        raise ValueError(f"Mice not in HQ cohort: {sorted(extra)}")


def _mice_for_lab(manifest: dict, lab: str) -> list[str]:
    if lab not in manifest["cohort"]:
        raise KeyError(f"Unknown lab {lab!r}; expected one of {HQ_LABS}")
    return list(manifest["cohort"][lab])


def _parse_folds(fold_args: list[str | int]) -> list[int]:
    if len(fold_args) == 1 and str(fold_args[0]).lower() == "all":
        return list(ALL_FOLDS)
    return [int(f) for f in fold_args]


def _write_config(path: Path, cfg: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return path


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
        out_path = out_dir / "cgmvae.yaml"
        written.append(_write_config(out_path, cfg))
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

            out_path = CONFIG_ROOT / "per_lab_holdout" / lab / f"fold_{fold}" / f"{model}.yaml"
            written.append(_write_config(out_path, cfg))
            print(
                f"Wrote {out_path} | train={len(train)} mice | holdout={', '.join(holdout)}"
            )
    return written


def generate_within_lab_holdout(
    manifest: dict,
    *,
    fold: int,
    labs: list[str],
    models: list[str],
) -> list[Path]:
    """Paper line: unified recipe per model, within-lab train scope, per-lab cvae_overrides."""
    written: list[Path] = []
    for lab in labs:
        holdout = holdout_for_fold(manifest, fold, "per_lab", lab)
        train = train_mice(manifest, fold, "per_lab", lab)
        _assert_hq_mice(manifest, holdout + train)

        train_entries = dataset_entries_with_lab_prepro(
            manifest, train, extract_cvae_overrides=extract_lab_cvae_overrides
        )
        val_entries = dataset_entries_with_lab_prepro(
            manifest, holdout, extract_cvae_overrides=extract_lab_cvae_overrides
        )

        for model in models:
            cfg = build_unified_holdout_skeleton(model)
            cfg["train_datasets"] = copy.deepcopy(train_entries)
            cfg["val_datasets"] = copy.deepcopy(val_entries)
            cfg["results_dir"] = f"results/cv4fold/unified_holdout/{lab}/fold_{fold}/{model}"
            cfg["run_name"] = f"unified_ho_f{fold}_{lab}_{model}"

            out_path = CONFIG_ROOT / "unified_holdout" / lab / f"fold_{fold}" / f"{model}.yaml"
            written.append(_write_config(out_path, cfg))
            print(
                f"Wrote {out_path} | train={len(train)} mice | holdout={', '.join(holdout)}"
            )
    return written


def generate_joint_holdout(
    manifest: dict,
    *,
    fold: int,
    models: list[str],
) -> list[Path]:
    """Paper line: unified recipe per model, joint train scope, per-lab cvae_overrides."""
    written: list[Path] = []
    holdout = holdout_for_fold(manifest, fold, "joint")
    train = train_mice(manifest, fold, "joint")
    _assert_hq_mice(manifest, holdout + train)

    train_entries = dataset_entries_with_lab_prepro(
        manifest, train, extract_cvae_overrides=extract_lab_cvae_overrides
    )
    val_entries = dataset_entries_with_lab_prepro(
        manifest, holdout, extract_cvae_overrides=extract_lab_cvae_overrides
    )

    for model in models:
        cfg = build_unified_holdout_skeleton(model)
        cfg["train_datasets"] = copy.deepcopy(train_entries)
        cfg["val_datasets"] = copy.deepcopy(val_entries)
        cfg["results_dir"] = f"results/cv4fold/joint_holdout/fold_{fold}/{model}"
        cfg["run_name"] = f"joint_ho_f{fold}_{model}"

        out_path = CONFIG_ROOT / "joint_holdout" / f"fold_{fold}" / f"{model}.yaml"
        written.append(_write_config(out_path, cfg))
        print(
            f"Wrote {out_path} | train={len(train)} mice | "
            f"holdout={len(holdout)} mice across labs"
        )
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=(
            "per_lab_incohort",
            "per_lab_holdout",
            "within_lab_holdout",
            "joint_holdout",
        ),
        required=True,
    )
    parser.add_argument(
        "--fold",
        nargs="+",
        default=["4"],
        help="Holdout fold(s): 1 2 3 4 or 'all' (default: 4)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(PAPER_MODELS),
        choices=PAPER_MODELS,
        help="VAE model variants for paper-line holdout phases",
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

    folds = _parse_folds(args.fold)

    if args.phase == "per_lab_incohort":
        generate_per_lab_incohort(manifest, args.labs)
    elif args.phase == "per_lab_holdout":
        for fold in folds:
            generate_per_lab_holdout(
                manifest,
                fold=fold,
                labs=args.labs,
                models=[m for m in args.models if m in HOLDOUT_MODELS],
            )
    elif args.phase == "within_lab_holdout":
        for fold in folds:
            generate_within_lab_holdout(
                manifest,
                fold=fold,
                labs=args.labs,
                models=args.models,
            )
    elif args.phase == "joint_holdout":
        for fold in folds:
            generate_joint_holdout(
                manifest,
                fold=fold,
                models=args.models,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
