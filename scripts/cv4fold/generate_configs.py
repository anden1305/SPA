#!/usr/bin/env python3
"""Generate cv4fold YAML configs from the HQ quality-cohort manifest."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

from scripts.cv4fold.manifest_utils import (
    all_mice,
    dataset_entries,
    load_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_PATH = (
    REPO_ROOT / "src/config/run/cvaemarhmm/cv4fold/templates/cgmvae_per_lab_incohort.yaml"
)
CONFIG_ROOT = REPO_ROOT / "src/config/run/cvaemarhmm/cv4fold"

HQ_LABS = ("lab_2", "lab_3", "lab_5")


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
    """All HQ mice in lab for train and val (no holdout)."""
    with TEMPLATE_PATH.open(encoding="utf-8") as f:
        template = yaml.safe_load(f)

    written: list[Path] = []
    for lab in labs:
        mice = _mice_for_lab(manifest, lab)
        _assert_hq_mice(manifest, mice)
        entries = dataset_entries(manifest, mice)

        cfg = copy.deepcopy(template)
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("per_lab_incohort",),
        required=True,
    )
    parser.add_argument(
        "--labs",
        nargs="+",
        default=["lab_5", "lab_2"],
        help=f"HQ labs to generate (default: lab_5 lab_2). Choices: {HQ_LABS}",
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
