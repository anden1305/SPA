#!/usr/bin/env python3
"""Generate incohort cHMM-GMVAE ablation configs (scratch only, no checkpoint hotstart).

Phases (use ``--sequence-lengths``; default ``1`` for legacy seq1 tree):
  baseline_simple / baseline_warm / prepro_delta / all_baselines / all

Seq-length compare (after prior lock): ``--phase seq_length_compare --sequence-lengths 32 64 128``

See docs/cv4fold/ablations/ablation_chmm_incohort.md.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml

from scripts.cv4fold.locked_recipes import (
    PriorTier,
    apply_model_variant,
    load_locked_recipe,
)
from scripts.cv4fold.manifest_utils import all_mice, dataset_entries, load_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "src/config/run/cvaemarhmm/cv4fold"

HQ_LABS = ("lab_2", "lab_3", "lab_5")
DEFAULT_EPOCHS = 200
LAB_EPOCHS: dict[str, int] = {
    "lab_2": DEFAULT_EPOCHS,
    "lab_3": DEFAULT_EPOCHS,
    "lab_5": 300,
}

EMG_WIDE = [[None, 20.0], [None, 20.0], [3.0, 100.0]]

PREPRO_DELTAS: dict[str, dict[str, dict[str, Any]]] = {
    "lab_2": {
        "notch50": {"notch_freqs": [50.0]},
    },
    "lab_5": {
        "emg_wide": {"band_pass_freqs": copy.deepcopy(EMG_WIDE)},
        "emg_wide_notch50": {
            "band_pass_freqs": copy.deepcopy(EMG_WIDE),
            "notch_freqs": [50.0],
        },
    },
}


def _out_root(sequence_length: int) -> Path:
    if sequence_length == 1:
        return CONFIG_ROOT / "ablation_chmm"
    return CONFIG_ROOT / f"ablation_chmm_seq{sequence_length}"


def _variant_tag(base: str, sequence_length: int) -> str:
    if sequence_length == 1:
        return base
    return f"{base}_seq{sequence_length}"


def _mice_for_lab(manifest: dict, lab: str) -> list[str]:
    if lab not in manifest["cohort"]:
        raise KeyError(f"Unknown lab {lab!r}")
    return list(manifest["cohort"][lab])


def _assert_scratch(cfg: dict[str, Any]) -> None:
    cvae = cfg.setdefault("cvae", {})
    if cvae.get("model_checkpoint_path") not in (None, "null"):
        raise ValueError("cHMM incohort ablations require model_checkpoint_path: null")
    cvae["model_checkpoint_path"] = None
    cvae["save_pretrained_checkpoint"] = False


def _finalize_cfg(
    cfg: dict[str, Any],
    *,
    lab: str,
    variant: str,
    prior_tier: PriorTier,
    sequence_length: int,
) -> dict[str, Any]:
    out = apply_model_variant(cfg, "chmmgmvae", prior_tier=prior_tier, lab=lab)
    out.setdefault("trainer", {})["epochs"] = LAB_EPOCHS.get(lab, DEFAULT_EPOCHS)
    out.setdefault("dataloader", {})["sequence_length"] = sequence_length
    out.setdefault("model", {}).setdefault("params", {})
    out["model"]["params"].setdefault("no_beta_epochs", 10)
    out.setdefault("visualizer", {})["save_results_npz"] = False
    out["runs"] = 3
    out["seed"] = 123
    _assert_scratch(out)
    if sequence_length == 1:
        out["results_dir"] = f"results/cv4fold/ablation_chmm/{lab}"
    else:
        out["results_dir"] = f"results/cv4fold/ablation_chmm_seq{sequence_length}/{lab}"
    out["run_name"] = f"abl_chmm_{lab}_{variant}"
    return out


def _write_cfg(cfg: dict[str, Any], lab: str, filename: str, sequence_length: int) -> Path:
    out_dir = _out_root(sequence_length) / lab
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    with out_path.open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"Wrote {out_path}")
    return out_path


def _attach_datasets(cfg: dict[str, Any], manifest: dict, lab: str) -> dict[str, Any]:
    mice = _mice_for_lab(manifest, lab)
    allowed = set(all_mice(manifest))
    extra = set(mice) - allowed
    if extra:
        raise ValueError(f"Mice not in HQ cohort: {sorted(extra)}")
    entries = dataset_entries(manifest, mice)
    out = copy.deepcopy(cfg)
    out["train_datasets"] = copy.deepcopy(entries)
    out["val_datasets"] = copy.deepcopy(entries)
    return out


def generate_baseline(
    manifest: dict,
    labs: list[str],
    *,
    prior_tier: PriorTier,
    variant_base: str,
    sequence_length: int,
) -> list[Path]:
    written: list[Path] = []
    variant = _variant_tag(variant_base, sequence_length)
    for lab in labs:
        base = load_locked_recipe(lab)
        base = _attach_datasets(base, manifest, lab)
        cfg = _finalize_cfg(
            base,
            lab=lab,
            variant=variant,
            prior_tier=prior_tier,
            sequence_length=sequence_length,
        )
        written.append(_write_cfg(cfg, lab, f"{variant}.yaml", sequence_length))
    return written


def generate_prepro_delta(
    manifest: dict,
    labs: list[str],
    *,
    prior_tiers: list[PriorTier],
    sequence_length: int,
) -> list[Path]:
    written: list[Path] = []
    for lab in labs:
        if lab not in PREPRO_DELTAS:
            print(f"Skip {lab}: no prepro_delta variants defined")
            continue
        base = load_locked_recipe(lab)
        base = _attach_datasets(base, manifest, lab)
        for delta_name, cvae_overrides in PREPRO_DELTAS[lab].items():
            for tier in prior_tiers:
                cfg = copy.deepcopy(base)
                cfg.setdefault("cvae", {})
                for key, val in cvae_overrides.items():
                    cfg["cvae"][key] = copy.deepcopy(val)
                variant = _variant_tag(f"{delta_name}_{tier}", sequence_length)
                cfg = _finalize_cfg(
                    cfg,
                    lab=lab,
                    variant=variant,
                    prior_tier=tier,
                    sequence_length=sequence_length,
                )
                written.append(_write_cfg(cfg, lab, f"{variant}.yaml", sequence_length))
    return written


def generate_all(manifest: dict, labs: list[str], *, sequence_length: int) -> None:
    generate_baseline(
        manifest,
        labs,
        prior_tier="simple",
        variant_base="hmm_gmm_locked",
        sequence_length=sequence_length,
    )
    generate_baseline(
        manifest,
        labs,
        prior_tier="warm",
        variant_base="warm_hmm_gmm_locked",
        sequence_length=sequence_length,
    )
    generate_prepro_delta(
        manifest,
        ["lab_2", "lab_5"],
        prior_tiers=["simple", "warm"],
        sequence_length=sequence_length,
    )


def generate_seq_length_compare(
    manifest: dict,
    labs: list[str],
    *,
    sequence_lengths: list[int],
    prior_tier: PriorTier,
) -> None:
    """Locked winner recipe at each T — pick best sequence_length (no prepro matrix)."""
    variant_base = "hmm_gmm_locked" if prior_tier == "simple" else "warm_hmm_gmm_locked"
    for seq_len in sequence_lengths:
        if seq_len == 1:
            continue
        generate_baseline(
            manifest,
            labs,
            prior_tier=prior_tier,
            variant_base=variant_base,
            sequence_length=seq_len,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=(
            "baseline_simple",
            "baseline_warm",
            "prepro_delta",
            "all_baselines",
            "all",
            "seq_length_compare",
        ),
        required=True,
    )
    parser.add_argument(
        "--labs",
        nargs="+",
        default=list(HQ_LABS),
        help=f"HQ labs (default: all). Choices: {HQ_LABS}",
    )
    parser.add_argument(
        "--sequence-lengths",
        nargs="+",
        type=int,
        default=[1],
        help="Dataloader sequence_length per config tree (default: 1).",
    )
    parser.add_argument(
        "--prior-tier",
        choices=("simple", "warm", "both"),
        default="both",
        help="For prepro_delta: which prior tier(s) to generate.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to cv_quality_cohort_v1.yaml",
    )
    args = parser.parse_args()

    if any(s < 1 for s in args.sequence_lengths):
        raise ValueError("sequence_lengths must be >= 1")

    manifest = load_manifest(args.manifest)
    for lab in args.labs:
        if lab not in HQ_LABS:
            raise ValueError(f"Invalid lab {lab!r}; must be one of {HQ_LABS}")

    if args.phase == "seq_length_compare":
        tier: PriorTier = "simple" if args.prior_tier == "both" else args.prior_tier  # type: ignore[assignment]
        generate_seq_length_compare(
            manifest,
            args.labs,
            sequence_lengths=args.sequence_lengths,
            prior_tier=tier,
        )
        return 0

    for seq_len in args.sequence_lengths:
        if args.phase == "baseline_simple":
            generate_baseline(
                manifest,
                args.labs,
                prior_tier="simple",
                variant_base="hmm_gmm_locked",
                sequence_length=seq_len,
            )
        elif args.phase == "baseline_warm":
            generate_baseline(
                manifest,
                args.labs,
                prior_tier="warm",
                variant_base="warm_hmm_gmm_locked",
                sequence_length=seq_len,
            )
        elif args.phase == "all_baselines":
            generate_baseline(
                manifest,
                args.labs,
                prior_tier="simple",
                variant_base="hmm_gmm_locked",
                sequence_length=seq_len,
            )
            generate_baseline(
                manifest,
                args.labs,
                prior_tier="warm",
                variant_base="warm_hmm_gmm_locked",
                sequence_length=seq_len,
            )
        elif args.phase == "prepro_delta":
            tiers: list[PriorTier] = (
                ["simple", "warm"] if args.prior_tier == "both" else [args.prior_tier]  # type: ignore[list-item]
            )
            generate_prepro_delta(manifest, args.labs, prior_tiers=tiers, sequence_length=seq_len)
        elif args.phase == "all":
            generate_all(manifest, list(HQ_LABS), sequence_length=seq_len)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
