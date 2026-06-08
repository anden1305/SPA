#!/usr/bin/env python3
"""Generate cGMVAE incohort configs: encoder+decoder conditioning vs locked decoder-only winners.

One job per HQ lab — same locked prepro/arch as incohort; only ``decoder_only_conditioning: false``.

See docs/cv4fold/ablations/ablation_enc_dec_conditioning.md.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

from scripts.cv4fold.locked_recipes import apply_model_variant, load_locked_recipe
from scripts.cv4fold.manifest_utils import all_mice, dataset_entries, load_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "src/config/run/cvaemarhmm/cv4fold"
OUT_ROOT = CONFIG_ROOT / "ablation_conditioning"

HQ_LABS = ("lab_2", "lab_3", "lab_5")
VARIANT = "enc_dec_locked"


def _mice_for_lab(manifest: dict, lab: str) -> list[str]:
    return list(manifest["cohort"][lab])


def _attach_datasets(cfg: dict, manifest: dict, lab: str) -> dict:
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


def generate(manifest: dict, labs: list[str]) -> list[Path]:
    written: list[Path] = []
    for lab in labs:
        cfg = load_locked_recipe(lab)
        cfg = apply_model_variant(cfg, "cgmvae")
        cfg = _attach_datasets(cfg, manifest, lab)
        cfg.setdefault("model", {}).setdefault("params", {})
        cfg["model"]["params"]["decoder_only_conditioning"] = False
        cfg["results_dir"] = f"results/cv4fold/ablation_conditioning/{lab}"
        cfg["run_name"] = f"abl_cgmvae_{lab}_{VARIANT}"
        cfg["runs"] = 3
        cfg["seed"] = 123

        out_dir = OUT_ROOT / lab
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{VARIANT}.yaml"
        with out_path.open("w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        written.append(out_path)
        print(f"Wrote {out_path}")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labs",
        nargs="+",
        default=list(HQ_LABS),
        choices=HQ_LABS,
        help="HQ labs (default: all three).",
    )
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    generate(manifest, args.labs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
