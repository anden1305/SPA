#!/usr/bin/env python3
"""Joint holdout cHMM-GMVAE with emb8 winner recipe (fold-4 ablation lock).

See docs/cv4fold/ablations/ablation_joint_chmm_fold4.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from scripts.cv4fold.generate_ablation_joint_chmm_fold4 import (
    _patch_emb,
    build_joint_chmm_base,
)
from scripts.cv4fold.manifest_utils import load_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "src/config/run/cvaemarhmm/cv4fold"
DEFAULT_FOLDS = (1, 2, 3)
WINNER_ID = "emb8"


def generate(manifest: dict, *, folds: list[int]) -> list[Path]:
    written: list[Path] = []
    for fold in folds:
        base = build_joint_chmm_base(manifest, fold=fold)
        cfg = _patch_emb(base, 8)
        cfg["results_dir"] = f"results/cv4fold/joint_holdout/fold_{fold}/chmmgmvae_emb8"
        cfg["run_name"] = f"joint_ho_f{fold}_chmmgmvae_emb8"
        cfg["runs"] = 3
        cfg["seed"] = 123

        out_dir = CONFIG_ROOT / "joint_holdout" / f"fold_{fold}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "chmmgmvae_emb8.yaml"
        with out_path.open("w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        written.append(out_path)
        print(f"Wrote {out_path} ({WINNER_ID}: latent=6, emb=8, T=64, hmm_gmm)")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fold",
        nargs="+",
        type=int,
        default=list(DEFAULT_FOLDS),
        help="Holdout folds (default: 1 2 3).",
    )
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    generate(manifest, folds=args.fold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
