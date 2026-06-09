#!/usr/bin/env python3
"""Joint fold-4 LR swap: cHMM lock at 3e-4, cGMVAE emb8 at 1.3e-3.

Sensitivity check for fair-comparison claims (see ablation_joint_chmm_fold4.md).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from scripts.cv4fold.generate_ablation_joint_chmm_fold4 import (
    _patch_emb,
    _patch_lr,
    _patch_sticky,
    _patch_warm_prior,
    build_joint_chmm_base,
)
from scripts.cv4fold.generate_joint_cgmvae_emb8_holdout import build_joint_cgmvae_base
from scripts.cv4fold.manifest_utils import load_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "src/config/run/cvaemarhmm/cv4fold"
CHMM_LR_SWAP = 3e-4
CGMVAE_LR_SWAP = 0.0013
FOLD = 4


def _patch_warm_emb8_sticky92(cfg: dict[str, Any]) -> dict[str, Any]:
    out = _patch_sticky(_patch_emb(_patch_warm_prior(cfg), 8), 0.92)
    return out


def generate(manifest: dict, *, fold: int = FOLD) -> list[Path]:
    written: list[Path] = []

    chmm_base = build_joint_chmm_base(manifest, fold=fold)
    chmm_cfg = _patch_lr(_patch_warm_emb8_sticky92(chmm_base), CHMM_LR_SWAP)
    chmm_cfg["results_dir"] = (
        f"results/cv4fold/ablation_joint_chmm/fold_{fold}/warm_emb8_sticky92_lr3e4"
    )
    chmm_cfg["run_name"] = f"abl_joint_chmm_f{fold}_warm_emb8_sticky92_lr3e4"
    chmm_cfg["runs"] = 3
    chmm_cfg["seed"] = 123
    chmm_out = CONFIG_ROOT / "ablation_joint_chmm" / f"fold_{fold}" / "warm_emb8_sticky92_lr3e4.yaml"
    chmm_out.parent.mkdir(parents=True, exist_ok=True)
    with chmm_out.open("w", encoding="utf-8") as f:
        yaml.dump(chmm_cfg, f, default_flow_style=False, sort_keys=False)
    written.append(chmm_out)
    print(f"Wrote {chmm_out} (warm+emb8+sticky92+T=64, lr={CHMM_LR_SWAP})")

    cgmvae_base = build_joint_cgmvae_base(manifest, fold=fold)
    cgmvae_cfg = _patch_lr(_patch_emb(cgmvae_base, 8), CGMVAE_LR_SWAP)
    cgmvae_cfg["results_dir"] = (
        f"results/cv4fold/joint_holdout/fold_{fold}/cgmvae_emb8_lr1p3e3"
    )
    cgmvae_cfg["run_name"] = f"joint_ho_f{fold}_cgmvae_emb8_lr1p3e3"
    cgmvae_cfg["runs"] = 3
    cgmvae_cfg["seed"] = 123
    cgmvae_out = CONFIG_ROOT / "joint_holdout" / f"fold_{fold}" / "cgmvae_emb8_lr1p3e3.yaml"
    cgmvae_out.parent.mkdir(parents=True, exist_ok=True)
    with cgmvae_out.open("w", encoding="utf-8") as f:
        yaml.dump(cgmvae_cfg, f, default_flow_style=False, sort_keys=False)
    written.append(cgmvae_out)
    print(f"Wrote {cgmvae_out} (emb8+T=1+gmm, lr={CGMVAE_LR_SWAP})")

    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, default=FOLD)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    generate(manifest, fold=args.fold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
