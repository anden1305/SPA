#!/usr/bin/env python3
"""K substage sweep on locked joint holdout cHMMGMVAE (paper Fig 3 gate).

Varies ``num_gmm_states``; fixed warm_emb8_sticky92, fold 4 by default.
See docs/paper/k_sweep_experiments.md
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
    build_joint_chmm_base,
)
from scripts.cv4fold.generate_joint_locked_holdout import LOCKED_LR

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "src/config/run/cvaemarhmm/cv4fold"
DEFAULT_K = tuple(range(3, 16))  # 3 … 15 inclusive
DEFAULT_FOLD = 4


def _patch_warm_emb8_sticky92(cfg: dict[str, Any]) -> dict[str, Any]:
    return _patch_sticky(_patch_emb(_patch_warm_prior(cfg), 8), 0.92)


def _patch_k(cfg: dict[str, Any], k: int) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    out.setdefault("model", {}).setdefault("params", {})["num_gmm_states"] = k
    return out


def _write_cfg(path: Path, cfg: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return path


def generate(
    manifest: dict,
    *,
    fold: int,
    k_values: list[int],
) -> list[Path]:
    written: list[Path] = []
    base = _patch_lr(_patch_warm_emb8_sticky92(build_joint_chmm_base(manifest, fold=fold)), LOCKED_LR)
    base["runs"] = 3
    base["seed"] = 123
    base.setdefault("visualizer", {})["save_results_npz"] = True

    for k in k_values:
        cfg = _patch_k(base, k)
        cfg["results_dir"] = f"results/cv4fold/paper_k_sweep/fold_{fold}/K{k}"
        cfg["run_name"] = f"joint_k_sweep_f{fold}_K{k}"
        out_path = _write_cfg(
            CONFIG_ROOT / "paper_k_sweep" / f"fold_{fold}" / f"chmmgmvae_K{k}.yaml",
            cfg,
        )
        written.append(out_path)
        print(f"Wrote {out_path} | K={k} fold={fold}")
    return written


def main() -> int:
    from scripts.cv4fold.manifest_utils import load_manifest

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, default=DEFAULT_FOLD)
    parser.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=list(DEFAULT_K),
        help="Mixture component counts (default: 3 5 7 9 11 13 15).",
    )
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    generate(manifest, fold=args.fold, k_values=args.k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
