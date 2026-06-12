#!/usr/bin/env python3
"""Spill-queue backup configs for population K-sweep (alternate base seeds)."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

from scripts.cv4fold.generate_chmm_k_sweep_population import (
    _patch_k,
    _patch_lr,
    _patch_warm_emb8_sticky92,
    _write_cfg,
    build_population_chmm_base,
)
from scripts.cv4fold.generate_chmm_k_sweep_population_retry import TOPUP_K
from scripts.cv4fold.generate_joint_locked_holdout import LOCKED_LR
from scripts.cv4fold.manifest_utils import load_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "src/config/run/cvaemarhmm/cv4fold/paper_k_sweep/population_spill"

SPILL_SEEDS = {
    "a100": 599,
    "a10": 699,
    "l40s": 799,
}


def _base_cfg(manifest: dict, *, seed: int) -> dict:
    cfg = _patch_lr(_patch_warm_emb8_sticky92(build_population_chmm_base(manifest)), LOCKED_LR)
    cfg["seed"] = seed
    cfg.setdefault("visualizer", {})["save_results_npz"] = True
    cfg.setdefault("dataloader", {})["batch_size"] = 96
    cfg["dataloader"]["validation_batch_size"] = 96
    return cfg


def _runs_for_k(k: int, *, full_backup: bool) -> int:
    if full_backup:
        return 5
    return TOPUP_K.get(k, 5)


def generate(
    manifest: dict,
    *,
    spill_label: str,
    k_values: list[int],
    full_backup: bool = True,
) -> list[Path]:
    if spill_label not in SPILL_SEEDS:
        raise ValueError(f"spill_label must be one of {list(SPILL_SEEDS)}")
    seed = SPILL_SEEDS[spill_label]
    base = _base_cfg(manifest, seed=seed)
    out_dir = CONFIG_ROOT / spill_label
    written: list[Path] = []

    for k in k_values:
        runs = _runs_for_k(k, full_backup=full_backup)
        cfg = copy.deepcopy(_patch_k(base, k))
        cfg["runs"] = runs
        cfg["results_dir"] = f"results/cv4fold/paper_k_sweep/population/K{k}"
        cfg["run_name"] = f"population_k_sweep_K{k}_spill_{spill_label}"
        path = _write_cfg(out_dir / f"chmmgmvae_K{k}.yaml", cfg)
        written.append(path)
        print(f"Wrote {path} | K={k} runs={runs} seed={seed} ({spill_label})")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spill", choices=list(SPILL_SEEDS), required=True)
    parser.add_argument("--k", type=int, nargs="+", default=list(range(7, 16)))
    parser.add_argument(
        "--topup-runs",
        action="store_true",
        help="Use top-up run counts for K12/K15 instead of full 5-seed backup",
    )
    args = parser.parse_args()
    manifest = load_manifest()
    paths = generate(
        manifest,
        spill_label=args.spill,
        k_values=args.k,
        full_backup=not args.topup_runs,
    )
    print(f"Generated {len(paths)} spill configs → {CONFIG_ROOT / args.spill}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
