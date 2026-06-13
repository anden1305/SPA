#!/usr/bin/env python3
"""Retry configs for population K-sweep (OOM recovery + seed top-up).

Writes to paper_k_sweep/population_retry/ — same results_dir as population/
so metrics merge across run directories.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

from scripts.cv4fold.generate_chmm_k_sweep_population import (
    POPULATION_SEED,
    build_population_chmm_base,
    _patch_k,
    _patch_warm_emb8_sticky92,
    _write_cfg,
)
from scripts.cv4fold.generate_joint_locked_holdout import LOCKED_LR
from scripts.cv4fold.generate_chmm_k_sweep_population import _patch_lr
from scripts.cv4fold.manifest_utils import load_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "src/config/run/cvaemarhmm/cv4fold/paper_k_sweep/population_retry"

# Failed on v100 OOM — full 5-seed retry
FULL_K = list(range(3, 12)) + [13, 14]
# Partial completion — additional seeds only
TOPUP_K: dict[int, int] = {12: 2, 15: 1}


def _base_cfg(manifest: dict) -> dict:
    cfg = _patch_lr(_patch_warm_emb8_sticky92(build_population_chmm_base(manifest)), LOCKED_LR)
    cfg["seed"] = POPULATION_SEED
    cfg.setdefault("visualizer", {})["save_results_npz"] = True
    # Smaller batches — full 20-mouse cohort OOMs at 128 on V100 (15 GB)
    cfg.setdefault("dataloader", {})["batch_size"] = 64
    cfg["dataloader"]["validation_batch_size"] = 64
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    manifest = load_manifest()
    base = _base_cfg(manifest)
    written: list[Path] = []

    for k in FULL_K:
        cfg = copy.deepcopy(_patch_k(base, k))
        cfg["runs"] = 5
        cfg["results_dir"] = f"results/cv4fold/paper_k_sweep/population/K{k}"
        cfg["run_name"] = f"population_k_sweep_K{k}"
        path = _write_cfg(CONFIG_ROOT / f"chmmgmvae_K{k}.yaml", cfg)
        written.append(path)
        print(f"Wrote {path} | K={k} runs=5 batch=64")

    for k, runs in TOPUP_K.items():
        cfg = copy.deepcopy(_patch_k(base, k))
        cfg["runs"] = runs
        cfg["results_dir"] = f"results/cv4fold/paper_k_sweep/population/K{k}"
        cfg["run_name"] = f"population_k_sweep_K{k}"
        path = _write_cfg(CONFIG_ROOT / f"chmmgmvae_K{k}.yaml", cfg)
        written.append(path)
        print(f"Wrote {path} | K={k} runs={runs} (top-up) batch=64")

    print(f"Generated {len(written)} retry configs → {CONFIG_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
