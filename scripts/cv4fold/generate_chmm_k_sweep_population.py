#!/usr/bin/env python3
"""K substage sweep on all 20 HQ mice (incohort train + val) for appendix / biology panels.

Locked cHMM–GMVAE (warm_emb8_sticky92), 5 seeds per K, K=3…15.
Separate from fold-4 holdout sweep — see docs/paper/k_sweep_population.md
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
)
from scripts.cv4fold.generate_joint_locked_holdout import LOCKED_LR
from scripts.cv4fold.locked_recipes import (
    build_unified_holdout_skeleton,
    extract_lab_cvae_overrides,
)
from scripts.cv4fold.manifest_utils import (
    all_mice,
    dataset_entries_with_lab_prepro,
    load_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "src/config/run/cvaemarhmm/cv4fold"
DEFAULT_K = tuple(range(3, 16))
POPULATION_RUNS = 5
POPULATION_SEED = 499


def _patch_warm_emb8_sticky92(cfg: dict[str, Any]) -> dict[str, Any]:
    return _patch_sticky(_patch_emb(_patch_warm_prior(cfg), 8), 0.92)


def _patch_k(cfg: dict[str, Any], k: int) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    out.setdefault("model", {}).setdefault("params", {})["num_gmm_states"] = k
    return out


def build_population_chmm_base(manifest: dict) -> dict[str, Any]:
    """All 20 mice in both train and val (incohort substage discovery)."""
    mice = all_mice(manifest)
    entries = dataset_entries_with_lab_prepro(
        manifest, mice, extract_cvae_overrides=extract_lab_cvae_overrides
    )
    cfg = build_unified_holdout_skeleton("chmmgmvae")
    cfg["train_datasets"] = copy.deepcopy(entries)
    cfg["val_datasets"] = copy.deepcopy(entries)
    return cfg


def _write_cfg(path: Path, cfg: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return path


def generate(manifest: dict, *, k_values: list[int]) -> list[Path]:
    written: list[Path] = []
    base = _patch_lr(_patch_warm_emb8_sticky92(build_population_chmm_base(manifest)), LOCKED_LR)
    base["runs"] = POPULATION_RUNS
    base["seed"] = POPULATION_SEED
    base.setdefault("visualizer", {})["save_results_npz"] = True

    for k in k_values:
        cfg = _patch_k(base, k)
        cfg["results_dir"] = f"results/cv4fold/paper_k_sweep/population/K{k}"
        cfg["run_name"] = f"population_k_sweep_K{k}"
        out_path = _write_cfg(
            CONFIG_ROOT / "paper_k_sweep" / "population" / f"chmmgmvae_K{k}.yaml",
            cfg,
        )
        written.append(out_path)
        print(f"Wrote {out_path} | K={k} mice=20 runs={POPULATION_RUNS} seed={POPULATION_SEED}")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, nargs="*", default=list(DEFAULT_K))
    args = parser.parse_args()
    manifest = load_manifest()
    paths = generate(manifest, k_values=args.k)
    print(f"Generated {len(paths)} population K-sweep configs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
