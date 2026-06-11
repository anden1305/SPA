#!/usr/bin/env python3
"""Generate supplemental K-sweep YAMLs: 2 extra seeds only (no rerun of seeds 1–3).

Existing jobs used runs=3, seed=123 → RNG seeds 124, 125, 126 (plots/1..3).
Extra jobs use runs=2, seed=126 → RNG seeds 127, 128 (plots/1..2 in a new run dir).

Analysis merges metrics across all run dirs under each K folder.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "src/config/run/cvaemarhmm/cv4fold/paper_k_sweep"
EXTRA_RUNS = 2
# train_cvae does seed += 1 before each run; after 3-run job (123→126), next runs are 127, 128
EXTRA_START_SEED = 126


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, default=4)
    parser.add_argument("--k-min", type=int, default=3)
    parser.add_argument("--k-max", type=int, default=15)
    args = parser.parse_args()

    written: list[Path] = []
    for k in range(args.k_min, args.k_max + 1):
        src = CONFIG_ROOT / f"fold_{args.fold}" / f"chmmgmvae_K{k}.yaml"
        if not src.is_file():
            print(f"Skip K={k}: missing {src}")
            continue
        with src.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg = copy.deepcopy(cfg)
        cfg["runs"] = EXTRA_RUNS
        cfg["seed"] = EXTRA_START_SEED
        cfg["run_name"] = f"joint_k_sweep_f{args.fold}_K{k}_extra2"
        out = CONFIG_ROOT / f"fold_{args.fold}" / f"chmmgmvae_K{k}_extra2.yaml"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        written.append(out)
        print(f"Wrote {out} | K={k} runs={EXTRA_RUNS} seed={EXTRA_START_SEED}")

    print(f"\n{len(written)} extra-seed configs (seeds 127–128 per K).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
