#!/usr/bin/env python3
"""Generate VAE architecture ablations on per-lab best preprocessing (overnight sweep).

Each lab uses the preprocessing winner from rounds 1–2 (see
docs/cv4fold/overnight_experiments_20260606.md). Only model.params knobs change.

Variants: lat4, lat16, wide_mlp, wide_lat16 (baseline arch = latent 8, [256,128] MLP).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PER_LAB = REPO_ROOT / "src/config/run/cvaemarhmm/cv4fold/per_lab_incohort"
OUT_ROOT = REPO_ROOT / "src/config/run/cvaemarhmm/cv4fold/ablation_arch"

LONG_EPOCHS = 200
WIDE_BP = [[None, 30.0], [None, 30.0], [5.0, 60.0]]
STD_BP = [[None, 20.0], [None, 20.0], [5.0, 60.0]]

# Best prepro per lab (rounds 1–2 interim).
LAB_PREPRO: dict[str, dict[str, Any]] = {
    "lab_2": {"post_normalize": False, "band_pass_freqs": copy.deepcopy(WIDE_BP)},
    "lab_3": {"post_normalize": True, "band_pass_freqs": copy.deepcopy(STD_BP)},
    "lab_5": {"post_normalize": True, "band_pass_freqs": copy.deepcopy(STD_BP)},
}

ARCH_VARIANTS: dict[str, dict[str, Any]] = {
    "lat4": {"latent_dim": 4},
    "lat16": {"latent_dim": 16},
    "wide_mlp": {
        "enc_hidden_dims": [512, 256, 128],
        "dec_hidden_dims": [128, 256, 512],
    },
    "wide_lat16": {
        "latent_dim": 16,
        "enc_hidden_dims": [512, 256, 128],
        "dec_hidden_dims": [128, 256, 512],
    },
}

LABS = ("lab_2", "lab_3", "lab_5")


def _load_base(lab: str) -> dict:
    with (PER_LAB / lab / "cgmvae.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write(base: dict, lab: str, variant: str) -> Path:
    cfg = copy.deepcopy(base)
    cfg["trainer"]["epochs"] = LONG_EPOCHS
    for key, val in LAB_PREPRO[lab].items():
        cfg["cvae"][key] = copy.deepcopy(val)
    cfg["cvae"]["model_checkpoint_path"] = None
    for key, val in ARCH_VARIANTS[variant].items():
        cfg["model"]["params"][key] = copy.deepcopy(val)
    cfg["results_dir"] = f"results/cv4fold/ablation_arch/{lab}"
    cfg["run_name"] = f"arch_{lab}_{variant}"
    out_dir = OUT_ROOT / lab
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{variant}.yaml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"Wrote {out_path}")
    return out_path


def main() -> int:
    for lab in LABS:
        base = _load_base(lab)
        for variant in ARCH_VARIANTS:
            _write(base, lab, variant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
