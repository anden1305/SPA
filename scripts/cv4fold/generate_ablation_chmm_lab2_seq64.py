#!/usr/bin/env python3
"""lab_2 cHMM-GMVAE seq64 — pruned stability ablations (scratch, incohort).

Only variants with a clear hypothesis from incohort evidence or chmm Bayes sweep
(5bkceef8: latent_dim=4, emb_dim=8, lr≈1.3e-3).

Generate:
  PYTHONPATH=. python3 scripts/cv4fold/generate_ablation_chmm_lab2_seq64.py
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from scripts.cv4fold.generate_ablation_chmm_incohort import (
    _attach_datasets,
    _finalize_cfg,
    _write_cfg,
)
from scripts.cv4fold.locked_recipes import PriorTier, load_locked_recipe
from scripts.cv4fold.manifest_utils import load_manifest

LAB = "lab_2"
SEQUENCE_LENGTH = 64

# (yaml stem, prior tier, cvae overrides, model.params overrides, trainer overrides)
VariantSpec = tuple[str, PriorTier, dict[str, Any], dict[str, Any], dict[str, Any]]

# Kept: 3 jobs. Dropped (with reason) documented in ablation_chmm_lab2_seq64.md.
LAB2_SEQ64_VARIANTS: list[VariantSpec] = [
    (
        "sweep_chmm_winner_seq64",
        "simple",
        {},
        {"latent_dim": 4, "emb_dim": 8},
        {"learning_rate": 0.0013},
    ),
    (
        "notch50_warm_seq64",
        "warm",
        {"notch_freqs": [50.0]},
        {},
        {},
    ),
    (
        "beta_anneal_seq64",
        "simple",
        {},
        {"beta_warmup_epochs": 50, "no_beta_epochs": 10},
        {},
    ),
]


def generate(manifest: dict) -> list[Path]:
    written: list[Path] = []
    base = load_locked_recipe(LAB)
    base = _attach_datasets(base, manifest, LAB)

    for stem, prior_tier, cvae_overrides, model_overrides, trainer_overrides in LAB2_SEQ64_VARIANTS:
        cfg = copy.deepcopy(base)
        if cvae_overrides:
            cfg.setdefault("cvae", {})
            for key, val in cvae_overrides.items():
                cfg["cvae"][key] = copy.deepcopy(val)
        variant = stem.replace("_seq64", "")
        cfg = _finalize_cfg(
            cfg,
            lab=LAB,
            variant=variant,
            prior_tier=prior_tier,
            sequence_length=SEQUENCE_LENGTH,
        )
        if model_overrides:
            cfg.setdefault("model", {}).setdefault("params", {}).update(model_overrides)
        if trainer_overrides:
            cfg.setdefault("trainer", {}).update(trainer_overrides)
        cfg["run_name"] = f"abl_chmm_{LAB}_{variant}_seq64"
        written.append(_write_cfg(cfg, LAB, f"{stem}.yaml", SEQUENCE_LENGTH))
    return written


def main() -> int:
    manifest = load_manifest(None)
    paths = generate(manifest)
    print(f"Generated {len(paths)} lab_2 seq64 cHMM configs (pruned matrix).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
