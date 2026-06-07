#!/usr/bin/env python3
"""Generate structured preprocessing/epochs ablation configs (lab_2, lab_3, lab_5).

Bases are the per-lab in-cohort configs (correct HQ datasets + signals + null
checkpoint). Each variant tweaks only the knob under test so differences are
attributable. See docs/cv4fold/ablations/ablation_prepro_lab2_lab5.md.

Round 1 (done): lab_2 {baseline_long, no_postnorm, widebp, no_postnorm_widebp},
                lab_5 {long}.
Round 2: lab_5 {no_postnorm, no_postnorm_widebp},
         lab_3 {baseline_long, no_postnorm, no_postnorm_widebp}.
Round 4: lab_3/5 {bp25, no_postnorm_bp25} — EEG 0.3-25 Hz (vs locked 0-20).
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PER_LAB = REPO_ROOT / "src/config/run/cvaemarhmm/cv4fold/per_lab_incohort"
OUT_ROOT = REPO_ROOT / "src/config/run/cvaemarhmm/cv4fold/ablation_prepro"

LONG_EPOCHS = 200
# EEG channels widened from 20 Hz to 30 Hz lowpass to retain Awake/REM
# discriminating power; EMG (3rd channel) kept at 5-60 Hz.
WIDE_BP = [[None, 30.0], [None, 30.0], [5.0, 60.0]]
# Paper (Rose et al. 2025) EEG band 0.3-35 Hz; EMG kept at 5-60 Hz for our
# spectral model (paper's single 0.3-35 band would drop EMG high-freq tone).
PAPER_BP = [[0.3, 35.0], [0.3, 35.0], [5.0, 60.0]]
# Mid band between locked 0-20 Hz (lab_3/5) and lab_2 winner 0-30 Hz; EEG high-pass
# at 0.3 Hz (paper-style) instead of null lowpass-only.
BP25 = [[0.3, 25.0], [0.3, 25.0], [5.0, 60.0]]

# variant -> cvae knob overrides (epochs=LONG_EPOCHS applied to all).
VARIANTS = {
    "long": {},
    "baseline_long": {},
    "no_postnorm": {"post_normalize": False},
    "widebp": {"band_pass_freqs": WIDE_BP},
    "no_postnorm_widebp": {"post_normalize": False, "band_pass_freqs": WIDE_BP},
    # Round 3: paper-style lab-agnostic front-end (median/IQR robust scaling +
    # 0.3-35 Hz EEG band), post/pre-norm off.
    "paper_robust": {
        "robust_normalize": True,
        "post_normalize": False,
        "pre_normalize": False,
        "band_pass_freqs": PAPER_BP,
    },
    "bp25": {"band_pass_freqs": BP25},
    "no_postnorm_bp25": {"post_normalize": False, "band_pass_freqs": BP25},
}

# Per-lab variant sets (covers round 1 + round 2 + round 3; regeneration is idempotent).
LAB_VARIANTS = {
    "lab_2": ["baseline_long", "no_postnorm", "widebp", "no_postnorm_widebp", "paper_robust"],
    "lab_3": [
        "baseline_long",
        "no_postnorm",
        "no_postnorm_widebp",
        "paper_robust",
        "bp25",
        "no_postnorm_bp25",
    ],
    "lab_5": [
        "long",
        "no_postnorm",
        "no_postnorm_widebp",
        "paper_robust",
        "bp25",
        "no_postnorm_bp25",
    ],
}


def _load_base(lab: str) -> dict:
    with (PER_LAB / lab / "cgmvae.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write(base: dict, lab: str, variant: str) -> Path:
    cfg = copy.deepcopy(base)
    cfg["trainer"]["epochs"] = LONG_EPOCHS
    for key, val in VARIANTS[variant].items():
        cfg["cvae"][key] = copy.deepcopy(val)
    cfg["results_dir"] = f"results/cv4fold/ablation_prepro/{lab}"
    cfg["run_name"] = f"abl_{lab}_{variant}"
    out_dir = OUT_ROOT / lab
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{variant}.yaml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"Wrote {out_path}")
    return out_path


def main() -> int:
    for lab, variants in LAB_VARIANTS.items():
        base = _load_base(lab)
        for variant in variants:
            _write(base, lab, variant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
