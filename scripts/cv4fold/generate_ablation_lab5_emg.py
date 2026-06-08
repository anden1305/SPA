#!/usr/bin/env python3
"""Generate lab_5 EMG-wide (+ optional 50 Hz notch) ablations on locked ``wide_mlp``.

Base: ``ablation_arch/lab_5/wide_mlp.yaml`` (postnorm, EEG 0–20 Hz, best **0.534**).
Hypothesis: wide EMG band (3–100 Hz) helps REM like lab_2; notch may fix sub-089 spike.

See docs/cv4fold/lab_preprocessing_review_20260607.md.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = REPO_ROOT / "src/config/run/cvaemarhmm/cv4fold/ablation_arch/lab_5/wide_mlp.yaml"
OUT_ROOT = REPO_ROOT / "src/config/run/cvaemarhmm/cv4fold/ablation_emg/lab_5"

EMG_WIDE = [[None, 20.0], [None, 20.0], [3.0, 100.0]]
TRAINER_EPOCHS = 300  # lab_5 prior NMI peaks late vs 200 ep locked wide_mlp

VARIANTS: dict[str, dict] = {
    "wide_mlp_notch50": {
        "notch_freqs": [50.0],
    },
    "wide_mlp_emg_wide": {
        "band_pass_freqs": EMG_WIDE,
    },
    "wide_mlp_emg_wide_notch50": {
        "band_pass_freqs": EMG_WIDE,
        "notch_freqs": [50.0],
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants",
        nargs="*",
        default=None,
        help="Subset of variant keys (default: all).",
    )
    args = parser.parse_args()
    if not BASE.is_file():
        raise SystemExit(f"Missing base config: {BASE}")

    with BASE.open(encoding="utf-8") as f:
        template = yaml.safe_load(f)

    keys = args.variants if args.variants else list(VARIANTS)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for variant in keys:
        if variant not in VARIANTS:
            raise SystemExit(f"Unknown variant: {variant}")
        cfg = copy.deepcopy(template)
        cfg.setdefault("trainer", {})["epochs"] = TRAINER_EPOCHS
        cfg.setdefault("cvae", {})
        for key, val in VARIANTS[variant].items():
            cfg["cvae"][key] = copy.deepcopy(val)
        cfg["results_dir"] = "results/cv4fold/ablation_emg/lab_5"
        cfg["run_name"] = f"abl_lab_5_{variant}"
        out_path = OUT_ROOT / f"{variant}.yaml"
        with out_path.open("w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
