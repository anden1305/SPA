#!/usr/bin/env python3
"""Generate lab_2 50 Hz notch ablation on locked winner ``rem_emg_wide_eeg4``.

Adds ``cvae.notch_freqs: [50]`` (zeros 49.5–50.5 Hz in rFFT after band-pass).
Control best NMI without notch: **0.593** (job 28607461).

See docs/cv4fold/lab_preprocessing_review_20260607.md.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WINNER = REPO_ROOT / "src/config/run/cvaemarhmm/cv4fold/ablation_rem/lab_2/rem_emg_wide_eeg4.yaml"
OUT_ROOT = REPO_ROOT / "src/config/run/cvaemarhmm/cv4fold/ablation_rem/lab_2"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        default="rem_emg_wide_eeg4_notch50",
        help="Output YAML stem (default: rem_emg_wide_eeg4_notch50).",
    )
    args = parser.parse_args()
    if not WINNER.is_file():
        raise SystemExit(f"Missing winner template: {WINNER}")

    with WINNER.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg = copy.deepcopy(cfg)
    cfg.setdefault("cvae", {})
    cfg["cvae"]["notch_freqs"] = [50.0]
    cfg["results_dir"] = "results/cv4fold/ablation_rem/lab_2"
    cfg["run_name"] = f"abl_lab_2_{args.variant}"

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = OUT_ROOT / f"{args.variant}.yaml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
