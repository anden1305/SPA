#!/usr/bin/env python3
"""Generate lab_2 REM-focused ablation configs (VAE encoder unchanged).

All variants keep the conv cGMVAE as the feature learner (thesis: VAE latent
beats parallel expert-feature pipelines). Only preprocessing / VAE *input*
knobs change: bandpass, paper robust scaling, optional per-channel RMS bin
appended to the FFT tensor before the encoder.

Base template: locked winner ``rem_emg_wide_eeg4`` (NMI **0.593**, EEG1+EEG4+EMG).
See docs/cv4fold/ablations/ablation_lab2_rem.md.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WINNER = REPO_ROOT / "src/config/run/cvaemarhmm/cv4fold/ablation_rem/lab_2/rem_emg_wide_eeg4.yaml"
OUT_ROOT = REPO_ROOT / "src/config/run/cvaemarhmm/cv4fold/ablation_rem/lab_2"

MONTAGE_SIGNALS = {
    "eeg3": ["EEG1", "EEG3", "EMG"],
    "eeg4": ["EEG1", "EEG4", "EMG"],
}

LONG_EPOCHS = 200
WIDE_BP = [[None, 30.0], [None, 30.0], [5.0, 60.0]]
EMG_WIDE = [[None, 30.0], [None, 30.0], [3.0, 100.0]]
EMG_LOW = [[None, 30.0], [None, 30.0], [1.0, 40.0]]
PAPER_BP = [[0.3, 35.0], [0.3, 35.0], [5.0, 60.0]]
PAPER_BP_EMG_WIDE = [[0.3, 35.0], [0.3, 35.0], [3.0, 100.0]]

# variant -> cvae overrides (epochs applied to all).
VARIANTS: dict[str, dict] = {
    # Baseline re-run under ablation_rem/ for apples-to-apples comparison.
    "rem_winner": {},
    "rem_paper_robust": {
        "robust_normalize": True,
        "band_pass_freqs": PAPER_BP,
    },
    "rem_emg_wide": {
        "band_pass_freqs": EMG_WIDE,
    },
    "rem_emg_low": {
        "band_pass_freqs": EMG_LOW,
    },
    "rem_paper_robust_emg_wide": {
        "robust_normalize": True,
        "band_pass_freqs": PAPER_BP_EMG_WIDE,
    },
    # Append one log-RMS bin per channel to FFT input (still through conv VAE).
    "rem_append_rms": {
        "append_channel_rms": True,
    },
    "rem_winner_append_rms": {
        "append_channel_rms": True,
    },
    "rem_paper_robust_append_rms": {
        "robust_normalize": True,
        "band_pass_freqs": PAPER_BP,
        "append_channel_rms": True,
    },
}


def _load_winner() -> dict:
    with WINNER.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _apply_signals(cfg: dict, signals: list[str]) -> None:
    for block in ("train_datasets", "val_datasets"):
        for ds in cfg.get(block, []):
            ds["signals"] = copy.deepcopy(signals)


def _write(base: dict, variant: str, montage: str) -> Path:
    cfg = copy.deepcopy(base)
    signals = MONTAGE_SIGNALS[montage]
    _apply_signals(cfg, signals)
    cfg["trainer"]["epochs"] = LONG_EPOCHS
    cfg["cvae"].setdefault("robust_normalize", False)
    cfg["cvae"].setdefault("append_channel_rms", False)
    for key, val in VARIANTS[variant].items():
        cfg["cvae"][key] = copy.deepcopy(val)
    cfg["results_dir"] = "results/cv4fold/ablation_rem/lab_2"
    suffix = f"_{montage}" if montage != "eeg3" else ""
    cfg["run_name"] = f"abl_lab_2_{variant}{suffix}"
    out_dir = OUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{variant}{suffix}.yaml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"Wrote {out_path}")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--montage",
        choices=("eeg3", "eeg4"),
        default="eeg4",
        help="Signal montage (default EEG1+EEG4+EMG; eeg3 legacy EEG1+EEG3+EMG).",
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        default=None,
        help="Subset of variant keys (default: all).",
    )
    args = parser.parse_args()
    if not WINNER.is_file():
        raise SystemExit(f"Missing winner template: {WINNER}")
    base = _load_winner()
    variants = args.variants if args.variants else list(VARIANTS)
    for variant in variants:
        if variant not in VARIANTS:
            raise SystemExit(f"Unknown variant: {variant}")
        _write(base, variant, args.montage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
