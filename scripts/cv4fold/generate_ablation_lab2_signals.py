#!/usr/bin/env python3
"""Generate lab_2 signal-montage ablation vs no_postnorm_widebp winner.

Compares EEG1+EEG3+EMG (current HQ montage) with EEG1+EEG4+EMG (P + second
frontal; EEG4 slightly cleaner on sub-072). Same prepro/arch as prepro winner.

See docs/cv4fold/ablation_lab2_signals.md
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WINNER = REPO_ROOT / "src/config/run/cvaemarhmm/cv4fold/ablation_prepro/lab_2/no_postnorm_widebp.yaml"
OUT_ROOT = REPO_ROOT / "src/config/run/cvaemarhmm/cv4fold/ablation_signals/lab_2"

SIGNALS = {
    "baseline_eeg1_eeg3": ["EEG1", "EEG3", "EMG"],
    "eeg1_eeg4": ["EEG1", "EEG4", "EMG"],
}


def _load_winner() -> dict:
    with WINNER.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _apply_signals(cfg: dict, signals: list[str]) -> None:
    for block in ("train_datasets", "val_datasets"):
        for ds in cfg.get(block, []):
            ds["signals"] = copy.deepcopy(signals)


def _write(base: dict, variant: str, signals: list[str]) -> Path:
    cfg = copy.deepcopy(base)
    _apply_signals(cfg, signals)
    cfg["results_dir"] = "results/cv4fold/ablation_signals/lab_2"
    cfg["run_name"] = f"abl_lab_2_{variant}"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = OUT_ROOT / f"{variant}.yaml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"Wrote {out_path}  signals={signals}")
    return out_path


def main() -> int:
    if not WINNER.is_file():
        raise SystemExit(f"Missing winner template: {WINNER}")
    base = _load_winner()
    for variant, signals in SIGNALS.items():
        _write(base, variant, signals)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
