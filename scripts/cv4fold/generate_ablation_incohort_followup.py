#!/usr/bin/env python3
"""Generate incohort follow-up cGMVAE ablations from analysis matrix (2026-06-08).

Variants:
  lab_5 — wide_mlp + emg_wide + notch50, 300 ep (complete 3-seed run)
  lab_2 — locked rem_emg_wide_eeg4 + wide_mlp
  lab_2 — locked recipe, 300 epochs
  lab_2 — locked recipe, validation_checkpoint: rem_recall

See docs/cv4fold/ablations/cgmvae_incohort_analysis_20260608.md.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "src/config/run/cvaemarhmm/cv4fold"
LAB2_BASE = CONFIG_ROOT / "ablation_rem/lab_2/rem_emg_wide_eeg4.yaml"
LAB5_BASE = CONFIG_ROOT / "ablation_arch/lab_5/wide_mlp.yaml"
OUT_ROOT = CONFIG_ROOT / "ablation_followup"

WIDE_MLP = {
    "enc_hidden_dims": [512, 256, 128],
    "dec_hidden_dims": [128, 256, 512],
}
EMG_WIDE = [[None, 20.0], [None, 20.0], [3.0, 100.0]]
LAB5_EPOCHS = 300
LAB2_EPOCHS_LONG = 300


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write(cfg: dict, rel_path: Path) -> None:
    rel_path.parent.mkdir(parents=True, exist_ok=True)
    with rel_path.open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"Wrote {rel_path}")


def _lab2_base() -> dict:
    cfg = copy.deepcopy(_load(LAB2_BASE))
    cfg["results_dir"] = "results/cv4fold/ablation_followup/lab_2"
    cfg["runs"] = 3
    cfg["seed"] = 123
    cfg.setdefault("cvae", {})
    cfg["cvae"]["model_checkpoint_path"] = None
    cfg["cvae"]["save_pretrained_checkpoint"] = False
    return cfg


def main() -> int:
    if not LAB2_BASE.is_file():
        raise SystemExit(f"Missing {LAB2_BASE}")
    if not LAB5_BASE.is_file():
        raise SystemExit(f"Missing {LAB5_BASE}")

    # P1 — lab_5 emg_wide + notch50, 300 ep
    lab5 = copy.deepcopy(_load(LAB5_BASE))
    lab5.setdefault("trainer", {})["epochs"] = LAB5_EPOCHS
    lab5.setdefault("cvae", {})
    lab5["cvae"]["band_pass_freqs"] = copy.deepcopy(EMG_WIDE)
    lab5["cvae"]["notch_freqs"] = [50.0]
    lab5["cvae"]["model_checkpoint_path"] = None
    lab5["cvae"]["save_pretrained_checkpoint"] = False
    lab5["results_dir"] = "results/cv4fold/ablation_followup/lab_5"
    lab5["run_name"] = "abl_followup_lab_5_emg_wide_notch50"
    lab5["runs"] = 3
    lab5["seed"] = 123
    _write(lab5, OUT_ROOT / "lab_5/emg_wide_notch50.yaml")

    # P2 — lab_2 locked + wide_mlp
    lab2_mlp = _lab2_base()
    for key, val in WIDE_MLP.items():
        lab2_mlp["model"]["params"][key] = copy.deepcopy(val)
    lab2_mlp["run_name"] = "abl_followup_lab_2_wide_mlp"
    _write(lab2_mlp, OUT_ROOT / "lab_2/wide_mlp.yaml")

    # P3 — lab_2 locked, 300 epochs
    lab2_long = _lab2_base()
    lab2_long["trainer"]["epochs"] = LAB2_EPOCHS_LONG
    lab2_long["run_name"] = "abl_followup_lab_2_epochs300"
    _write(lab2_long, OUT_ROOT / "lab_2/epochs300.yaml")

    # P4 — lab_2 locked, validate with best REM-recall checkpoint
    lab2_recall = _lab2_base()
    lab2_recall.setdefault("trainer", {})["validation_checkpoint"] = "rem_recall"
    lab2_recall["run_name"] = "abl_followup_lab_2_rem_recall_ckpt"
    _write(lab2_recall, OUT_ROOT / "lab_2/rem_recall_ckpt.yaml")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
