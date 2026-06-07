"""Per-lab locked training recipes from ablation winners (best-of-3 incohort)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "src/config/run/cvaemarhmm/cv4fold"

LAB2_SIGNALS = ["EEG1", "EEG4", "EMG"]

# prepro YAML holds cvae + default model; optional arch YAML overlays model.params (+ enc/dec dims).
LOCKED_SOURCES: dict[str, dict[str, str | None]] = {
    "lab_2": {
        "prepro": "ablation_rem/lab_2/rem_emg_wide_eeg4.yaml",
        "arch": None,
        "note": "EEG1+EEG4+EMG, no postnorm, EEG 0–30 Hz, EMG 3–100 Hz",
    },
    "lab_3": {
        "prepro": "ablation_prepro/lab_3/baseline_long.yaml",
        "arch": "ablation_arch/lab_3/wide_mlp.yaml",
        "note": "postnorm, EEG 0–20 Hz, wide_mlp",
    },
    "lab_5": {
        "prepro": "ablation_prepro/lab_5/long.yaml",
        "arch": "ablation_arch/lab_5/wide_mlp.yaml",
        "note": "postnorm, EEG 0–20 Hz, 200 ep, wide_mlp",
    },
}

CHMMGMVAE_PRIOR_OVERRIDES: dict[str, Any] = {
    "prior": "warm_hmm_gmm",
    "num_gmm_states": 3,
    "gmm_warmup_epochs": 18,
    "hmm_warmup_epochs": 37,
    "hmm_transition_ramp_epochs": 18,
    "hmm_sticky_kappa": 0.86,
    "hmm_estimate_transitions": True,
}

HOLDOUT_KEYS = ("trainer", "validator", "visualizer", "dataloader", "cvae", "model", "verbose", "validate_data")


def _load_yaml(rel_path: str) -> dict[str, Any]:
    path = CONFIG_ROOT / rel_path
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse to a dict")
    return data


def load_locked_recipe(lab: str) -> dict[str, Any]:
    """Return a config skeleton (no datasets / run paths) for one HQ lab."""
    if lab not in LOCKED_SOURCES:
        raise KeyError(f"Unknown lab {lab!r}; expected one of {sorted(LOCKED_SOURCES)}")

    src = LOCKED_SOURCES[lab]
    base = _load_yaml(str(src["prepro"]))
    cfg = {k: copy.deepcopy(base[k]) for k in HOLDOUT_KEYS if k in base}

    arch_rel = src.get("arch")
    if arch_rel:
        arch = _load_yaml(str(arch_rel))
        if "model" in arch:
            cfg["model"] = copy.deepcopy(arch["model"])

    cfg.setdefault("cvae", {})
    cfg["cvae"]["model_checkpoint_path"] = None
    cfg["cvae"]["save_pretrained_checkpoint"] = False
    cfg["cvae"]["traning_pipeline"] = "cvae"
    cfg.setdefault("model", {}).setdefault("params", {})
    cfg["model"]["params"]["decoder_only_conditioning"] = True
    cfg.setdefault("visualizer", {})["save_results_npz"] = False
    cfg["runs"] = 3
    cfg["seed"] = 123
    return cfg


def apply_model_variant(cfg: dict[str, Any], model: str) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    params = out.setdefault("model", {}).setdefault("params", {})
    if model == "cgmvae":
        params["prior"] = "gmm"
        params.setdefault("num_gmm_states", 3)
        for key in (
            "hmm_warmup_epochs",
            "hmm_transition_ramp_epochs",
            "hmm_estimate_transitions",
        ):
            params.pop(key, None)
    elif model == "chmmgmvae":
        params.update(CHMMGMVAE_PRIOR_OVERRIDES)
    else:
        raise ValueError(f"Unknown model {model!r}; use cgmvae or chmmgmvae")
    return out
