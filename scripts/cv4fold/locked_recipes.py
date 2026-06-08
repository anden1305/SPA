"""Per-lab locked training recipes from ablation winners (best-of-3 incohort)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Literal

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "src/config/run/cvaemarhmm/cv4fold"

PriorTier = Literal["simple", "warm"]

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

# Locked cHMM prior tier per lab after incohort ablation (update when Phase 1/2 completes).
# Default ``simple`` (``hmm_gmm``) unless warm clearly beats simple on best-of-3 NMI.
LOCKED_CHMM_PRIOR_TIER: dict[str, PriorTier] = {
    "lab_2": "simple",
    "lab_3": "warm",
    "lab_5": "simple",
}

# Locked dataloader.sequence_length for cHMM-GMVAE (incohort seq compare 2026-06-08).
# Use best-of-3 cHMM NMI per lab (may be below cGMVAE — holdout still runs for comparison).
LOCKED_CHMM_SEQUENCE_LENGTH: dict[str, int] = {
    "lab_2": 64,  # best T>1 cHMM (0.546); T=1 inactive HMM — use temporal seq for comparison
    "lab_3": 64,
    "lab_5": 32,
}

LOCKED_CGMVAE_SEQUENCE_LENGTH: int = 1

CHMMGMVAE_SIMPLE_OVERRIDES: dict[str, Any] = {
    "prior": "hmm_gmm",
    "num_gmm_states": 3,
    "hmm_sticky_kappa": 0.86,
    "hmm_estimate_transitions": True,
}

CHMMGMVAE_WARM_PRIOR_OVERRIDES: dict[str, Any] = {
    "prior": "warm_hmm_gmm",
    "num_gmm_states": 3,
    "gmm_warmup_epochs": 18,
    "hmm_warmup_epochs": 37,
    "hmm_transition_ramp_epochs": 18,
    "hmm_sticky_kappa": 0.86,
    "hmm_estimate_transitions": True,
}

# Back-compat alias (holdout docs referenced this name for warm schedule).
CHMMGMVAE_PRIOR_OVERRIDES = CHMMGMVAE_WARM_PRIOR_OVERRIDES

CHMM_WARMUP_KEYS = (
    "gmm_warmup_epochs",
    "hmm_warmup_epochs",
    "hmm_transition_ramp_epochs",
)

HOLDOUT_KEYS = ("trainer", "validator", "visualizer", "dataloader", "cvae", "model", "verbose", "validate_data")

CV4FOLD_DIAGNOSTIC_DEFAULTS: dict[str, Any] = {
    "validate_data": True,
    "validator": {
        "state_distinctness": True,
        "summary_statistics": True,
    },
    "visualizer": {
        "state_distinctness": True,
        "summary_statistics": True,
        "losses": True,
        "pca_tripanel": True,
    },
}


def ensure_cv4fold_diagnostics(cfg: dict[str, Any]) -> dict[str, Any]:
    """Ensure ablation / holdout configs emit separability and training-loss plots."""
    out = copy.deepcopy(cfg)
    out["validate_data"] = True
    for section, defaults in CV4FOLD_DIAGNOSTIC_DEFAULTS.items():
        if section == "validate_data":
            continue
        if not isinstance(defaults, dict):
            out[section] = defaults
            continue
        target = out.setdefault(section, {})
        if isinstance(target, dict):
            for key, value in defaults.items():
                target.setdefault(key, value)
    return out


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
    return ensure_cv4fold_diagnostics(cfg)


def resolve_chmm_prior_tier(lab: str | None, prior_tier: PriorTier | None) -> PriorTier:
    if prior_tier is not None:
        return prior_tier
    if lab is not None and lab in LOCKED_CHMM_PRIOR_TIER:
        return LOCKED_CHMM_PRIOR_TIER[lab]
    return "simple"


def apply_chmm_prior_tier(params: dict[str, Any], prior_tier: PriorTier) -> None:
    for key in CHMM_WARMUP_KEYS:
        params.pop(key, None)
    if prior_tier == "simple":
        params.update(copy.deepcopy(CHMMGMVAE_SIMPLE_OVERRIDES))
    elif prior_tier == "warm":
        params.update(copy.deepcopy(CHMMGMVAE_WARM_PRIOR_OVERRIDES))
    else:
        raise ValueError(f"Unknown prior_tier {prior_tier!r}; use 'simple' or 'warm'")


def apply_model_variant(
    cfg: dict[str, Any],
    model: str,
    *,
    prior_tier: PriorTier | None = None,
    lab: str | None = None,
) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    params = out.setdefault("model", {}).setdefault("params", {})
    if model == "cgmvae":
        params["prior"] = "gmm"
        params.setdefault("num_gmm_states", 3)
        for key in (
            *CHMM_WARMUP_KEYS,
            "hmm_sticky_kappa",
            "hmm_estimate_transitions",
        ):
            params.pop(key, None)
    elif model == "chmmgmvae":
        tier = resolve_chmm_prior_tier(lab, prior_tier)
        apply_chmm_prior_tier(params, tier)
    else:
        raise ValueError(f"Unknown model {model!r}; use cgmvae or chmmgmvae")
    return out


def apply_locked_sequence_length(
    cfg: dict[str, Any],
    model: str,
    lab: str,
) -> dict[str, Any]:
    """Set ``dataloader.sequence_length`` from per-lab locks (cHMM never seq=1)."""
    out = copy.deepcopy(cfg)
    dl = out.setdefault("dataloader", {})
    if model == "chmmgmvae":
        if lab not in LOCKED_CHMM_SEQUENCE_LENGTH:
            raise ValueError(
                f"No locked cHMM sequence_length for {lab!r}. "
                "Add an entry to LOCKED_CHMM_SEQUENCE_LENGTH in locked_recipes.py."
            )
        dl["sequence_length"] = LOCKED_CHMM_SEQUENCE_LENGTH[lab]
    else:
        dl["sequence_length"] = LOCKED_CGMVAE_SEQUENCE_LENGTH
    return out
