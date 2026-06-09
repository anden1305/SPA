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

# Paper main line: one unified recipe per model family (within-lab + joint scopes).
PAPER_MODELS = ("cgmvae", "hmmgmvae", "chmmgmvae")
UNIFIED_LATENT_DIM = 6
UNIFIED_EMB_DIM = 4
UNIFIED_JOINT_HMM_SEQUENCE_LENGTH = 64

CVAE_OVERRIDE_KEYS = (
    "normalize_global",
    "pre_normalize",
    "post_normalize",
    "robust_normalize",
    "band_pass_freqs",
    "notch_freqs",
    "append_channel_rms",
    "percentile_clip_channels",
    "perform_hanning_window",
    "band_pass_filter_fft",
    "band_pass_filter_type",
)

UNIFIED_WIDE_MLP_PARAMS: dict[str, Any] = {
    "latent_dim": UNIFIED_LATENT_DIM,
    "enc_hidden_dims": [512, 256, 128],
    "dec_hidden_dims": [128, 256, 512],
    "conv_channels": [32, 64, 128, 128],
    "kernel_sizes": [7, 5, 5, 3],
    "strides": [2, 2, 2, 2],
    "paddings": [3, 2, 2, 1],
    "lags": [1, 2, 4],
    "ridge": 0.1,
    "var_reg": 0.05,
    "sticky_coef": 0.1,
    "sticky_kappa": 0.9,
    "decoder_only_conditioning": True,
    "use_rms": False,
    "min_beta": 0.01,
    "max_beta": 1.0,
    "beta_warmup_epochs": 0,
    "beta_slowdown_epochs": 0,
    "gmm_warmup_epochs": 0,
    "free_nats_per_dim": 0.0,
    "no_beta_epochs": 10,
}

UNIFIED_DATALOADER: dict[str, Any] = {
    "num_batches": 64,
    "batch_size": 128,
    "validation_batch_size": 128,
    "window_size": 512,
    "stride": 512,
    "shuffle": True,
    "normalize": True,
    "use_legacy": False,
    "transforms": [],
}

UNIFIED_TRAINER_BASE: dict[str, Any] = {
    "epochs": 200,
    "optimizer": "adam",
    "grad_clip": 0.5,
    "validate_per_epoch": 1,
    "early_stopping": {
        "enabled": False,
        "patience": 100,
        "min_delta": 0.001,
    },
    "scheduler": {
        "enabled": False,
        "type": "exponential",
        "step_size": None,
        "gamma": 0.99,
    },
}

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
    elif model in ("chmmgmvae", "hmmgmvae"):
        tier = resolve_chmm_prior_tier(lab, prior_tier)
        apply_chmm_prior_tier(params, tier)
    else:
        raise ValueError(f"Unknown model {model!r}; use cgmvae, hmmgmvae, or chmmgmvae")
    return out


def apply_locked_sequence_length(
    cfg: dict[str, Any],
    model: str,
    lab: str,
) -> dict[str, Any]:
    """Set ``dataloader.sequence_length`` from per-lab locks (cHMM never seq=1)."""
    out = copy.deepcopy(cfg)
    dl = out.setdefault("dataloader", {})
    if model in ("chmmgmvae", "hmmgmvae"):
        if lab is not None and lab in LOCKED_CHMM_SEQUENCE_LENGTH:
            dl["sequence_length"] = LOCKED_CHMM_SEQUENCE_LENGTH[lab]
        else:
            dl["sequence_length"] = UNIFIED_JOINT_HMM_SEQUENCE_LENGTH
    else:
        dl["sequence_length"] = LOCKED_CGMVAE_SEQUENCE_LENGTH
    return out


def extract_lab_cvae_overrides(lab: str) -> dict[str, Any]:
    """Per-lab front-end from incohort lock (full cvae block minus checkpoint paths)."""
    if lab not in LOCKED_SOURCES:
        raise KeyError(f"Unknown lab {lab!r}")
    prepro = _load_yaml(str(LOCKED_SOURCES[lab]["prepro"]))
    cvae = copy.deepcopy(prepro.get("cvae", {}))
    for key in ("model_checkpoint_path", "save_pretrained_checkpoint", "traning_pipeline"):
        cvae.pop(key, None)
    return {k: cvae[k] for k in CVAE_OVERRIDE_KEYS if k in cvae}


def _deep_merge_cvae(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, val in override.items():
        out[key] = copy.deepcopy(val)
    return out


def apply_unified_sequence_length(cfg: dict[str, Any], model: str) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    dl = out.setdefault("dataloader", {})
    if model == "cgmvae":
        dl["sequence_length"] = LOCKED_CGMVAE_SEQUENCE_LENGTH
    else:
        dl["sequence_length"] = UNIFIED_JOINT_HMM_SEQUENCE_LENGTH
    return out


def apply_unified_recipe(cfg: dict[str, Any], model: str) -> dict[str, Any]:
    """Apply paper-line unified hyperparams for one model family."""
    out = copy.deepcopy(cfg)
    out.setdefault("trainer", {}).update(copy.deepcopy(UNIFIED_TRAINER_BASE))
    out["dataloader"] = copy.deepcopy(UNIFIED_DATALOADER)
    out = apply_unified_sequence_length(out, model)

    params = out.setdefault("model", {}).setdefault("params", {})
    params.update(copy.deepcopy(UNIFIED_WIDE_MLP_PARAMS))
    params["latent_dim"] = UNIFIED_LATENT_DIM

    if model == "cgmvae":
        out["trainer"]["learning_rate"] = 3e-4
        params["emb_dim"] = UNIFIED_EMB_DIM
        params["prior"] = "gmm"
        params.setdefault("num_gmm_states", 3)
        for key in (
            *CHMM_WARMUP_KEYS,
            "hmm_sticky_kappa",
            "hmm_estimate_transitions",
        ):
            params.pop(key, None)
    elif model == "chmmgmvae":
        out["trainer"]["learning_rate"] = 0.0013
        params["emb_dim"] = UNIFIED_EMB_DIM
        apply_chmm_prior_tier(params, "simple")
    elif model == "hmmgmvae":
        out["trainer"]["learning_rate"] = 0.0013
        params["emb_dim"] = 0
        apply_chmm_prior_tier(params, "simple")
    else:
        raise ValueError(f"Unknown paper model {model!r}")

    out.setdefault("cvae", {})
    out["cvae"].setdefault("normalize_global", False)
    out["cvae"]["model_checkpoint_path"] = None
    out["cvae"]["save_pretrained_checkpoint"] = False
    out["cvae"]["traning_pipeline"] = "cvae"
    out.setdefault("model", {}).setdefault("params", {})["decoder_only_conditioning"] = True
    out["runs"] = 3
    out["seed"] = 123
    out.setdefault("visualizer", {})["save_results_npz"] = True
    return ensure_cv4fold_diagnostics(out)


def build_unified_holdout_skeleton(model: str) -> dict[str, Any]:
    """Scratch config skeleton for paper-line holdout (no datasets). Base from lab_3 wide_mlp."""
    base = load_locked_recipe("lab_3")
    return apply_unified_recipe(base, model)
