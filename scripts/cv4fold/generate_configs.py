#!/usr/bin/env python3
"""Generate cv4fold training yaml from manifest + templates."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

from scripts.cv4fold.manifest_utils import (
    all_mice,
    dataset_entries,
    holdout_for_fold,
    load_manifest,
    train_mice,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

VAE_MODELS = ("cgmvae", "hmmgmvae", "chmmgmvae")
SWEEP_VAE_MODELS = ("cgmvae", "chmmgmvae")
SCOPES = (
    ("joint", None),
    ("per_lab", "lab_2"),
    ("per_lab", "lab_3"),
    ("per_lab", "lab_5"),
)

# W&B Bayesian tune: metrics go to wandb; avoid ~1.2 GiB results.npz per trial.
TUNE_SWEEP_MINIMAL = {
    "visualizer": {
        "losses": False,
        "learning_rate": False,
        "pca_tripanel": False,
        "confusion_matrix": False,
        "state_distinctness": False,
        "summary_statistics": False,
        "historic_values": False,
        "save_results_npz": False,
    },
    "validator": {
        "cross_nmi": False,
        "learning_rate": False,
        "state_distinctness": False,
        "summary_statistics": False,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _scope_label(scope: str, lab: str | None) -> str:
    return "joint" if scope == "joint" else f"per_lab/{lab}"


def _load_template(model: str) -> dict:
    if model == "hmm_raw":
        path = REPO_ROOT / "src/config/run/hmm/cv4fold/templates/hmm_raw_thesis_base.yaml"
    else:
        path = REPO_ROOT / f"src/config/run/cvaeprior/cv4fold/templates/{model}_base.yaml"
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _config_path(model: str, scope: str, lab: str | None, fold: int, phase: str) -> Path:
    if model == "hmm_raw":
        base = REPO_ROOT / "src/config/run/hmm/cv4fold"
    else:
        base = REPO_ROOT / "src/config/run/cvaeprior/cv4fold"

    if phase in ("smoke", "smoke_80"):
        scope_dir = _scope_label(scope, lab).replace("/", "_")
        fname = f"{model}_{scope_dir}_fold_{fold}.yaml"
        subdir = "smoke" if phase == "smoke" else "smoke80"
        return base / subdir / fname

    if phase == "tune":
        raise ValueError("tune paths built inline")

    if scope == "joint":
        return base / "joint" / f"fold_{fold}" / f"{model}.yaml"
    return base / "per_lab" / lab / f"fold_{fold}" / f"{model}.yaml"


SUBJECT_LAB_TUNE_MODELS = ("cgmvae", "chmmgmvae")


def _tune_winner_overrides(
    manifest: dict,
    model: str,
    fold: int,
    *,
    experiment: str,
    conditioning_source: str,
    sequence_length: int | None = None,
) -> dict:
    winners = manifest.get("tune_winners")
    if not winners or model not in winners:
        raise KeyError(f"tune_winners.{model} missing in manifest")
    raw = copy.deepcopy(winners[model])
    raw.pop("trial_id", None)
    raw["run_name"] = f"cv4fold_{model}_{experiment}_joint_fold{fold}"
    raw["results_dir"] = f"results/cv4fold/{experiment}/{model}/joint/fold_{fold}"
    if "model" not in raw:
        raw["model"] = {}
    raw["model"]["conditioning_source"] = conditioning_source
    raw["model"].setdefault("params", {})["decoder_only_conditioning"] = True
    if sequence_length is not None:
        raw.setdefault("dataloader", {})["sequence_length"] = sequence_length
    return raw


def _tune_winner_config_path(experiment: str, model: str, fold: int) -> Path:
    return (
        REPO_ROOT
        / "src/config/run/cvaeprior/cv4fold"
        / experiment
        / "joint"
        / f"fold_{fold}"
        / f"{model}.yaml"
    )


def _generate_tune_winner_experiment(
    manifest: dict,
    *,
    experiment: str,
    conditioning_source: str,
    folds: list[int] | None = None,
    sequence_length: int | None = None,
) -> list[Path]:
    fold_list = folds if folds is not None else [1, 2, 3, 4]
    written: list[Path] = []
    for model in SUBJECT_LAB_TUNE_MODELS:
        for fold in fold_list:
            overrides = _tune_winner_overrides(
                manifest,
                model,
                fold,
                experiment=experiment,
                conditioning_source=conditioning_source,
                sequence_length=sequence_length,
            )
            cfg = build_config(manifest, model, "joint", None, fold, overrides)
            path = _tune_winner_config_path(experiment, model, fold)
            write_config(path, cfg)
            written.append(path)
    return written


def generate_subject_lab_tune_winners(manifest: dict, folds: list[int] | None = None) -> list[Path]:
    return _generate_tune_winner_experiment(
        manifest,
        experiment="subject_lab_tune_winners",
        conditioning_source="subject_lab",
        folds=folds,
    )


def generate_subject_tune_winners(manifest: dict, folds: list[int] | None = None) -> list[Path]:
    """Same tune hyperparams as winners; subject-only conditioning (tune ablation)."""
    return _generate_tune_winner_experiment(
        manifest,
        experiment="subject_tune_winners",
        conditioning_source="subject",
        folds=folds,
    )


def generate_subject_tune_winners_seq1(manifest: dict, folds: list[int] | None = None) -> list[Path]:
    """Tune winners + subject conditioning; dataloader.sequence_length=1 (sanity vs seq 64)."""
    return _generate_tune_winner_experiment(
        manifest,
        experiment="subject_tune_winners_seq1",
        conditioning_source="subject",
        folds=folds,
        sequence_length=1,
    )


def generate_subject_lab_tune_winners_seq1(manifest: dict, folds: list[int] | None = None) -> list[Path]:
    """Tune winners + subject_lab conditioning; sequence_length=1."""
    return _generate_tune_winner_experiment(
        manifest,
        experiment="subject_lab_tune_winners_seq1",
        conditioning_source="subject_lab",
        folds=folds,
        sequence_length=1,
    )


# src/config/run/cvaemarhmm/final/cvae_final.yaml training recipe (encoder+decoder conditioning).
CGMVAE_CVAE_FINAL_OVERRIDES: dict = {
    "trainer": {
        "epochs": 80,
        "learning_rate": 0.0003,
        "validate_per_epoch": 1,
    },
    "dataloader": {
        "sequence_length": 1,
        "batch_size": 128,
        "validation_batch_size": 128,
        "num_batches": 64,
    },
    "model": {
        "conditioning_source": "subject",
        "params": {
            "latent_dim": 8,
            "emb_dim": 4,
            "decoder_only_conditioning": False,
            "prior": "gmm",
            "min_beta": 0.01,
            "max_beta": 1.0,
            "beta_warmup_epochs": 0,
            "beta_slowdown_epochs": 0,
            "gmm_warmup_epochs": 0,
            "free_nats_per_dim": 0.0,
            "no_beta_epochs": 10,
        },
    },
}


def generate_cgmvae_cvae_final(manifest: dict, folds: list[int] | None = None) -> list[Path]:
    """Fold CV with cvae_final.yaml hyperparams (cgmvae only); no pretrained checkpoint."""
    experiment = "cgmvae_cvae_final"
    fold_list = folds if folds is not None else [4]
    written: list[Path] = []
    for fold in fold_list:
        overrides = _deep_merge(
            copy.deepcopy(CGMVAE_CVAE_FINAL_OVERRIDES),
            {
                "run_name": f"cv4fold_cgmvae_{experiment}_joint_fold{fold}",
                "results_dir": f"results/cv4fold/{experiment}/cgmvae/joint/fold_{fold}",
            },
        )
        cfg = build_config(manifest, "cgmvae", "joint", None, fold, overrides)
        path = _tune_winner_config_path(experiment, "cgmvae", fold)
        write_config(path, cfg)
        written.append(path)
    return written


def build_config(
    manifest: dict,
    model: str,
    scope: str,
    lab: str | None,
    fold: int,
    overrides: dict | None = None,
) -> dict:
    cfg = _load_template(model)
    train_ids = train_mice(manifest, fold, scope, lab)
    val_ids = holdout_for_fold(manifest, fold, scope, lab)
    scope_label = _scope_label(scope, lab).replace("/", "_")

    cfg["train_datasets"] = dataset_entries(manifest, train_ids)
    cfg["val_datasets"] = dataset_entries(manifest, val_ids)
    cfg["run_name"] = f"cv4fold_{model}_{scope_label}_fold{fold}"
    cfg["results_dir"] = f"results/cv4fold/{model}/{_scope_label(scope, lab)}/fold_{fold}"

    hp = manifest.get("hyperparams", {})
    if "learning_rate" in hp:
        cfg.setdefault("trainer", {})["learning_rate"] = hp["learning_rate"]
    if "epochs" in hp and model != "hmm_raw":
        cfg.setdefault("trainer", {})["epochs"] = hp["epochs"]

    if model == "hmm_raw":
        cfg["runs"] = manifest["runs_policy"]["hmm_raw"]["runs"]
        cfg["trainer"]["validate_per_epoch"] = manifest["trainer"]["hmm_raw_validate_per_epoch"]
    else:
        cfg["runs"] = manifest["runs_policy"]["vae"]["runs"]
        cfg["trainer"]["validate_per_epoch"] = manifest["trainer"]["vae_validate_per_epoch"]

    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return cfg


def write_config(path: Path, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


SWEEP_TAG = "bayes50"
SWEEP_MIN_EPOCHS = 80

# Hand-tuned cHMM-GMVAE configs for linuxsh interactive runs (runs=1, wandb off).
INTERACTIVE_CHMM_VARIANTS: dict[str, dict] = {
    "stable_joint_fold4": {
        "scope": "joint",
        "lab": None,
        "fold": 4,
        "overrides": {
            "runs": 1,
            "wandb": {"enabled": False},
            "trainer": {"epochs": 120, "learning_rate": 0.0003, "validate_per_epoch": 10, "grad_clip": 0.5},
            "model": {
                "params": {
                    "gmm_warmup_epochs": 10,
                    "hmm_warmup_epochs": 35,
                    "hmm_transition_ramp_epochs": 16,
                    "hmm_sticky_kappa": 0.92,
                    "latent_dim": 8,
                    "no_beta_epochs": 10,
                    "beta_schedule": "anneal",
                }
            },
            "run_name": "cv4fold_interactive_chmmgmvae_stable_joint_fold4",
            "results_dir": "results/cv4fold/interactive/chmmgmvae/stable_joint/fold_4",
        },
    },
    "tune_winner_joint_fold4": {
        "scope": "joint",
        "lab": None,
        "fold": 4,
        "overrides": {
            "runs": 1,
            "wandb": {"enabled": False},
            "trainer": {"epochs": 120, "learning_rate": 0.0003, "validate_per_epoch": 10, "grad_clip": 0.5},
            "run_name": "cv4fold_interactive_chmmgmvae_tune_winner_joint_fold4",
            "results_dir": "results/cv4fold/interactive/chmmgmvae/tune_winner_joint/fold_4",
        },
    },
    "quick_joint_fold4": {
        "scope": "joint",
        "lab": None,
        "fold": 4,
        "overrides": {
            "runs": 1,
            "wandb": {"enabled": False},
            "trainer": {"epochs": 40, "learning_rate": 0.0003, "validate_per_epoch": 5, "grad_clip": 0.5},
            "model": {
                "params": {
                    "gmm_warmup_epochs": 8,
                    "hmm_warmup_epochs": 28,
                    "hmm_transition_ramp_epochs": 12,
                }
            },
            "run_name": "cv4fold_interactive_chmmgmvae_quick_joint_fold4",
            "results_dir": "results/cv4fold/interactive/chmmgmvae/quick_joint/fold_4",
        },
    },
    "stable_per_lab3_fold4": {
        "scope": "per_lab",
        "lab": "lab_3",
        "fold": 4,
        "overrides": {
            "runs": 1,
            "wandb": {"enabled": False},
            "trainer": {"epochs": 120, "learning_rate": 0.0003, "validate_per_epoch": 10, "grad_clip": 0.5},
            "model": {
                "params": {
                    "gmm_warmup_epochs": 10,
                    "hmm_warmup_epochs": 35,
                    "hmm_transition_ramp_epochs": 16,
                    "hmm_sticky_kappa": 0.92,
                }
            },
            "run_name": "cv4fold_interactive_chmmgmvae_stable_per_lab3_fold4",
            "results_dir": "results/cv4fold/interactive/chmmgmvae/stable_per_lab3/fold_4",
        },
    },
}

# cGMVAE interactive configs (seq1 parity vs decoder-only reliability).
INTERACTIVE_CGMVAE_VARIANTS: dict[str, dict] = {
    "cgmvae_lab3_seq1_fold4": {
        "scope": "per_lab",
        "lab": "lab_3",
        "fold": 4,
        "overrides": {
            "runs": 1,
            "wandb": {"enabled": False},
            "trainer": {
                "epochs": 80,
                "learning_rate": 0.0003,
                "validate_per_epoch": 1,
                "grad_clip": 0.5,
            },
            "dataloader": {
                "sequence_length": 1,
                "batch_size": 128,
                "validation_batch_size": 128,
                "num_batches": 64,
            },
            "model": {
                "params": {
                    "prior": "gmm",
                    "latent_dim": 8,
                    "no_beta_epochs": 10,
                    "beta_schedule": "anneal",
                    "gmm_warmup_epochs": 0,
                }
            },
            "run_name": "cv4fold_interactive_cgmvae_lab3_seq1_fold4",
            "results_dir": "results/cv4fold/interactive/cgmvae/lab3_seq1/fold_4",
        },
    },
}


def _sweep_common_parameters() -> dict:
    return {
        "trainer.learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 0.002,
        },
        "trainer.epochs": {"min": SWEEP_MIN_EPOCHS, "max": 300},
        "trainer.grad_clip": {"distribution": "uniform", "min": 0.3, "max": 1.0},
        "trainer.validate_per_epoch": {"values": [5, 10]},
        "dataloader.num_batches": {"values": [32, 64, 128, 256]},
        "dataloader.batch_size": {"values": [32, 64]},
        "model.params.latent_dim": {"values": [4, 8, 16]},
        "model.params.emb_dim": {"values": [4, 8]},
        "model.params.beta_schedule": {"values": ["anneal", "cyclical"]},
        "model.params.no_beta_epochs": {"min": 0, "max": 30},
        "model.params.beta_warmup_epochs": {"min": 0, "max": 80},
        "model.params.beta_slowdown_epochs": {"values": [0, 10, 20, 40]},
        "model.params.beta_cyclical_period_epochs": {"min": 20, "max": 80},
        "model.params.min_beta": {
            "distribution": "log_uniform_values",
            "min": 0.001,
            "max": 0.05,
        },
        "model.params.ridge": {
            "distribution": "log_uniform_values",
            "min": 0.01,
            "max": 0.3,
        },
        "model.params.var_reg": {
            "distribution": "log_uniform_values",
            "min": 0.01,
            "max": 0.2,
        },
    }


def _sweep_parameters(model: str, *, all_mice: bool = False) -> dict:
    params = _sweep_common_parameters()
    if all_mice and model == "chmmgmvae":
        # All-mice cHMM-GMVAE on gpua100: match base (64); still cap num_batches vs latent_dim for VRAM.
        params["dataloader.num_batches"] = {"values": [32, 64, 128]}
        params["dataloader.batch_size"] = {"values": [32, 64, 128]}
        params["model.params.latent_dim"] = {"values": [4, 8]}
    params["model.params.gmm_warmup_epochs"] = {"min": 0, "max": 25}
    if model == "chmmgmvae":
        params["model.params.gmm_warmup_epochs"] = {"min": 5, "max": 25}
        params.update(
            {
                "model.params.hmm_warmup_epochs": {"min": 25, "max": 80},
                "model.params.hmm_transition_ramp_epochs": {"min": 8, "max": 32},
                "model.params.hmm_sticky_kappa": {
                    "distribution": "uniform",
                    "min": 0.85,
                    "max": 0.98,
                },
                "model.params.sticky_coef": {
                    "distribution": "uniform",
                    "min": 0.05,
                    "max": 0.2,
                },
            }
        )
    return params


def write_sweep_yaml(
    model: str,
    *,
    fold: int | None = None,
    all_mice: bool = False,
    lab: str | None = None,
) -> Path:
    sweep_dir = REPO_ROOT / "src/config/sweep/cv4fold"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    if all_mice:
        slug = "all_mice"
        base_config = (
            f"src/config/run/cvaeprior/cv4fold/tune/{model}_joint_all_mice_sweep_base.yaml"
        )
        comment = (
            f"# Bayesian tune for cv4fold {model} (joint, all mice, in-sample val). "
            f"50 trials, min {SWEEP_MIN_EPOCHS} ep.\n"
        )
    elif lab is not None:
        if fold is None:
            raise ValueError("fold required for per_lab sweep")
        slug = f"per_lab_{lab}_fold{fold}"
        base_config = (
            f"src/config/run/cvaeprior/cv4fold/tune/{model}_per_lab_{lab}_fold{fold}_sweep_base.yaml"
        )
        comment = (
            f"# Bayesian tune for cv4fold {model} (per_lab {lab} fold {fold}). "
            f"50 trials, min {SWEEP_MIN_EPOCHS} ep.\n"
        )
    else:
        if fold is None:
            raise ValueError("fold required when all_mice is False and lab is None")
        slug = f"fold{fold}"
        base_config = (
            f"src/config/run/cvaeprior/cv4fold/tune/{model}_joint_fold{fold}_sweep_base.yaml"
        )
        comment = (
            f"# Bayesian tune for cv4fold {model} (joint fold {fold}). "
            f"50 trials, min {SWEEP_MIN_EPOCHS} ep.\n"
        )
    if lab is not None:
        fname = f"{model}_{slug}_{SWEEP_TAG}.yaml"
    else:
        fname = f"{model}_joint_{slug}_{SWEEP_TAG}.yaml"
    path = sweep_dir / fname
    body = {
        "sweep_mode": "bayes",
        "method": "bayes",
        "count": 50,
        "metric": {"name": "val/prior_pred_nmi", "goal": "maximize"},
        "parameters": _sweep_parameters(model, all_mice=all_mice),
        "base_config": base_config,
    }
    with path.open("w", encoding="utf-8") as f:
        f.write(comment)
        yaml.safe_dump(body, f, sort_keys=False, allow_unicode=True)
    return path


def build_all_mice_sweep_config(manifest: dict, model: str, overrides: dict | None = None) -> dict:
    cfg = _load_template(model)
    mice = all_mice(manifest)
    cfg["train_datasets"] = dataset_entries(manifest, mice)
    cfg["val_datasets"] = dataset_entries(manifest, mice)
    cfg["run_name"] = f"cv4fold_tune_sweep_{model}_joint_all_mice"
    cfg["results_dir"] = f"results/cv4fold/tune_sweep/{model}/joint/all_mice"
    cfg["runs"] = manifest["runs_policy"]["vae"]["runs"]
    cfg.setdefault("trainer", {})["validate_per_epoch"] = manifest["trainer"]["vae_validate_per_epoch"]
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return cfg


def generate_all(
    manifest: dict,
    phase: str = "full",
    models: tuple[str, ...] | None = None,
    folds: list[int] | None = None,
    all_mice: bool = False,
    variants: tuple[str, ...] | None = None,
    per_lab: str | None = None,
) -> list[Path]:
    written: list[Path] = []
    model_list = models or (*VAE_MODELS, "hmm_raw")

    default_folds = [4] if phase in ("smoke", "smoke_80", "tune", "tune_sweep_base") else [1, 2, 3, 4]
    fold_list = folds if folds is not None else default_folds
    scopes = SCOPES if phase == "full" else (("joint", None),)
    if phase == "tune":
        scopes = (("joint", None),)

    smoke_overrides_vae = {
        "trainer": {"epochs": 20, "validate_per_epoch": 5},
        "runs": 1,
    }
    hmm_smoke_fast = {
        "trainer": {"validate_per_epoch": 10, "learning_rate": 0.005},
        "dataloader": {
            "num_batches": 200,
            "batch_size": 120,
            "max_batches_per_epoch": 1000,
        },
        "validator": {
            "nmi": True,
            "accuracy": True,
            "cross_nmi": False,
            "learning_rate": True,
            "state_distinctness": False,
            "summary_statistics": False,
            "log_likelihood": True,
            "validate_train": False,
        },
    }
    smoke_overrides_hmm = {
        **hmm_smoke_fast,
        "trainer": {**hmm_smoke_fast["trainer"], "epochs": 150},
        "runs": 1,
    }
    tune_lrs = [0.0002, 0.0003, 0.0005]
    tune_epochs = [80, 120]

    if phase == "subject_lab_tune_winners":
        return generate_subject_lab_tune_winners(manifest, folds=fold_list)

    if phase == "subject_tune_winners":
        return generate_subject_tune_winners(manifest, folds=fold_list)

    if phase == "subject_tune_winners_seq1":
        return generate_subject_tune_winners_seq1(manifest, folds=fold_list)

    if phase == "subject_lab_tune_winners_seq1":
        return generate_subject_lab_tune_winners_seq1(manifest, folds=fold_list)

    if phase == "cgmvae_cvae_final":
        return generate_cgmvae_cvae_final(manifest, folds=fold_list)

    if phase == "interactive":
        out_dir = REPO_ROOT / "src/config/run/cvaeprior/cv4fold/interactive"
        interactive_specs: list[tuple[str, str, dict]] = [
            *(("chmmgmvae", name, spec) for name, spec in INTERACTIVE_CHMM_VARIANTS.items()),
            *(("cgmvae", name, spec) for name, spec in INTERACTIVE_CGMVAE_VARIANTS.items()),
        ]
        for model, name, spec in interactive_specs:
            if variants and name not in variants:
                continue
            scope = spec["scope"]
            lab = spec.get("lab")
            fold = spec["fold"]
            cfg = build_config(manifest, model, scope, lab, fold, spec["overrides"])
            path = out_dir / f"{name}.yaml"
            write_config(path, cfg)
            written.append(path)
        return written

    if phase == "tune_sweep_base":
        if not all_mice:
            raise ValueError(
                "tune_sweep_base requires --all-mice (full-cohort in-sample tuning). "
                "Per-fold sweeps are not generated."
            )
        sweep_models = models if models else SWEEP_VAE_MODELS
        for sweep_model in sweep_models:
            if sweep_model not in SWEEP_VAE_MODELS:
                continue
            overrides = {
                "runs": 1,
                "trainer": {
                    "epochs": max(120, SWEEP_MIN_EPOCHS),
                    "learning_rate": 0.0003,
                    "validate_per_epoch": 10,
                },
                "wandb": {
                    "enabled": True,
                    "group": f"cv4fold_tune_sweep_{sweep_model}_all_mice",
                    "tags": ["cv4fold", "tune", sweep_model, "joint", "all_mice"],
                },
                **TUNE_SWEEP_MINIMAL,
            }
            path = (
                REPO_ROOT
                / f"src/config/run/cvaeprior/cv4fold/tune/{sweep_model}_joint_all_mice_sweep_base.yaml"
            )
            cfg = build_all_mice_sweep_config(manifest, sweep_model, overrides)
            write_config(path, cfg)
            written.append(path)
            written.append(write_sweep_yaml(sweep_model, all_mice=True))
        return written

    for model in model_list:
        for scope, lab in scopes:
            for fold in fold_list:
                overrides = None
                if phase == "smoke":
                    if model == "hmm_raw":
                        overrides = smoke_overrides_hmm
                    else:
                        overrides = smoke_overrides_vae
                if phase == "smoke_80":
                    scope_label = _scope_label(scope, lab).replace("/", "_")
                    if model == "hmm_raw":
                        overrides = {
                            **hmm_smoke_fast,
                            "trainer": {**hmm_smoke_fast["trainer"], "epochs": 80},
                            "runs": 1,
                            "run_name": f"cv4fold_hmm_raw_{scope_label}_fold{fold}_smoke80",
                            "results_dir": f"results/cv4fold/smoke80/hmm_raw/{_scope_label(scope, lab)}/fold_{fold}",
                        }
                    else:
                        overrides = {
                            "trainer": {"epochs": 80, "validate_per_epoch": 10},
                            "runs": 1,
                            "run_name": f"cv4fold_{model}_{scope_label}_fold{fold}_smoke80",
                            "results_dir": f"results/cv4fold/smoke80/{model}/{_scope_label(scope, lab)}/fold_{fold}",
                        }
                if phase == "tune":
                    if model != "chmmgmvae":
                        continue
                    for lr in tune_lrs:
                        for ep in tune_epochs:
                            overrides = {
                                "trainer": {"learning_rate": lr, "epochs": ep},
                                "runs": 1,
                                "run_name": f"cv4fold_tune_chmmgmvae_joint_fold{fold}_lr{lr}_ep{ep}",
                                "results_dir": f"results/cv4fold/tune/chmmgmvae/joint/fold_{fold}/lr{lr}_ep{ep}",
                            }
                            path = REPO_ROOT / (
                                f"src/config/run/cvaeprior/cv4fold/tune/"
                                f"chmmgmvae_joint_fold{fold}_lr{lr}_ep{ep}.yaml"
                            )
                            cfg = build_config(manifest, model, scope, lab, fold, overrides)
                            write_config(path, cfg)
                            written.append(path)
                    continue

                path = _config_path(model, scope, lab, fold, phase)
                cfg = build_config(manifest, model, scope, lab, fold, overrides)
                write_config(path, cfg)
                written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=(
            "full",
            "smoke",
            "smoke_80",
            "tune",
            "tune_sweep_base",
            "interactive",
            "subject_lab_tune_winners",
            "subject_tune_winners",
            "subject_tune_winners_seq1",
            "subject_lab_tune_winners_seq1",
            "cgmvae_cvae_final",
        ),
        default="full",
        help=(
            "full=64 yaml; smoke=20ep; smoke_80=80ep; interactive=linuxsh chmmgmvae tests; "
            "tune_sweep_base=W&B; subject_lab_tune_winners=8 yaml; "
            "subject_tune_winners=same hyperparams, subject conditioning; "
            "subject_*_seq1=tune winners with sequence_length=1; "
            "cgmvae_cvae_final=cvae_final.yaml recipe on joint CV folds"
        ),
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        choices=(1, 2, 3, 4),
        default=None,
        help="Folds to generate (ignored for tune_sweep_base; use --all-mice there)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=(*VAE_MODELS, "hmm_raw"),
        default=None,
        help="Limit generated models (tune_sweep_base: cgmvae and/or chmmgmvae)",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(INTERACTIVE_CHMM_VARIANTS.keys()) + tuple(INTERACTIVE_CGMVAE_VARIANTS.keys()),
        default=None,
        help="interactive phase: which test config(s) to write (default: all)",
    )
    parser.add_argument(
        "--lab",
        choices=("lab_2", "lab_3", "lab_5"),
        default=None,
        help="ignored for tune_sweep_base (requires --all-mice)",
    )
    parser.add_argument(
        "--all-mice",
        action="store_true",
        help="tune_sweep_base: train and validate on full cohort (in-sample tuning)",
    )
    args = parser.parse_args()

    if args.phase == "tune_sweep_base" and not args.all_mice:
        parser.error("tune_sweep_base requires --all-mice")
    if args.all_mice and args.lab:
        parser.error("--all-mice and --lab are mutually exclusive")

    manifest = load_manifest(args.manifest)
    model_filter = tuple(args.models) if args.models else None
    paths = generate_all(
        manifest,
        phase=args.phase,
        models=model_filter,
        folds=args.folds,
        all_mice=args.all_mice,
        variants=tuple(args.variants) if args.variants else None,
        per_lab=args.lab,
    )
    print(f"Wrote {len(paths)} configs (phase={args.phase})")


if __name__ == "__main__":
    main()
