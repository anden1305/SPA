#!/usr/bin/env python3
"""Generate joint holdout fold-4 cHMM-GMVAE ablations (paper-line base + targeted deltas).

See docs/cv4fold/ablations/ablation_joint_chmm_fold4.md.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Callable

import yaml

from scripts.cv4fold.locked_recipes import (
    apply_chmm_prior_tier,
    build_unified_holdout_skeleton,
    extract_lab_cvae_overrides,
)
from scripts.cv4fold.manifest_utils import (
    all_mice,
    dataset_entries_with_lab_prepro,
    holdout_for_fold,
    load_manifest,
    train_mice,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "src/config/run/cvaemarhmm/cv4fold"
OUT_ROOT = CONFIG_ROOT / "ablation_joint_chmm" / "fold_4"
FOLD = 4

PatchFn = Callable[[dict[str, Any]], dict[str, Any]]


def _assert_hq_mice(manifest: dict, mouse_ids: list[str]) -> None:
    allowed = set(all_mice(manifest))
    extra = set(mouse_ids) - allowed
    if extra:
        raise ValueError(f"Mice not in HQ cohort: {sorted(extra)}")


def build_joint_chmm_base(manifest: dict, *, fold: int = FOLD) -> dict[str, Any]:
    holdout = holdout_for_fold(manifest, fold, "joint")
    train = train_mice(manifest, fold, "joint")
    _assert_hq_mice(manifest, holdout + train)
    train_entries = dataset_entries_with_lab_prepro(
        manifest, train, extract_cvae_overrides=extract_lab_cvae_overrides
    )
    val_entries = dataset_entries_with_lab_prepro(
        manifest, holdout, extract_cvae_overrides=extract_lab_cvae_overrides
    )
    cfg = build_unified_holdout_skeleton("chmmgmvae")
    cfg["train_datasets"] = copy.deepcopy(train_entries)
    cfg["val_datasets"] = copy.deepcopy(val_entries)
    return cfg


def _patch_warm_prior(cfg: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    params = out.setdefault("model", {}).setdefault("params", {})
    apply_chmm_prior_tier(params, "warm")
    return out


def _patch_seq32(cfg: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    out.setdefault("dataloader", {})["sequence_length"] = 32
    return out


def _patch_lr(cfg: dict[str, Any], lr: float) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    out.setdefault("trainer", {})["learning_rate"] = lr
    return out


def _patch_sticky(cfg: dict[str, Any], kappa: float) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    out.setdefault("model", {}).setdefault("params", {})["hmm_sticky_kappa"] = kappa
    return out


def _patch_latent(cfg: dict[str, Any], dim: int) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    out.setdefault("model", {}).setdefault("params", {})["latent_dim"] = dim
    return out


def _patch_emb(cfg: dict[str, Any], dim: int) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    out.setdefault("model", {}).setdefault("params", {})["emb_dim"] = dim
    return out


def _patch_no_beta(cfg: dict[str, Any], epochs: int) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    out.setdefault("model", {}).setdefault("params", {})["no_beta_epochs"] = epochs
    return out


# Nine deltas aimed at lifting joint fold-4 best-of-3 above unified baseline (~0.41).
# Baseline already run: joint_ho_f4_chmmgmvae (simple hmm_gmm, T=64, lr 1.3e-3, latent 6).
ABLATIONS: list[tuple[str, PatchFn, str]] = [
    ("warm_prior", _patch_warm_prior, "warm_hmm_gmm (lab_3 incohort lock)"),
    ("seq32", _patch_seq32, "T=32 (lab_5 incohort lock; less HMM collapse risk)"),
    (
        "warm_seq32",
        lambda c: _patch_seq32(_patch_warm_prior(c)),
        "warm prior + T=32",
    ),
    ("lr8e4", lambda c: _patch_lr(c, 8e-4), "lower LR for stabler HMM transitions"),
    ("sticky92", lambda c: _patch_sticky(c, 0.92), "stickier HMM (anti single-state collapse)"),
    ("latent8", lambda c: _patch_latent(c, 8), "more latent capacity (cGMVAE hit 0.58 on same fold)"),
    ("no_beta20", lambda c: _patch_no_beta(c, 20), "longer reconstruction-first phase"),
    ("emb8", lambda c: _patch_emb(c, 8), "richer subject embedding for cross-lab holdout"),
    (
        "warm_latent8",
        lambda c: _patch_latent(_patch_warm_prior(c), 8),
        "warm prior + latent 8",
    ),
]

# Round 2 — combos from early winners (emb8 best 0.625; seq32 seed1 0.649).
ABLATIONS_ROUND2: list[tuple[str, PatchFn, str]] = [
    (
        "emb8_seq32",
        lambda c: _patch_emb(_patch_seq32(c), 8),
        "emb8 + T=32 (top two single-axis signals)",
    ),
    (
        "warm_emb8",
        lambda c: _patch_emb(_patch_warm_prior(c), 8),
        "warm prior + emb8",
    ),
    (
        "emb8_sticky92",
        lambda c: _patch_sticky(_patch_emb(c, 8), 0.92),
        "emb8 + stickier HMM",
    ),
    (
        "warm_emb8_seq32",
        lambda c: _patch_seq32(_patch_emb(_patch_warm_prior(c), 8)),
        "warm + emb8 + T=32",
    ),
]

# Round 3 — joint cHMM lock candidate.
ABLATIONS_ROUND3: list[tuple[str, PatchFn, str]] = [
    (
        "warm_emb8_sticky92",
        lambda c: _patch_sticky(_patch_emb(_patch_warm_prior(c), 8), 0.92),
        "warm + emb8 + sticky92 + T=64",
    ),
]

ALL_ABLATIONS: dict[str, list[tuple[str, PatchFn, str]]] = {
    "1": ABLATIONS,
    "2": ABLATIONS_ROUND2,
    "3": ABLATIONS_ROUND3,
}


def _resolve_rows(
    *,
    round_id: str | None,
    ablation_ids: list[str] | None,
) -> list[tuple[str, PatchFn, str]]:
    if round_id is not None:
        if round_id not in ALL_ABLATIONS:
            raise ValueError(f"Unknown round {round_id!r}; use 1, 2, or 3")
        pool = ALL_ABLATIONS[round_id]
    else:
        pool = ABLATIONS + ABLATIONS_ROUND2

    selected = {aid for aid, _, _ in pool}
    if ablation_ids:
        unknown = set(ablation_ids) - selected
        if unknown:
            raise ValueError(f"Unknown ablation id(s): {sorted(unknown)}")
        return [(aid, fn, note) for aid, fn, note in pool if aid in ablation_ids]
    return list(pool)


def generate(
    manifest: dict,
    *,
    round_id: str | None = None,
    ablation_ids: list[str] | None = None,
) -> list[Path]:
    base = build_joint_chmm_base(manifest)
    written: list[Path] = []
    rows = _resolve_rows(round_id=round_id, ablation_ids=ablation_ids)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for abl_id, patch_fn, note in rows:
        cfg = patch_fn(base)
        cfg["results_dir"] = f"results/cv4fold/ablation_joint_chmm/fold_{FOLD}/{abl_id}"
        cfg["run_name"] = f"abl_joint_chmm_f4_{abl_id}"
        cfg["runs"] = 3
        cfg["seed"] = 123

        out_path = OUT_ROOT / f"{abl_id}.yaml"
        with out_path.open("w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        written.append(out_path)
        print(f"Wrote {out_path} — {note}")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--round",
        choices=("1", "2", "3"),
        default=None,
        help="Ablation round (1=original 9, 2=combos, 3=warm_emb8_sticky92 lock).",
    )
    parser.add_argument(
        "--ablations",
        nargs="+",
        default=None,
        help="Subset of ablation ids (default: all in selected round).",
    )
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    generate(manifest, round_id=args.round, ablation_ids=args.ablations)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
