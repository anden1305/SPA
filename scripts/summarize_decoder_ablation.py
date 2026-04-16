#!/usr/bin/env python3
"""Summarize baseline vs decoder-only CVAE ablation from local W&B files.

Outputs mean, std, SEM, and paired seed deltas for:
- val/cvae_latent_kmeans_nmi
- val/log_likelihood

Usage:
  uv run python scripts/summarize_decoder_ablation.py
  python scripts/summarize_decoder_ablation.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Optional, Tuple

import yaml


def _safe_mean(xs: List[float]) -> Optional[float]:
    return mean(xs) if xs else None


def _safe_std(xs: List[float]) -> Optional[float]:
    if len(xs) < 2:
        return 0.0 if len(xs) == 1 else None
    return stdev(xs)


def _safe_sem(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    std = _safe_std(xs)
    if std is None:
        return None
    return std / math.sqrt(len(xs))


def _fmt(x: Optional[float]) -> str:
    return "NA" if x is None else f"{x:.4f}"


def _load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def _load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def _condition_from_cfg(cfg: dict) -> Optional[str]:
    # Primary key: explicit boolean switch in model params
    try:
        dec_only = cfg["model"]["value"]["params"]["decoder_only_conditioning"]
    except Exception:
        dec_only = None

    if dec_only is True:
        return "decoder_only"
    if dec_only is False:
        return "baseline"

    # Fallback to args path
    try:
        args = cfg["_wandb"]["value"]["e"]
        first_writer = next(iter(args.values()))
        argv = first_writer.get("args", [])
        joined = " ".join(argv)
        if "cvae_final_decoder_only.yaml" in joined:
            return "decoder_only"
        if "cvae_final.yaml" in joined:
            return "baseline"
    except Exception:
        pass

    return None


def _extract_seed(cfg: dict) -> Optional[int]:
    try:
        return int(cfg["seed"]["value"])
    except Exception:
        return None


def _extract_metrics(summary: dict) -> Tuple[Optional[float], Optional[float]]:
    nmi = summary.get("val/cvae_latent_kmeans_nmi")
    log_likelihood = summary.get("val/log_likelihood")
    return nmi, log_likelihood


def collect_runs(wandb_dir: Path) -> Dict[str, Dict[int, dict]]:
    results: Dict[str, Dict[int, dict]] = {"baseline": {}, "decoder_only": {}}

    for run_dir in sorted(wandb_dir.glob("run-*/files")):
        cfg_path = run_dir / "config.yaml"
        summary_path = run_dir / "wandb-summary.json"
        if not cfg_path.exists() or not summary_path.exists():
            continue

        try:
            cfg = _load_yaml(cfg_path)
            summary = _load_json(summary_path)
        except Exception:
            continue

        condition = _condition_from_cfg(cfg)
        if condition is None:
            continue

        seed = _extract_seed(cfg)
        if seed is None:
            continue

        nmi, ll = _extract_metrics(summary)

        # Keep latest run for a seed/condition by replacing older entry.
        results[condition][seed] = {
            "nmi": nmi,
            "log_likelihood": ll,
            "path": str(run_dir.parent),
        }

    return results


def summarize(results: Dict[str, Dict[int, dict]]) -> None:
    baseline = results["baseline"]
    decoder = results["decoder_only"]

    seeds = sorted(set(baseline.keys()) & set(decoder.keys()))
    print("Matched seeds:", seeds)
    if not seeds:
        print("No matched seeds found between baseline and decoder-only runs.")
        return

    b_nmi = [baseline[s]["nmi"] for s in seeds if baseline[s]["nmi"] is not None]
    d_nmi = [decoder[s]["nmi"] for s in seeds if decoder[s]["nmi"] is not None]
    b_ll = [baseline[s]["log_likelihood"] for s in seeds if baseline[s]["log_likelihood"] is not None]
    d_ll = [decoder[s]["log_likelihood"] for s in seeds if decoder[s]["log_likelihood"] is not None]

    nmi_deltas = []
    ll_deltas = []
    for s in seeds:
        bn, dn = baseline[s]["nmi"], decoder[s]["nmi"]
        bl, dl = baseline[s]["log_likelihood"], decoder[s]["log_likelihood"]
        if bn is not None and dn is not None:
            nmi_deltas.append(dn - bn)
        if bl is not None and dl is not None:
            ll_deltas.append(dl - bl)

    print("\n=== Aggregate (mean +- SEM) ===")
    print(f"NMI baseline      : {_fmt(_safe_mean(b_nmi))} +- {_fmt(_safe_sem(b_nmi))}")
    print(f"NMI decoder-only  : {_fmt(_safe_mean(d_nmi))} +- {_fmt(_safe_sem(d_nmi))}")
    print(f"NMI delta (D-B)   : {_fmt(_safe_mean(nmi_deltas))} +- {_fmt(_safe_sem(nmi_deltas))}")

    print(f"LogLik baseline   : {_fmt(_safe_mean(b_ll))} +- {_fmt(_safe_sem(b_ll))}")
    print(f"LogLik decoder-only: {_fmt(_safe_mean(d_ll))} +- {_fmt(_safe_sem(d_ll))}")
    print(f"LogLik delta (D-B): {_fmt(_safe_mean(ll_deltas))} +- {_fmt(_safe_sem(ll_deltas))}")

    print("\n=== Per-seed ===")
    print("seed\tNMI_baseline\tNMI_decoder\tDelta\tLL_baseline\tLL_decoder\tDelta")
    for s in seeds:
        bn, dn = baseline[s]["nmi"], decoder[s]["nmi"]
        bl, dl = baseline[s]["log_likelihood"], decoder[s]["log_likelihood"]
        d_n = None if (bn is None or dn is None) else (dn - bn)
        d_l = None if (bl is None or dl is None) else (dl - bl)
        print(
            f"{s}\t{_fmt(bn)}\t{_fmt(dn)}\t{_fmt(d_n)}\t{_fmt(bl)}\t{_fmt(dl)}\t{_fmt(d_l)}"
        )


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    wandb_dir = repo_root / "wandb"
    if not wandb_dir.exists():
        raise SystemExit(f"No wandb directory found at: {wandb_dir}")

    results = collect_runs(wandb_dir)
    summarize(results)
