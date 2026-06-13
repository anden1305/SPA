"""Collect K-sweep metrics across multiple run dirs (base 3-seed + extra2 jobs)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from scripts.paper.plot_k_sweep_dual_axis import _parse_metrics


def _read_seed_metrics(run_dir: Path) -> list[tuple[int, dict[str, float]]]:
    """Return [(plot_index, metrics), ...] for one run directory."""
    plots = run_dir / "plots"
    if not plots.is_dir():
        return []
    out: list[tuple[int, dict[str, float]]] = []
    for seed_dir in sorted(
        (p for p in plots.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    ):
        m = _parse_metrics(seed_dir / "metrics.txt")
        if m:
            out.append((int(seed_dir.name), m))
    if not out:
        m = _parse_metrics(plots / "metrics.txt")
        if m:
            out.append((1, m))
    return out


def collect_all_seeds_for_k(k_dir: Path) -> list[dict]:
    """Merge seed metrics from all run dirs under K{k}/ (oldest run first)."""
    if not k_dir.is_dir():
        return []
    runs = sorted(
        (p for p in k_dir.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
    )
    merged: list[dict] = []
    for run in runs:
        for _idx, m in _read_seed_metrics(run):
            merged.append({**m, "run": run.name})
    return merged


def collect_k_sweep_merged(
    root: Path,
    k_min: int = 3,
    k_max: int = 15,
    *,
    include_extra2: bool = False,
) -> dict[int, dict]:
    """Per-K aggregates using seeds across run directories.

    By default uses one run per K: holdout excludes ``extra2`` batches; when several
    non-extra2 runs exist (population retries), keeps only the latest run directory.
    Pass ``include_extra2=True`` to merge all run directories.
    """
    out: dict[int, dict] = {}
    if not root.is_dir():
        return out
    for k in range(k_min, k_max + 1):
        k_dir = root / f"K{k}"
        seeds = collect_all_seeds_for_k(k_dir)
        if not include_extra2:
            seeds = [s for s in seeds if "extra2" not in s.get("run", "")]
            run_names = sorted({s["run"] for s in seeds})
            if len(run_names) > 1:
                latest = max(run_names, key=lambda name: (k_dir / name).stat().st_mtime)
                seeds = [s for s in seeds if s["run"] == latest]
        if not seeds:
            continue
        nmis = [s["nmi"] for s in seeds]
        lls = [s["log_p_z"] for s in seeds if "log_p_z" in s]
        runs = sorted({s["run"] for s in seeds})
        out[k] = {
            "run": runs[-1] if runs else "",
            "runs_merged": runs,
            "n_seeds": len(seeds),
            "seeds": [{k: v for k, v in s.items() if k != "run"} for s in seeds],
            "nmi_mean": float(np.mean(nmis)),
            "nmi_std": float(np.std(nmis)) if len(nmis) > 1 else 0.0,
            "nmi_best": float(max(nmis)),
            "nmi_min": float(min(nmis)),
            "log_p_z_mean": float(np.mean(lls)) if lls else None,
            "log_p_z_std": float(np.std(lls)) if len(lls) > 1 else 0.0,
        }
    return out


def best_npz_for_k_merged(root: Path, k: int) -> tuple[Path, int, float, str] | None:
    """Best (npz, plot_index, nmi, run_name) across all runs at K."""
    k_dir = root / f"K{k}"
    if not k_dir.is_dir():
        return None
    best: tuple[Path, int, float, str] | None = None
    for run in sorted((p for p in k_dir.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime):
        for idx, m in _read_seed_metrics(run):
            npz = run / "plots" / str(idx) / "results.npz"
            if not npz.is_file():
                continue
            nmi = m["nmi"]
            if best is None or nmi > best[2]:
                best = (npz, idx, nmi, run.name)
    return best
