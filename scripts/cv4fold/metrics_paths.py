"""Resolve cv4fold result paths across layout variants."""

from __future__ import annotations

from pathlib import Path


def metrics_path(result_root: Path, run_num: int) -> Path | None:
    """Return post-train metrics.txt for seed ``run_num`` if present."""
    for candidate in (
        result_root / "plots" / str(run_num) / "metrics.txt",
        result_root / str(run_num) / "plots" / str(run_num) / "metrics.txt",
        result_root / str(run_num) / "plots" / "metrics.txt",
    ):
        if candidate.is_file():
            return candidate
    return None


def results_npz_path(result_root: Path, run_num: int) -> Path | None:
    """Return validation results.npz for seed ``run_num`` if present."""
    for candidate in (
        result_root / "plots" / str(run_num) / "results.npz",
        result_root / str(run_num) / "plots" / str(run_num) / "results.npz",
        result_root / str(run_num) / "plots" / "results.npz",
    ):
        if candidate.is_file():
            return candidate
    return None
