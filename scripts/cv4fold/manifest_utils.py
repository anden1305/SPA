"""Shared manifest loading for cv4fold scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "data/manifests/cv_quality_cohort_v1.yaml"


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    manifest_path = path or DEFAULT_MANIFEST
    with manifest_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def all_mice(manifest: dict) -> list[str]:
    mice: list[str] = []
    for lab_mice in manifest["cohort"].values():
        mice.extend(lab_mice)
    return mice


def holdout_for_fold(manifest: dict, fold: int, scope: str, lab: str | None = None) -> list[str]:
    fold_key = f"fold_{fold}"
    splits = manifest["splits"][fold_key]
    if scope == "joint":
        holdout: list[str] = []
        for mice in splits.values():
            holdout.extend(mice)
        return holdout
    if lab is None:
        raise ValueError("lab required for per_lab scope")
    return list(splits[lab])


def train_mice(manifest: dict, fold: int, scope: str, lab: str | None = None) -> list[str]:
    holdout = set(holdout_for_fold(manifest, fold, scope, lab))
    if scope == "joint":
        pool = all_mice(manifest)
    else:
        if lab is None:
            raise ValueError("lab required for per_lab scope")
        pool = list(manifest["cohort"][lab])
    return [m for m in pool if m not in holdout]


def dataset_entries(manifest: dict, mouse_ids: list[str]) -> list[dict]:
    entries: list[dict] = []
    for mouse_id in mouse_ids:
        info = manifest["inventory"][mouse_id]
        for run in info["runs"]:
            entry: dict[str, Any] = {
                "type": "mssv",
                "id": mouse_id,
                "run": run,
                "remove_artifact": True,
            }
            if "signals" in info:
                entry["signals"] = list(info["signals"])
            entries.append(entry)
    return entries
