"""Manifest helpers for cv4fold quality cohort."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "data/manifests/cv_quality_cohort_v1.yaml"
LABS = ("lab_2", "lab_3", "lab_5")


@dataclass(frozen=True)
class RunRecord:
    participant_id: str
    lab: str
    run: int
    signals: list[str]
    hours: float


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    manifest_path = path or DEFAULT_MANIFEST
    with manifest_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def iter_runs(manifest: dict[str, Any]) -> list[RunRecord]:
    records: list[RunRecord] = []
    for mouse_id, info in manifest["inventory"].items():
        lab = info["lab"]
        for run in info["runs"]:
            records.append(
                RunRecord(
                    participant_id=mouse_id,
                    lab=lab,
                    run=int(run),
                    signals=list(info["signals"]),
                    hours=float(info.get("hours", 0.0)),
                )
            )
    return records


def mice_by_lab(manifest: dict[str, Any]) -> dict[str, list[str]]:
    return {lab: list(manifest["cohort"][lab]) for lab in LABS}


def export_cohort_inventory(manifest: dict[str, Any], out_path: Path) -> None:
    import csv

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "participant_id",
                "lab",
                "run",
                "hours",
                "epochs_4s",
                "signals",
            ]
        )
        for rec in iter_runs(manifest):
            info = manifest["inventory"][rec.participant_id]
            w.writerow(
                [
                    rec.participant_id,
                    rec.lab,
                    rec.run,
                    rec.hours,
                    info.get("epochs_4s", ""),
                    ";".join(rec.signals),
                ]
            )
