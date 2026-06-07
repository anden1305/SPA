#!/usr/bin/env python3
"""Build cv4fold quality-cohort manifest from metadata.csv (login-node safe)."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
METADATA_PATH = REPO_ROOT / "data/ds006366_processed/metadata.csv"
MANIFEST_PATH = REPO_ROOT / "data/manifests/cv_quality_cohort_v1.yaml"
SUMMARY_CSV = REPO_ROOT / "results/cv4fold/manifest_summary.csv"

COHORT = {
    "lab_2": ["sub-071", "sub-072", "sub-076", "sub-077", "sub-080", "sub-081"],
    "lab_3": [
        "sub-038", "sub-039", "sub-041", "sub-043", "sub-048", "sub-054",
        "sub-056", "sub-059", "sub-060", "sub-069",
    ],
    "lab_5": ["sub-087", "sub-088", "sub-089", "sub-092"],
}

LAB_SIGNALS = {
    "lab_2": ["EEG1", "EEG3", "EMG"],
    "lab_3": ["EEG1", "EEG2", "EMG"],
    "lab_5": ["EEG1", "EEG2", "EMG"],
}

SPLITS = {
    "fold_1": {"lab_2": ["sub-071"], "lab_3": ["sub-038", "sub-039"], "lab_5": ["sub-087"]},
    "fold_2": {
        "lab_2": ["sub-072", "sub-076"],
        "lab_3": ["sub-041", "sub-048", "sub-069"],
        "lab_5": ["sub-088"],
    },
    "fold_3": {"lab_2": ["sub-077"], "lab_3": ["sub-043", "sub-054"], "lab_5": ["sub-089"]},
    "fold_4": {
        "lab_2": ["sub-080", "sub-081"],
        "lab_3": ["sub-056", "sub-059", "sub-060"],
        "lab_5": ["sub-092"],
    },
}


def _all_mice() -> list[str]:
    out: list[str] = []
    for mice in COHORT.values():
        out.extend(mice)
    return out


def _mouse_lab(mouse_id: str) -> str:
    for lab, mice in COHORT.items():
        if mouse_id in mice:
            return lab
    raise KeyError(mouse_id)


def build_inventory(meta: pd.DataFrame) -> dict:
    inventory: dict = {}
    for mouse_id in _all_mice():
        rows = meta.loc[meta["participant_id"] == mouse_id].sort_values("run")
        if rows.empty:
            raise ValueError(f"No metadata for {mouse_id}")
        runs = [int(r) for r in rows["run"].tolist()]
        hours = float(rows["seconds"].sum() / 3600.0)
        epochs_4s = int(rows["seconds"].sum() / 4.0)
        inventory[mouse_id] = {
            "lab": _mouse_lab(mouse_id),
            "runs": runs,
            "hours": round(hours, 2),
            "epochs_4s": epochs_4s,
            "signals": LAB_SIGNALS[_mouse_lab(mouse_id)],
        }
    return inventory


def _fold_holdout_union(fold_key: str) -> list[str]:
    holdout: list[str] = []
    for mice in SPLITS[fold_key].values():
        holdout.extend(mice)
    return holdout


def build_fold_summary(inventory: dict) -> list[dict]:
    rows: list[dict] = []
    for fold_key, by_lab in SPLITS.items():
        fold_idx = int(fold_key.split("_")[1])
        holdout = _fold_holdout_union(fold_key)
        holdout_runs = sum(len(inventory[m]["runs"]) for m in holdout)
        holdout_hours = sum(inventory[m]["hours"] for m in holdout)
        rows.append({
            "fold": fold_idx,
            "scope": "joint",
            "n_holdout_mice": len(holdout),
            "holdout_runs": holdout_runs,
            "holdout_hours": round(holdout_hours, 2),
            "holdout_mice": ",".join(holdout),
        })
        for lab, mice in by_lab.items():
            rows.append({
                "fold": fold_idx,
                "scope": f"per_lab/{lab}",
                "n_holdout_mice": len(mice),
                "holdout_runs": sum(len(inventory[m]["runs"]) for m in mice),
                "holdout_hours": round(sum(inventory[m]["hours"] for m in mice), 2),
                "holdout_mice": ",".join(mice),
            })
    return rows


def main() -> None:
    meta = pd.read_csv(METADATA_PATH)
    inventory = build_inventory(meta)

    manifest = {
        "cohort": COHORT,
        "lab_signals": LAB_SIGNALS,
        "inventory": inventory,
        "cv_protocol": "4fold",
        "eval": {"report_per_mouse": True},
        "runs_policy": {
            "vae": {"runs": 3, "select": "best_val_gmm_pred_nmi"},
            "hmm_raw": {"runs": 1},
        },
        "trainer": {
            "vae_validate_per_epoch": 10,
            "hmm_raw_validate_per_epoch": 1000,
        },
        "hyperparams": {
            "learning_rate": 0.0003,
            "epochs": 80,
        },
        "splits": SPLITS,
        "changelog": [
            "lab_3 rebalance: sub-069 in fold_2 (not fold_4) to avoid 4-mice lab_3 holdout",
        ],
    }

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False, allow_unicode=True)

    summary_rows = []
    for mouse_id, info in sorted(inventory.items()):
        summary_rows.append({
            "participant_id": mouse_id,
            "lab": info["lab"],
            "n_runs": len(info["runs"]),
            "hours": info["hours"],
            "epochs_4s": info["epochs_4s"],
        })
    summary_rows.extend(build_fold_summary(inventory))

    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"Wrote {MANIFEST_PATH}")
    print(f"Wrote {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
