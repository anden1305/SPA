#!/usr/bin/env python3
"""Fold-level summary CSV for subject_lab_tune_winners results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.cv4fold.aggregate_val_nmi_by_lab import _val_mouse_ids
from scripts.cv4fold.select_best_vae_run import parse_nmi

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "results/cv4fold/subject_lab_tune_winners"
MODELS = ("cgmvae", "chmmgmvae")
FOLDS = (1, 2, 3, 4)
LABS = ("lab_2", "lab_3", "lab_5")


def _latest_timestamp_dir(fold_dir: Path) -> Path | None:
    if not fold_dir.is_dir():
        return None
    candidates = [p for p in fold_dir.iterdir() if p.is_dir() and (p / "config.json").exists()]
    if not candidates:
        if (fold_dir / "config.json").exists():
            return fold_dir
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _macro_mice_nmi(result_root: Path, run_num: int, val_ids: list[str]) -> float | None:
    nmis: list[float] = []
    for pid in val_ids:
        metrics_path = result_root / "per_mouse" / f"run_{run_num}" / pid / "metrics.json"
        if metrics_path.exists():
            nmis.append(float(json.loads(metrics_path.read_text(encoding="utf-8"))["nmi"]))
    return float(sum(nmis) / len(nmis)) if nmis else None


def collect_rows(model: str, fold: int, result_root: Path) -> list[dict]:
    by_lab_path = result_root / "val_nmi_by_lab.csv"
    if not by_lab_path.exists():
        return []
    cfg = json.loads((result_root / "config.json").read_text(encoding="utf-8"))
    val_ids = _val_mouse_ids(cfg)
    df = pd.read_csv(by_lab_path)
    rows: list[dict] = []
    for run_num in (1, 2, 3):
        sub = df[df["run"] == run_num]
        if sub.empty:
            continue
        row: dict = {
            "model": model,
            "fold": fold,
            "run": run_num,
            "result_dir": str(result_root),
            "nmi_pooled": parse_nmi(result_root / str(run_num) / "plots" / "metrics.txt"),
            "nmi_macro_mice": _macro_mice_nmi(result_root, run_num, val_ids),
        }
        for lab in LABS:
            lab_sub = sub[sub["lab"] == lab]
            row[f"nmi_{lab}"] = float(lab_sub["nmi_macro"].iloc[0]) if len(lab_sub) else None
        rows.append(row)
    return rows


def summarize(results_root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for model in MODELS:
        for fold in FOLDS:
            fold_dir = results_root / model / "joint" / f"fold_{fold}"
            latest = _latest_timestamp_dir(fold_dir)
            if latest is None:
                continue
            rows.extend(collect_rows(model, fold, latest))
    out = pd.DataFrame(rows)
    results_root.mkdir(parents=True, exist_ok=True)
    out_path = results_root / "summary.csv"
    out.to_csv(out_path, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    df = summarize(args.results_root)
    print(f"Wrote {args.results_root / 'summary.csv'} ({len(df)} rows)")


if __name__ == "__main__":
    main()
