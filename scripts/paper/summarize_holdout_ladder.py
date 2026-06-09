#!/usr/bin/env python3
"""Aggregate best-of-3 holdout NMI for paper model ladder (Fig 2 / Table)."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

from scripts.cv4fold.select_best_vae_run import select_best_run

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOTS = (
    REPO_ROOT / "results/cv4fold/joint_holdout",
    REPO_ROOT / "results/cv4fold/unified_holdout",
)

MODEL_ORDER = ("hmm_features", "hmmgmvae_locked", "cgmvae_locked", "chmmgmvae_locked")
MODEL_LABEL = {
    "hmm_features": "HMM (features)",
    "hmmgmvae_locked": "HMMGMVAE",
    "cgmvae_locked": "cGMVAE",
    "chmmgmvae_locked": "cHMMGMVAE",
}

# Thesis population LOSO reference (supplement baseline rung).
THESIS_BASELINE = {
    "hmm_features": {"nmi": 0.51, "source": "thesis Table 10 population"},
}


@dataclass
class Row:
    scope: str
    fold: str
    lab: str
    model: str
    nmi: float
    result_root: str
    selected_run: int


def _parse_joint_path(model_dir: Path) -> tuple[str, str, str]:
    # joint_holdout/fold_{k}/{model}_locked/<run_ts>
    fold = model_dir.parent.name.replace("fold_", "")
    return "joint", fold, "all"


def _parse_within_path(model_dir: Path) -> tuple[str, str, str]:
    # unified_holdout/{lab}/fold_{k}/{model}_locked/<run_ts>
    lab = model_dir.parent.parent.name
    fold = model_dir.parent.name.replace("fold_", "")
    return "within_lab", fold, lab


def _latest_run_dir(model_dir: Path) -> Path | None:
    if not model_dir.is_dir():
        return None
    candidates = sorted(
        (p for p in model_dir.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def collect_rows(roots: tuple[Path, ...]) -> list[Row]:
    rows: list[Row] = []
    for root in roots:
        if not root.exists():
            continue
        for model_dir in root.glob("**/fold_*/*_locked"):
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            if model not in MODEL_LABEL:
                continue
            run_dir = _latest_run_dir(model_dir)
            if run_dir is None:
                continue
            try:
                sel = select_best_run(run_dir)
            except FileNotFoundError:
                continue
            if "joint_holdout" in str(model_dir):
                scope, fold, lab = _parse_joint_path(model_dir)
            else:
                scope, fold, lab = _parse_within_path(model_dir)
            rows.append(
                Row(
                    scope=scope,
                    fold=fold,
                    lab=lab,
                    model=model,
                    nmi=sel["nmi"],
                    result_root=sel["result_root"],
                    selected_run=sel["selected_run"],
                )
            )
    return rows


def write_csv(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["scope", "fold", "lab", "model", "model_label", "nmi", "selected_run", "result_root"]
        )
        for r in rows:
            w.writerow(
                [
                    r.scope,
                    r.fold,
                    r.lab,
                    r.model,
                    MODEL_LABEL.get(r.model, r.model),
                    f"{r.nmi:.4f}",
                    r.selected_run,
                    r.result_root,
                ]
            )


def joint_summary(rows: list[Row]) -> dict[str, dict[str, float]]:
    """Mean best-of-3 NMI per model across folds (joint scope only)."""
    by_model: dict[str, list[float]] = {m: [] for m in MODEL_ORDER if m != "hmm_features"}
    for r in rows:
        if r.scope != "joint":
            continue
        by_model.setdefault(r.model, []).append(r.nmi)
    out: dict[str, dict[str, float]] = {}
    for model, nmis in by_model.items():
        if not nmis:
            continue
        out[model] = {
            "mean_nmi": sum(nmis) / len(nmis),
            "n_folds": len(nmis),
            "min_nmi": min(nmis),
            "max_nmi": max(nmis),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "paper/overleaf/tables/holdout_ladder.csv",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=REPO_ROOT / "paper/overleaf/tables/holdout_ladder_summary.json",
    )
    args = parser.parse_args()

    rows = collect_rows(DEFAULT_ROOTS)
    write_csv(args.out, rows)

    summary = {
        "joint_by_model": joint_summary(rows),
        "thesis_baseline": THESIS_BASELINE,
        "n_rows": len(rows),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {args.out} ({len(rows)} rows)")
    print(f"Wrote {args.json}")
    if summary["joint_by_model"]:
        print("\nJoint holdout mean NMI (available folds):")
        for model in MODEL_ORDER:
            if model == "hmm_features":
                print(f"  {MODEL_LABEL[model]:12s}  {THESIS_BASELINE['hmm_features']['nmi']:.3f}  (thesis)")
                continue
            if model in summary["joint_by_model"]:
                m = summary["joint_by_model"][model]
                print(
                    f"  {MODEL_LABEL[model]:12s}  {m['mean_nmi']:.3f}  "
                    f"(n={m['n_folds']}, range {m['min_nmi']:.3f}–{m['max_nmi']:.3f})"
                )
    else:
        print("No joint holdout results yet — submit holdout jobs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
