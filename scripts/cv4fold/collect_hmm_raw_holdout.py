#!/usr/bin/env python3
"""Inventory HMM-on-raw holdout runs and write paper/cv4fold status CSV."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scripts.cv4fold.manifest_utils import load_manifest
from scripts.cv4fold.metrics_paths import metrics_path, results_npz_path
from scripts.cv4fold.select_best_vae_run import parse_nmi
from scripts.cv4fold.split_per_mouse import subject_id_map_from_config
from scripts.paper.scrape_holdout_accuracy import macro_map_accuracy
from src.helpers.nmi import calculate_nmi

REPO = Path(__file__).resolve().parents[2]
HMM_ROOT = REPO / "results/cv4fold/hmm_raw"
PARTIAL_JOINT_JSON = REPO / "paper/overleaf/tables/hmm_raw_joint_partial.json"
LABS = ("lab_2", "lab_3", "lab_5")
FOLDS = (1, 2, 3, 4)


def _partial_joint_nmi() -> dict[str, float]:
    """Accepted log NMI for joint folds that OOM'd / hit walltime before results.npz."""
    if not PARTIAL_JOINT_JSON.is_file():
        return {}
    data = json.loads(PARTIAL_JOINT_JSON.read_text(encoding="utf-8"))
    return {str(k): float(v["nmi"]) for k, v in data.get("folds", {}).items()}


@dataclass
class HmmRawRow:
    scope: str
    fold: str
    lab: str
    nmi: float | None
    macro_accuracy: float | None
    status: str
    result_root: str


def _latest_run(model_dir: Path) -> Path | None:
    if not model_dir.is_dir():
        return None
    runs = sorted(
        (p for p in model_dir.iterdir() if p.is_dir() and "hmm_raw" in p.name),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for run in runs:
        if results_npz_path(run, 1) is not None:
            return run
        mp = metrics_path(run, 1)
        if mp is not None and mp.exists():
            return run
    return runs[0] if runs else None


def _lab_for_sub_id(
    sub_id: int,
    lab_by_mouse: dict[str, str],
    id_map: dict[int, str] | None,
) -> str | None:
    if id_map is not None and int(sub_id) in id_map:
        return lab_by_mouse.get(id_map[int(sub_id)])
    return lab_by_mouse.get(f"sub-{int(sub_id):03d}")


def _load_npz_labels(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(npz_path, mmap_mode="r")
    return (
        data["y_true"].astype(int).ravel(),
        data["y_hat"].astype(int).ravel(),
        data["sub_ids"].astype(int).ravel(),
    )


def _metrics_pooled(npz_path: Path) -> tuple[float, float]:
    yt, yh, _ = _load_npz_labels(npz_path)
    return float(calculate_nmi(yh, yt)), float(macro_map_accuracy(yt, yh))


def _metrics_for_lab(
    npz_path: Path,
    target_lab: str,
    lab_by_mouse: dict[str, str],
    id_map: dict[int, str] | None = None,
) -> tuple[float, float] | None:
    yt, yh, sub_ids = _load_npz_labels(npz_path)
    if id_map is not None:
        target_sids = [
            sid for sid, pid in id_map.items() if lab_by_mouse.get(pid) == target_lab
        ]
        mask = np.isin(sub_ids, target_sids) if target_sids else np.zeros_like(sub_ids, dtype=bool)
    else:
        target_sids = [
            int(s)
            for s in np.unique(sub_ids)
            if _lab_for_sub_id(int(s), lab_by_mouse, None) == target_lab
        ]
        mask = np.isin(sub_ids, target_sids) if target_sids else np.zeros_like(sub_ids, dtype=bool)
    if not mask.any():
        return None
    return float(calculate_nmi(yh[mask], yt[mask])), float(macro_map_accuracy(yt[mask], yh[mask]))


def collect_rows(manifest: dict) -> list[HmmRawRow]:
    lab_by_mouse = {mid: info["lab"] for mid, info in manifest["inventory"].items()}
    rows: list[HmmRawRow] = []
    partial_joint = _partial_joint_nmi()

    for fold in FOLDS:
        run_dir = _latest_run(HMM_ROOT / "joint" / f"fold_{fold}")
        if run_dir is None:
            rows.append(HmmRawRow("joint", str(fold), "all", None, None, "missing", ""))
            continue
        npz = results_npz_path(run_dir, 1)
        if npz is None:
            mp = metrics_path(run_dir, 1)
            nmi = parse_nmi(mp) if mp else None
            if nmi is None and str(fold) in partial_joint:
                nmi = partial_joint[str(fold)]
                rows.append(
                    HmmRawRow(
                        "joint",
                        str(fold),
                        "all",
                        nmi,
                        None,
                        "partial_log",
                        str(run_dir),
                    )
                )
                continue
            rows.append(
                HmmRawRow(
                    "joint",
                    str(fold),
                    "all",
                    nmi,
                    None,
                    "metrics_only" if nmi is not None else "no_npz",
                    str(run_dir),
                )
            )
            continue
        nmi, acc = _metrics_pooled(npz)
        rows.append(HmmRawRow("joint", str(fold), "all", nmi, acc, "ok", str(run_dir)))

    for lab in LABS:
        for fold in FOLDS:
            run_dir = _latest_run(HMM_ROOT / "unified_holdout" / lab / f"fold_{fold}")
            if run_dir is None:
                rows.append(HmmRawRow("within_lab", str(fold), lab, None, None, "missing", ""))
                continue
            npz = results_npz_path(run_dir, 1)
            if npz is None:
                mp = metrics_path(run_dir, 1)
                nmi = parse_nmi(mp) if mp else None
                rows.append(
                    HmmRawRow(
                        "within_lab",
                        str(fold),
                        lab,
                        nmi,
                        None,
                        "metrics_only" if nmi is not None else "no_npz",
                        str(run_dir),
                    )
                )
                continue
            nmi, acc = _metrics_pooled(npz)
            rows.append(HmmRawRow("within_lab", str(fold), lab, nmi, acc, "ok", str(run_dir)))

    return rows


def collect_joint_per_lab_rows(manifest: dict) -> list[HmmRawRow]:
    """Joint train, metrics on one lab's holdout mice only (fair comparison)."""
    lab_by_mouse = {mid: info["lab"] for mid, info in manifest["inventory"].items()}
    out: list[HmmRawRow] = []
    for fold in FOLDS:
        run_dir = _latest_run(HMM_ROOT / "joint" / f"fold_{fold}")
        if run_dir is None:
            for lab in LABS:
                out.append(HmmRawRow("joint", str(fold), lab, None, None, "missing", ""))
            continue
        npz = results_npz_path(run_dir, 1)
        if npz is None:
            for lab in LABS:
                out.append(HmmRawRow("joint", str(fold), lab, None, None, "no_npz", str(run_dir)))
            continue
        id_map = subject_id_map_from_config(run_dir / "config.json")
        for lab in LABS:
            m = _metrics_for_lab(npz, lab, lab_by_mouse, id_map)
            if m is None:
                out.append(HmmRawRow("joint", str(fold), lab, None, None, "no_lab_epochs", str(run_dir)))
            else:
                out.append(HmmRawRow("joint", str(fold), lab, m[0], m[1], "ok", str(run_dir)))
    return out


def write_csv(path: Path, rows: list[HmmRawRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["scope", "fold", "lab", "model", "model_label", "nmi", "macro_accuracy", "status", "result_root"]
        )
        for r in rows:
            w.writerow(
                [
                    r.scope,
                    r.fold,
                    r.lab,
                    "hmm_raw",
                    "HMM (raw)",
                    "" if r.nmi is None else f"{r.nmi:.4f}",
                    "" if r.macro_accuracy is None else f"{r.macro_accuracy:.4f}",
                    r.status,
                    r.result_root,
                ]
            )


def write_status_json(path: Path, rows: list[HmmRawRow], joint_per_lab: list[HmmRawRow]) -> None:
    def _count_ok(rs: list[HmmRawRow]) -> int:
        return sum(1 for r in rs if r.status == "ok")

    summary = {
        "expected_joint_folds": len(FOLDS),
        "expected_within_lab_cells": len(LABS) * len(FOLDS),
        "joint_pooled_ok": _count_ok([r for r in rows if r.scope == "joint"]),
        "within_lab_ok": _count_ok([r for r in rows if r.scope == "within_lab"]),
        "joint_per_lab_ok": _count_ok(joint_per_lab),
        "incomplete": True,
        "cells": [
            {
                "scope": r.scope,
                "fold": r.fold,
                "lab": r.lab,
                "status": r.status,
                "nmi": r.nmi,
                "macro_accuracy": r.macro_accuracy,
            }
            for r in rows
        ],
        "joint_per_lab": [
            {
                "fold": r.fold,
                "lab": r.lab,
                "status": r.status,
                "nmi": r.nmi,
                "macro_accuracy": r.macro_accuracy,
            }
            for r in joint_per_lab
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_status_tex(path: Path, rows: list[HmmRawRow]) -> None:
    """LaTeX grid: which joint / within-lab HMM-raw cells have results.npz."""
    by_key: dict[tuple[str, str, str], str] = {}
    for r in rows:
        if r.scope == "joint" and r.lab != "all":
            continue
        by_key[(r.scope, r.fold, r.lab if r.scope == "within_lab" else "joint")] = r.status

    def cell(scope: str, fold: int, lab: str) -> str:
        key = (scope, str(fold), lab)
        st = by_key.get(key)
        if st == "ok":
            return "$\\checkmark$"
        if st == "partial_log":
            return "$\\checkmark^{\\dagger}$"
        return "---"

    lines = [
        "% Auto-generated by scripts/cv4fold/collect_hmm_raw_holdout.py",
        "\\begin{tabular}{|l|c|c|c|c|}",
        "\\hline",
        "\\textbf{Scope} & \\textbf{Fold 1} & \\textbf{Fold 2} & \\textbf{Fold 3} & \\textbf{Fold 4} \\\\",
        "\\thickhline",
        f"Joint & {cell('joint',1,'joint')} & {cell('joint',2,'joint')} & "
        f"{cell('joint',3,'joint')} & {cell('joint',4,'joint')} \\\\",
        "\\hline",
    ]
    for lab in LABS:
        label = lab.replace("_", " ").title()
        lines.append(
            f"Within {label} & {cell('within_lab',1,lab)} & {cell('within_lab',2,lab)} & "
            f"{cell('within_lab',3,lab)} & {cell('within_lab',4,lab)} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=REPO / "paper/overleaf/tables/hmm_raw_holdout.csv",
    )
    parser.add_argument(
        "--status-json",
        type=Path,
        default=REPO / "paper/overleaf/tables/hmm_raw_holdout_status.json",
    )
    parser.add_argument(
        "--status-tex",
        type=Path,
        default=REPO / "docs/cv4fold/hmm_raw_holdout_status.tex",
    )
    args = parser.parse_args()

    manifest = load_manifest()
    rows = collect_rows(manifest)
    joint_per_lab = collect_joint_per_lab_rows(manifest)
    write_csv(args.out_csv, rows + joint_per_lab)
    write_status_json(args.status_json, rows, joint_per_lab)
    write_status_tex(args.status_tex, rows)

    ok_pooled = sum(1 for r in rows if r.status == "ok")
    ok_jpl = sum(1 for r in joint_per_lab if r.status == "ok")
    print(f"Wrote {args.out_csv} ({ok_pooled} pooled ok, {ok_jpl} joint-per-lab ok)")
    print(f"Wrote {args.status_json}")
    print(f"Wrote {args.status_tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
