#!/usr/bin/env python3
"""Macro-average validation NMI per lab from per_mouse metrics."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from scripts.cv4fold.manifest_utils import DEFAULT_MANIFEST, load_manifest
from scripts.cv4fold.select_best_vae_run import parse_nmi


def _val_mouse_ids(config: dict) -> list[str]:
    seen: list[str] = []
    for entry in config.get("val_datasets", []):
        pid = entry["id"]
        if pid not in seen:
            seen.append(pid)
    return seen


def _lab_for_mouse(manifest: dict) -> dict[str, str]:
    return {mid: info["lab"] for mid, info in manifest["inventory"].items()}


def aggregate_run(
    per_mouse_run_dir: Path,
    val_ids: list[str],
    lab_map: dict[str, str],
) -> list[dict]:
    by_lab: dict[str, list[float]] = defaultdict(list)
    for pid in val_ids:
        metrics_path = per_mouse_run_dir / pid / "metrics.json"
        if not metrics_path.exists():
            continue
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        lab = lab_map.get(pid)
        if lab is None:
            continue
        by_lab[lab].append(float(payload["nmi"]))
    rows: list[dict] = []
    for lab in sorted(by_lab.keys()):
        nmis = by_lab[lab]
        rows.append(
            {
                "lab": lab,
                "nmi_macro": float(sum(nmis) / len(nmis)),
                "n_mice": len(nmis),
            }
        )
    return rows


def aggregate_val_nmi_by_lab(
    result_root: Path,
    manifest: dict | None = None,
) -> pd.DataFrame:
    """Write val_nmi_by_lab.csv and val_nmi_summary.json under result_root."""
    config_path = result_root / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = manifest or load_manifest()
    val_ids = _val_mouse_ids(cfg)
    lab_map = _lab_for_mouse(manifest)

    rows: list[dict] = []
    summary_runs: list[dict] = []
    for run_num in (1, 2, 3):
        per_mouse_dir = result_root / "per_mouse" / f"run_{run_num}"
        metrics_txt = result_root / str(run_num) / "plots" / "metrics.txt"
        pooled = parse_nmi(metrics_txt)
        if per_mouse_dir.is_dir():
            for lab_row in aggregate_run(per_mouse_dir, val_ids, lab_map):
                rows.append({"run": run_num, **lab_row})
        summary_runs.append({"run": run_num, "nmi_pooled": pooled})

    df = pd.DataFrame(rows)
    df.to_csv(result_root / "val_nmi_by_lab.csv", index=False)

    macro_by_run: dict[int, float] = {}
    for run_num in (1, 2, 3):
        per_mouse_dir = result_root / "per_mouse" / f"run_{run_num}"
        if not per_mouse_dir.is_dir():
            continue
        nmis: list[float] = []
        for pid in val_ids:
            metrics_path = per_mouse_dir / pid / "metrics.json"
            if metrics_path.exists():
                nmis.append(float(json.loads(metrics_path.read_text())["nmi"]))
        if nmis:
            macro_by_run[run_num] = float(sum(nmis) / len(nmis))

    summary = {
        "val_mouse_ids": val_ids,
        "runs": summary_runs,
        "nmi_macro_mice_by_run": macro_by_run,
        "nmi_macro_mice_mean_across_runs": (
            float(sum(macro_by_run.values()) / len(macro_by_run)) if macro_by_run else None
        ),
    }
    (result_root / "val_nmi_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    df = aggregate_val_nmi_by_lab(args.result_root, load_manifest(args.manifest))
    print(f"Wrote {args.result_root / 'val_nmi_by_lab.csv'} ({len(df)} rows)")


if __name__ == "__main__":
    main()
