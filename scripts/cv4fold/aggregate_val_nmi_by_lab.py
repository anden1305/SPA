#!/usr/bin/env python3
"""Write small CSV/JSON summaries from post-train metrics.txt.

Default uses pooled NMI only (kilobytes). Per-mouse / per-lab breakdown requires
``--per-mouse-metrics`` and an existing ``results.npz`` from training.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from scripts.cv4fold.manifest_utils import DEFAULT_MANIFEST, load_manifest
from scripts.cv4fold.metrics_paths import metrics_path
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


def _labs_for_val_mice(val_ids: list[str], lab_map: dict[str, str]) -> list[str]:
    return sorted({lab_map[pid] for pid in val_ids if pid in lab_map})


def aggregate_run_from_per_mouse(
    per_mouse_run_dir: Path,
    val_ids: list[str],
    lab_map: dict[str, str],
) -> list[dict]:
    by_lab: dict[str, list[float]] = defaultdict(list)
    for pid in val_ids:
        metrics_file = per_mouse_run_dir / pid / "metrics.json"
        if not metrics_file.exists():
            continue
        payload = json.loads(metrics_file.read_text(encoding="utf-8"))
        lab = lab_map.get(pid)
        if lab is None:
            continue
        by_lab[lab].append(float(payload["nmi"]))
    return [
        {
            "lab": lab,
            "nmi_macro": float(sum(nmis) / len(nmis)),
            "n_mice": len(nmis),
        }
        for lab, nmis in sorted(by_lab.items())
    ]


def aggregate_val_nmi_by_lab(
    result_root: Path,
    manifest: dict | None = None,
    *,
    use_per_mouse: bool = False,
) -> pd.DataFrame:
    """Write ``val_nmi_by_lab.csv`` and ``val_nmi_summary.json`` under ``result_root``."""
    config_path = result_root / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = manifest or load_manifest()
    val_ids = _val_mouse_ids(cfg)
    lab_map = _lab_for_mouse(manifest)
    val_labs = _labs_for_val_mice(val_ids, lab_map)

    rows: list[dict] = []
    summary_runs: list[dict] = []
    for run_num in (1, 2, 3):
        metrics_file = metrics_path(result_root, run_num)
        pooled = parse_nmi(metrics_file) if metrics_file else None
        per_mouse_dir = result_root / "per_mouse" / f"run_{run_num}"

        if use_per_mouse and per_mouse_dir.is_dir():
            for lab_row in aggregate_run_from_per_mouse(per_mouse_dir, val_ids, lab_map):
                rows.append({"run": run_num, "nmi_pooled": pooled, **lab_row})
        elif pooled is not None and val_labs:
            for lab in val_labs:
                rows.append(
                    {
                        "run": run_num,
                        "lab": lab,
                        "nmi_macro": pooled,
                        "n_mice": len([v for v in val_ids if lab_map.get(v) == lab]),
                        "nmi_pooled": pooled,
                        "source": "pooled_metrics_txt",
                    }
                )
        summary_runs.append({"run": run_num, "nmi_pooled": pooled})

    df = pd.DataFrame(rows)
    df.to_csv(result_root / "val_nmi_by_lab.csv", index=False)

    best = max(
        (r for r in summary_runs if r["nmi_pooled"] is not None),
        key=lambda r: r["nmi_pooled"],
        default=None,
    )
    summary = {
        "val_mouse_ids": val_ids,
        "val_labs": val_labs,
        "runs": summary_runs,
        "best_run": best,
        "note": (
            "nmi_macro equals pooled holdout NMI unless --per-mouse-metrics was used"
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
    parser.add_argument(
        "--per-mouse-metrics",
        action="store_true",
        help="Use per_mouse/*/metrics.json instead of pooled metrics.txt",
    )
    args = parser.parse_args()
    df = aggregate_val_nmi_by_lab(
        args.result_root,
        load_manifest(args.manifest),
        use_per_mouse=args.per_mouse_metrics,
    )
    print(f"Wrote {args.result_root / 'val_nmi_by_lab.csv'} ({len(df)} rows)")


if __name__ == "__main__":
    main()
