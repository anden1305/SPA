#!/usr/bin/env python3
"""Lean post-train summaries for one cv4fold result folder.

Default (~few KB): best-of-3 selection + CSV from ``metrics.txt`` only.
Optional per-mouse split reads ``results.npz`` once; never copies it by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.cv4fold.aggregate_val_nmi_by_lab import aggregate_val_nmi_by_lab
from scripts.cv4fold.manifest_utils import DEFAULT_MANIFEST, load_manifest
from scripts.cv4fold.select_best_vae_run import select_best_run
from scripts.cv4fold.split_per_mouse import split_per_mouse, subject_id_map_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root",
        type=Path,
        required=True,
        help="Timestamped dir with config.json and plots/{1,2,3}/metrics.txt",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--per-mouse-metrics",
        action="store_true",
        help="Split results.npz → per_mouse/run_*/sub-XXX/metrics.json only (~100 B/mouse)",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="With --per-mouse-metrics, also write predictions.npz (large)",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="With --per-mouse-metrics, write minimal PCA png per mouse",
    )
    parser.add_argument(
        "--skip-lab-aggregate",
        action="store_true",
        help="Skip val_nmi_by_lab.csv / val_nmi_summary.json",
    )
    args = parser.parse_args()
    root = args.result_root
    config_path = root / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    if args.per_mouse_metrics:
        id_map = subject_id_map_from_config(config_path)
        for run_num in (1, 2, 3):
            split_per_mouse(
                root,
                run_num,
                root / "per_mouse" / f"run_{run_num}",
                id_map,
                save_predictions=args.save_predictions,
                save_plots=args.save_plots,
            )

    selection = select_best_run(root)
    (root / "best_run.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(f"Best run: seed {selection['selected_run']} NMI={selection['nmi']:.4f}")

    if not args.skip_lab_aggregate:
        manifest = load_manifest(args.manifest)
        aggregate_val_nmi_by_lab(
            root,
            manifest,
            use_per_mouse=args.per_mouse_metrics,
        )
        print(f"Wrote aggregates under {root}")


if __name__ == "__main__":
    main()
