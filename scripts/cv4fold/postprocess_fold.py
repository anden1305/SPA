#!/usr/bin/env python3
"""Post-train per-mouse splits for all runs in a cv4fold result folder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.cv4fold.aggregate_val_nmi_by_lab import aggregate_val_nmi_by_lab
from scripts.cv4fold.manifest_utils import DEFAULT_MANIFEST, load_manifest
from scripts.cv4fold.split_per_mouse import split_per_mouse, subject_id_map_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root",
        type=Path,
        required=True,
        help="Directory containing config.json and run subdirs 1/, 2/, 3/",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Manifest for participant_id -> lab mapping",
    )
    parser.add_argument(
        "--skip-lab-aggregate",
        action="store_true",
        help="Only split per_mouse; do not write val_nmi_by_lab.csv",
    )
    args = parser.parse_args()
    root = args.result_root
    config_path = root / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    id_map = subject_id_map_from_config(config_path)
    cfg = json.loads(config_path.read_text(encoding="utf-8"))

    for run_num in (1, 2, 3):
        plots = root / str(run_num) / "plots"
        if plots.is_dir():
            split_per_mouse(plots, root / "per_mouse" / f"run_{run_num}", id_map, config=cfg)

    if not args.skip_lab_aggregate:
        manifest = load_manifest(args.manifest)
        aggregate_val_nmi_by_lab(root, manifest)
        print(f"Wrote lab aggregates under {root}")


if __name__ == "__main__":
    main()
