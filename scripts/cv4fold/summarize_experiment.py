#!/usr/bin/env python3
"""Scan cv4fold result trees and write one small summary CSV (no npz I/O)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.cv4fold.select_best_vae_run import select_best_run


def find_result_roots(search_root: Path) -> list[Path]:
    roots: list[Path] = []
    for config_path in sorted(search_root.rglob("config.json")):
        parent = config_path.parent
        if (parent / "plots").is_dir() or (parent / "1").is_dir():
            roots.append(parent)
    return roots


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--search-root",
        type=Path,
        required=True,
        help="e.g. results/cv4fold/per_lab_holdout/lab_2/fold_4/cgmvae",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Default: <search-root>/summary.csv",
    )
    args = parser.parse_args()
    rows: list[dict] = []
    for root in find_result_roots(args.search_root):
        try:
            sel = select_best_run(root)
        except FileNotFoundError:
            continue
        rel = root.relative_to(args.search_root) if root.is_relative_to(args.search_root) else root
        rows.append(
            {
                "result_dir": str(rel),
                "best_run": sel["selected_run"],
                "best_nmi": sel["nmi"],
                "run_1": sel["all_runs"].get("1"),
                "run_2": sel["all_runs"].get("2"),
                "run_3": sel["all_runs"].get("3"),
            }
        )

    out = args.out or (args.search_root / "summary.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
