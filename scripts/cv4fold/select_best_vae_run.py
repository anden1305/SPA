#!/usr/bin/env python3
"""Select best of 3 VAE runs by post-train NMI in metrics.txt."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from scripts.cv4fold.metrics_paths import metrics_path

NMI_RE = re.compile(r"NMI:\s*([0-9.eE+-]+)")


def parse_nmi(metrics_path: Path) -> float | None:
    if not metrics_path.exists():
        return None
    text = metrics_path.read_text(encoding="utf-8")
    m = NMI_RE.search(text)
    return float(m.group(1)) if m else None


def collect_run_nmis(result_root: Path) -> list[tuple[int, float, Path]]:
    candidates: list[tuple[int, float, Path]] = []
    for run_num in (1, 2, 3):
        path = metrics_path(result_root, run_num)
        if path is None:
            continue
        nmi = parse_nmi(path)
        if nmi is not None:
            candidates.append((run_num, nmi, path))
    return candidates


def select_best_run(fold_result_dir: Path) -> dict:
    """``fold_result_dir`` is a timestamped run dir or contains one."""
    resolved = fold_result_dir
    candidates = collect_run_nmis(resolved)
    if not candidates:
        for run_name_dir in sorted(fold_result_dir.iterdir()):
            if not run_name_dir.is_dir():
                continue
            candidates = collect_run_nmis(run_name_dir)
            if candidates:
                resolved = run_name_dir
                break

    if not candidates:
        raise FileNotFoundError(f"No metrics.txt found under {fold_result_dir}")

    best = max(candidates, key=lambda t: t[1])
    return {
        "result_root": str(resolved),
        "selected_run": best[0],
        "nmi": best[1],
        "metrics_path": str(best[2]),
        "all_runs": {str(r): n for r, n, _ in candidates},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fold-dir",
        type=Path,
        required=True,
        help="Timestamped run dir or parent containing run_name_* subdirs",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Append entry to JSON (default: results/cv4fold/selection/best_runs.json)",
    )
    parser.add_argument("--key", type=str, default=None, help="JSON key for this fold")
    args = parser.parse_args()

    selection = select_best_run(args.fold_dir)
    print(json.dumps(selection, indent=2))

    out_path = args.out or Path("results/cv4fold/selection/best_runs.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    store: dict = {}
    if out_path.exists():
        store = json.loads(out_path.read_text(encoding="utf-8"))
    key = args.key or str(args.fold_dir)
    store[key] = selection
    out_path.write_text(json.dumps(store, indent=2), encoding="utf-8")
    print(f"Updated {out_path}")


if __name__ == "__main__":
    main()
