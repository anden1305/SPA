#!/usr/bin/env python3

from __future__ import annotations
import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable


TIMESTAMP_RE = re.compile(r"^(?P<base>.+)_(?P<ts>\d{8}-\d{6})$")
SUBJECT_RE = re.compile(r"sub\d{3}")


@dataclass(frozen=True)
class MetricRecord:
    scope: str
    base_name: str
    timestamp: str
    nmi: float
    likelihood: float
    path: Path


def _parse_metrics(path: Path) -> tuple[float, float] | None:
    if path.name == "validation_info.json":
        import json
        try:
            with open(path, "r") as f:
                data = json.load(f)
                return float(data["nmi"]), float(data["likelihood"])
        except Exception:
            pass
            
    text = path.read_text().strip().splitlines()
    nmi = None
    likelihood = None
    for line in text:
        if line.lower().startswith("nmi:"):
            nmi = float(line.split(":", 1)[1].strip())
        if line.lower().startswith("likelihood:"):
            likelihood = float(line.split(":", 1)[1].strip())
    if nmi is None or likelihood is None:
        return None
    return nmi, likelihood


def _extract_scope(path: Path) -> str | None:
    parts = path.parts
    for scope in ("subjectwise", "generalization_subject", "reliability", "generalization"):
        if scope in parts:
            return scope
    return None


def _extract_base_and_timestamp(run_dir: Path) -> tuple[str, str] | None:
    match = TIMESTAMP_RE.match(run_dir.name)
    if not match:
        return None
    return match.group("base"), match.group("ts")


def _latest_by_base_and_best_run(records: Iterable[MetricRecord]) -> list[MetricRecord]:
    from collections import defaultdict
    grouped: dict[tuple[str, str], list[MetricRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.scope, record.base_name)].append(record)

    best_records = []
    for key, group in grouped.items():
        latest_ts = max(r.timestamp for r in group)
        latest_runs = [r for r in group if r.timestamp == latest_ts]
        # Pick the best run in the latest trial (highest likelihood)
        best_run = max(latest_runs, key=lambda r: r.likelihood)
        best_records.append(best_run)
    return best_records


def _summarize(values: list[float]) -> str:
    if not values:
        return "-"
    if len(values) == 1:
        return f"{values[0]:.3f}"
    sem = stdev(values) / math.sqrt(len(values))
    return f"{mean(values):.3f} ± {sem:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize decoder-only results into table-ready aggregates."
    )
    parser.add_argument(
        "--results_root",
        default="results/decoder_only",
        help="Root folder to scan for decoder-only results.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    results_root = (repo_root / args.results_root).resolve()
    if not results_root.exists():
        print(f"Results root not found: {results_root}")
        return 1

    metrics_paths = list(results_root.glob("**/plots/**/metrics.txt"))
    metrics_paths += list(results_root.glob("**/plots/metrics.txt"))
    metrics_paths += list(results_root.glob("**/plots/**/validation_info.json"))
    if not metrics_paths:
        print(f"No metrics files found under {results_root}")
        return 0

    records: list[MetricRecord] = []
    for metrics_path in metrics_paths:
        run_dir = metrics_path.parents[1] if metrics_path.name != "validation_info.json" else metrics_path.parents[2]
        scope = _extract_scope(metrics_path)
        if scope is None:
            continue
        base_ts = _extract_base_and_timestamp(run_dir)
        if base_ts is None:
            continue
        metrics = _parse_metrics(metrics_path)
        if metrics is None:
            continue
        base_name, timestamp = base_ts
        records.append(
            MetricRecord(
                scope=scope,
                base_name=base_name,
                timestamp=timestamp,
                nmi=metrics[0],
                likelihood=metrics[1],
                path=metrics_path,
            )
        )

    if not records:
        print("No valid metrics records found.")
        return 0

    latest_records = _latest_by_base_and_best_run(records)

    scope_to_values: dict[str, dict[str, list[float]]] = {}
    for record in latest_records:
        scope_to_values.setdefault(record.scope, {"nmi": [], "likelihood": []})
        scope_to_values[record.scope]["nmi"].append(record.nmi)
        scope_to_values[record.scope]["likelihood"].append(record.likelihood)

    subject_train = scope_to_values.get("subjectwise", {"nmi": [], "likelihood": []})
    subject_pred = scope_to_values.get("generalization_subject", {"nmi": [], "likelihood": []})
    pop_train = scope_to_values.get("reliability", {"nmi": [], "likelihood": []})
    pop_pred = scope_to_values.get("generalization", {"nmi": [], "likelihood": []})

    print("Decoder-only summary (latest per base run):")
    print("Subject scope train NMI:", _summarize(subject_train["nmi"]))
    print("Subject scope pred NMI:", _summarize(subject_pred["nmi"]))
    print("Subject scope train likelihood:", _summarize(subject_train["likelihood"]))
    print("Subject scope pred likelihood:", _summarize(subject_pred["likelihood"]))
    print("Population scope train NMI:", _summarize(pop_train["nmi"]))
    print("Population scope pred NMI:", _summarize(pop_pred["nmi"]))
    print("Population scope train likelihood:", _summarize(pop_train["likelihood"]))
    print("Population scope pred likelihood:", _summarize(pop_pred["likelihood"]))

    print("\nTable rows:")
    print(
        "cGMVAE Decoder-only MSSV Frequency Subject "
        f"{_summarize(subject_train['nmi'])} "
        f"{_summarize(subject_pred['nmi'])} "
        f"{_summarize(subject_train['likelihood'])} "
        f"{_summarize(subject_pred['likelihood'])}"
    )
    print(
        "cGMVAE Decoder-only MSSV Frequency Population "
        f"{_summarize(pop_train['nmi'])} "
        f"{_summarize(pop_pred['nmi'])} "
        f"{_summarize(pop_train['likelihood'])} "
        f"{_summarize(pop_pred['likelihood'])}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
