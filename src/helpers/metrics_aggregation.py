"""Utilities to aggregate per-run metrics into summary statistics.

Keeps logic out of Validator to avoid clutter and duplication.
"""
from __future__ import annotations

import statistics
from typing import Any, Iterable, Mapping, MutableMapping


def _mean_std(values: Iterable[float]) -> tuple[float | None, float | None]:
    nums = [float(v) for v in values if isinstance(v, (int, float))]
    if not nums:
        return None, None
    try:
        mean_v = statistics.mean(nums)
        std_v = statistics.stdev(nums) if len(nums) > 1 else 0.0
        return float(mean_v), float(std_v)
    except Exception:
        return None, None


def summarize_metrics(
    per_run: Mapping[str, Any],
    *,
    n_runs: int | None = None,
    attach_to: MutableMapping[str, Any] | None = None,
    namespace: str = "summary",
    style: str = "nested",
    best_by: str | None = "loss",
) -> dict[str, Any]:
    """Compute mean/std for each metric in a per-run mapping and annotate best run.

    Arguments
    - per_run: mapping of metric_name -> either a Mapping[run_number, value] or an Iterable[value].
    - n_runs: optional number of runs to include in summary.
    - attach_to: when provided, the computed summary will be attached in-place to this mapping.
    - namespace: key under which to store the nested summary when style == 'nested'. Default '_summary'.
    - style: one of 'nested' (default) or 'flat'.
    - best_by: metric name to determine the best run by lowest value. Defaults to 'loss'.

    Returns
    - A dict of summary statistics. When style == 'flat', keys are '<metric>_mean'/'<metric>_std'.
      When style == 'nested', returns {'n_runs': ..., '<metric>': {'mean': ..., 'std': ...}}.
      If attach_to is provided, the same summary is also attached to that mapping in-place.
    """
    # First collect raw numeric summaries
    pairs: dict[str, tuple[float, float]] = {}
    for name, data in per_run.items():
        if isinstance(data, Mapping):
            values = list(data.values())
        else:
            try:
                values = list(data)  # type: ignore[arg-type]
            except TypeError:
                # Single value; skip
                continue
        mean_v, std_v = _mean_std(values)
        if mean_v is None:
            continue
        pairs[name] = (mean_v, std_v)

    # Determine best run based on a given metric (lowest value)
    best_run: Any | None = None
    best_metrics: dict[str, Any] = {}
    if best_by is not None:
        try:
            data = per_run.get(best_by, None)  # type: ignore[assignment]
            if isinstance(data, Mapping) and data:
                # choose min numeric value (ignore non-numeric)
                numeric_items = [(k, float(v)) for k, v in data.items() if isinstance(v, (int, float))]
                if numeric_items:
                    best_run = min(numeric_items, key=lambda kv: kv[1])[0]
                    # collect metric values at best_run for all mapping metrics
                    for name, d in per_run.items():
                        if isinstance(d, Mapping):
                            best_metrics[name] = d.get(best_run, None)
        except Exception:
            # If anything goes wrong, we simply skip best selection
            best_run = None

    # Build summary in requested style
    if style == "flat":
        summary: dict[str, Any] = {}
        if n_runs is not None:
            summary["n_runs"] = n_runs
        for name, (m, s) in pairs.items():
            summary[f"{name}_mean"] = m
            summary[f"{name}_std"] = s
        if best_run is not None:
            summary["best_run"] = best_run
            for k, v in best_metrics.items():
                summary[f"best_{k}"] = v
    else:  # nested
        summary = {}
        if n_runs is not None:
            summary["n_runs"] = n_runs
        for name, (m, s) in pairs.items():
            summary[name] = {"mean": m, "std": s}
        if best_run is not None:
            summary["best"] = {"run": best_run, "metrics": best_metrics}

    # Optionally attach to provided mapping without altering existing per-run structures
    if attach_to is not None:
        if style == "flat":
            # flat style merges at top-level
            attach_to.update(summary)
        else:
            # nested style stored under a reserved namespace
            attach_to[namespace] = summary

    return summary
