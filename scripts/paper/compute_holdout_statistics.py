#!/usr/bin/env python3
"""Paired inferential tests for joint holdout ladder (best-of-3 and seed-mean)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as stats
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

from scripts.cv4fold.select_best_vae_run import collect_run_nmis
from scripts.paper.summarize_holdout_ladder import DEFAULT_ROOTS

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO / "paper/overleaf/tables/holdout_ladder.csv"
FOLDS = ("1", "2", "3", "4")
COMPARE = (
    ("cgmvae_locked", "chmmgmvae_locked", "cGMVAE", "cHMM--GMVAE"),
)


def _load_joint_best(csv_path: Path, model: str) -> dict[str, float]:
    out: dict[str, float] = {}
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["scope"] == "joint" and row["lab"] == "all" and row["model"] == model:
                out[row["fold"]] = float(row["nmi"])
    return out


def _seed_means(fold: str, model: str) -> list[float]:
    root = REPO / "results/cv4fold/joint_holdout" / f"fold_{fold}" / model
    if not root.is_dir():
        return []
    run_dirs = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    if not run_dirs:
        return []
    runs = collect_run_nmis(run_dirs[0])
    return [n for _, n, _ in runs]


def _bootstrap_mean_ci(deltas: np.ndarray, n_boot: int = 10000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = deltas.size
    if n == 0:
        return float("nan"), float("nan")
    boots = np.array([rng.choice(deltas, size=n, replace=True).mean() for _ in range(n_boot)])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _sign_test_positive(deltas: np.ndarray) -> float:
    """Two-sided exact binomial test for median > 0 (no zero deltas)."""
    k = int((deltas > 0).sum())
    n = int(deltas.size)
    if n == 0:
        return float("nan")
    # scipy binomtest two-sided
    return float(sp_stats.binomtest(k, n, 0.5, alternative="two-sided").pvalue)


def _paired_tests(a: np.ndarray, b: np.ndarray) -> dict:
    if a.size != b.size or a.size < 2:
        return {}
    delta = b - a
    out: dict[str, float | int | list[float]] = {
        "n_pairs": int(a.size),
        "delta_mean": float(delta.mean()),
        "delta_median": float(np.median(delta)),
        "delta_per_pair": [float(x) for x in delta],
        "ci95_low": float("nan"),
        "ci95_high": float("nan"),
    }
    lo, hi = _bootstrap_mean_ci(delta)
    out["ci95_low"] = lo
    out["ci95_high"] = hi
    out["sign_test_p"] = _sign_test_positive(delta)
    try:
        t_res = sp_stats.ttest_rel(b, a, nan_policy="omit")
        out["paired_t_p"] = float(t_res.pvalue) if t_res.pvalue is not None else float("nan")
        out["paired_t_stat"] = float(t_res.statistic) if t_res.statistic is not None else float("nan")
    except Exception:
        out["paired_t_p"] = float("nan")
        out["paired_t_stat"] = float("nan")
    try:
        w_res = sp_stats.wilcoxon(b, a, zero_method="wilcox", alternative="greater", method="auto")
        out["wilcoxon_p"] = float(w_res.pvalue)
        out["wilcoxon_stat"] = float(w_res.statistic)
    except ValueError:
        out["wilcoxon_p"] = float("nan")
        out["wilcoxon_stat"] = float("nan")
    return out


def _load_per_mouse_fold4(model: str, seed: int) -> dict[str, float]:
    base = REPO / f"results/cv4fold/paper_figures/per_mouse/fold_4/{model}"
    root = base / f"seed_{seed}"
    if not root.is_dir():
        alt = sorted((p for p in base.glob("seed_*") if p.is_dir()), key=lambda p: p.name)
        root = alt[0] if alt else root
    out: dict[str, float] = {}
    if not root.is_dir():
        return out
    for mouse_dir in root.iterdir():
        mf = mouse_dir / "metrics.json"
        if mf.is_file():
            payload = json.loads(mf.read_text(encoding="utf-8"))
            out[str(payload.get("participant_id", mouse_dir.name))] = float(payload["nmi"])
    return out


def compute_all(csv_path: Path) -> dict:
    results: dict = {"comparisons": []}
    for model_a, model_b, label_a, label_b in COMPARE:
        best_a = _load_joint_best(csv_path, model_a)
        best_b = _load_joint_best(csv_path, model_b)
        folds = [f for f in FOLDS if f in best_a and f in best_b]
        a = np.array([best_a[f] for f in folds])
        b = np.array([best_b[f] for f in folds])
        entry = {
            "comparison": f"{label_b} vs {label_a}",
            "metric": "best_of_three_per_fold",
            "folds": folds,
            "values_a": {f: best_a[f] for f in folds},
            "values_b": {f: best_b[f] for f in folds},
            **_paired_tests(a, b),
        }
        results["comparisons"].append(entry)

        # Seed-mean paired (folds with >=2 seeds on both sides)
        seed_fold_a: list[float] = []
        seed_fold_b: list[float] = []
        seed_folds: list[str] = []
        for f in FOLDS:
            sa = _seed_means(f, model_a)
            sb = _seed_means(f, model_b)
            if len(sa) >= 2 and len(sb) >= 2:
                seed_folds.append(f)
                seed_fold_a.append(stats.mean(sa))
                seed_fold_b.append(stats.mean(sb))
        if len(seed_folds) >= 2:
            results["comparisons"].append(
                {
                    "comparison": f"{label_b} vs {label_a}",
                    "metric": "seed_mean_per_fold",
                    "folds": seed_folds,
                    **_paired_tests(np.array(seed_fold_a), np.array(seed_fold_b)),
                }
            )

    # Fold-4 per-mouse (best seeds from ladder csv)
    cgm_seed = int(
        next(r["selected_run"] for r in csv.DictReader(csv_path.open()) if r["scope"] == "joint" and r["fold"] == "4" and r["model"] == "cgmvae_locked")
    )
    chmm_seed = int(
        next(r["selected_run"] for r in csv.DictReader(csv_path.open()) if r["scope"] == "joint" and r["fold"] == "4" and r["model"] == "chmmgmvae_locked")
    )
    pm_cgm = _load_per_mouse_fold4("cgmvae_locked", cgm_seed)
    pm_chmm = _load_per_mouse_fold4("chmmgmvae_locked", chmm_seed)
    mice = sorted(set(pm_cgm) & set(pm_chmm))
    if len(mice) >= 2:
        a = np.array([pm_cgm[m] for m in mice])
        b = np.array([pm_chmm[m] for m in mice])
        results["comparisons"].append(
            {
                "comparison": "cHMM--GMVAE vs cGMVAE",
                "metric": "per_mouse_fold4",
                "mice": mice,
                **_paired_tests(a, b),
            }
        )
    return results


def _fmt_p(p: float) -> str:
    if p != p:
        return "---"
    if p < 0.001:
        return "$<0.001$"
    return f"{p:.3f}"


def write_tex(path: Path, summary: dict) -> None:
    primary = next(c for c in summary["comparisons"] if c["metric"] == "best_of_three_per_fold")
    seed_mean = next((c for c in summary["comparisons"] if c["metric"] == "seed_mean_per_fold"), None)
    per_mouse = next((c for c in summary["comparisons"] if c["metric"] == "per_mouse_fold4"), None)

    lines = [
        "% Auto-generated by scripts/paper/compute_holdout_statistics.py",
        "\\begin{tabular}{|l|l|r|r|r|}",
        "\\hline",
        "\\textbf{Comparison} & \\textbf{Unit} & \\textbf{$\\Delta$ mean} & "
        "\\textbf{95\\% CI} & \\textbf{$p$ (Wilcoxon)} \\\\",
        "\\thickhline",
        f"Primary & {primary['n_pairs']} folds & {primary['delta_mean']:.3f} & "
        f"[{primary['ci95_low']:.3f}, {primary['ci95_high']:.3f}] & {_fmt_p(primary['wilcoxon_p'])} \\\\",
    ]
    if seed_mean:
        lines.append(
            f"Sensitivity & {seed_mean['n_pairs']} folds (seed mean) & {seed_mean['delta_mean']:.3f} & "
            f"[{seed_mean['ci95_low']:.3f}, {seed_mean['ci95_high']:.3f}] & {_fmt_p(seed_mean['wilcoxon_p'])} \\\\"
        )
    if per_mouse:
        lines.append(
            f"Fold~4 mice & {per_mouse['n_pairs']} mice & {per_mouse['delta_mean']:.3f} & "
            f"[{per_mouse['ci95_low']:.3f}, {per_mouse['ci95_high']:.3f}] & {_fmt_p(per_mouse['wilcoxon_p'])} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=REPO / "docs/paper/figures/supplementary/S12_holdout_statistics.json",
    )
    parser.add_argument(
        "--out-tex",
        type=Path,
        default=REPO / "docs/paper/assets/tables/S12_holdout_statistics.tex",
    )
    args = parser.parse_args()

    summary = compute_all(args.csv)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_tex(args.out_tex, summary)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
