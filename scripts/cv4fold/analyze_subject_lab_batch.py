#!/usr/bin/env python3
"""Summarize subject_lab_tune_winners results and recommend follow-up."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from scripts.cv4fold.select_best_vae_run import parse_nmi
from scripts.cv4fold.summarize_subject_lab_cv import FOLDS, LABS, MODELS, _latest_timestamp_dir, collect_rows

REPO_ROOT = Path(__file__).resolve().parents[2]
SUBJECT_LAB_ROOT = REPO_ROOT / "results/cv4fold/subject_lab_tune_winners"
REPORT_PATH = SUBJECT_LAB_ROOT / "analysis_report.md"
SUMMARY_PATH = SUBJECT_LAB_ROOT / "summary.csv"
FAIL_NMI_THRESHOLD = 0.05


@dataclass
class BatchStatus:
    n_complete_folds: dict[str, int]
    mean_best_nmi: dict[str, float]
    all_failed: bool
    pending_jobs_note: str


def _count_metrics(root: Path) -> dict[str, set[int]]:
    """model -> set of folds with a completed timestamp dir (3 runs x metrics.txt)."""
    done: dict[str, set[int]] = {m: set() for m in MODELS}
    for model in MODELS:
        for fold in FOLDS:
            fold_dir = root / model / "joint" / f"fold_{fold}"
            latest = _latest_timestamp_dir(fold_dir)
            if latest is None:
                continue
            ok = all((latest / str(r) / "plots" / "metrics.txt").exists() for r in (1, 2, 3))
            if ok:
                done[model].add(fold)
    return done


def _best_nmi_per_fold(root: Path, model: str, fold: int) -> float | None:
    fold_dir = root / model / "joint" / f"fold_{fold}"
    latest = _latest_timestamp_dir(fold_dir)
    if latest is None:
        return None
    best: float | None = None
    for r in (1, 2, 3):
        nmi = parse_nmi(latest / str(r) / "plots" / "metrics.txt")
        if nmi is not None and (best is None or nmi > best):
            best = nmi
    return best


def assess_batch(root: Path, *, fail_threshold: float = FAIL_NMI_THRESHOLD) -> BatchStatus:
    done = _count_metrics(root)
    mean_best: dict[str, float] = {}
    for model in MODELS:
        nmis = [_best_nmi_per_fold(root, model, f) for f in FOLDS]
        vals = [n for n in nmis if n is not None]
        mean_best[model] = float(sum(vals) / len(vals)) if vals else float("nan")
    cgmvae_mean = mean_best.get("cgmvae", float("nan"))
    all_failed = len(done.get("cgmvae", set())) >= 1 and (
        cgmvae_mean != cgmvae_mean or cgmvae_mean < fail_threshold
    )
    return BatchStatus(
        n_complete_folds={m: len(done[m]) for m in MODELS},
        mean_best_nmi=mean_best,
        all_failed=all_failed,
        pending_jobs_note="",
    )


def write_report(
    root: Path,
    status: BatchStatus,
    df: pd.DataFrame,
    done_folds: dict[str, set[int]],
) -> None:
    lines = [
        "# subject_lab_tune_winners — batch analysis",
        "",
        f"Results root: `{root}`",
        "",
        "## Completion",
        "",
        f"- **cgmvae** folds complete: {sorted(done_folds.get('cgmvae', set()))} "
        f"({status.n_complete_folds.get('cgmvae', 0)}/4)",
        f"- **chmmgmvae** folds complete: {sorted(done_folds.get('chmmgmvae', set()))} "
        f"({status.n_complete_folds.get('chmmgmvae', 0)}/4)",
        "",
        "## Mean best prior NMI per model (across completed folds)",
        "",
    ]
    for model in MODELS:
        v = status.mean_best_nmi.get(model, float("nan"))
        lines.append(f"- **{model}**: {v:.4f}" if v == v else f"- **{model}**: n/a")
    lines.extend(["", "## Per-run table", ""])
    if df.empty:
        lines.append("(no rows yet — batch still running or postprocess missing)")
    else:
        try:
            lines.append(df.to_markdown(index=False))
        except ImportError:
            lines.append("```\n" + df.to_csv(index=False) + "\n```")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
        ]
    )
    if status.all_failed and status.n_complete_folds.get("cgmvae", 0) >= 1:
        lines.append(
            "- **subject_lab + tune-winner hyperparams** shows **prior NMI ≈ 0** on completed "
            "cgmvae folds → likely collapse / wrong conditioning transfer, not a queue issue."
        )
        lines.append(
            "- Tune winners were selected with **`conditioning_source: subject`** (in-sample); "
            "this holdout batch is a different experiment."
        )
        lines.append(
            "- **Reasonable follow-up:** `subject_tune_winners` ablation (same hyperparams, "
            "`subject` only) on **fold 4** first (2 bsub jobs), before re-running all 8 folds."
        )
    elif not status.all_failed:
        lines.append("- Some folds show usable NMI; compare per-lab columns in `summary.csv`.")
    lines.extend(
        [
            "",
            "## Commands",
            "",
            "```bash",
            "cd /work3/s204070/SPA",
            "PYTHONPATH=. python3 scripts/cv4fold/analyze_subject_lab_batch.py",
            "PYTHONPATH=. python3 scripts/cv4fold/analyze_subject_lab_batch.py --submit-followup",
            "```",
            "",
            "Follow-up submit (fold 4, subject-only ablation):",
            "`bash hpc/submit/cv4fold/submit_subject_tune_winners_fold4.sh`",
            "",
        ]
    )
    root.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=SUBJECT_LAB_ROOT)
    parser.add_argument(
        "--submit-followup",
        action="store_true",
        help="If cgmvae looks failed, generate configs and bsub fold-4 subject_tune_winners (2 jobs)",
    )
    parser.add_argument(
        "--fail-nmi-threshold",
        type=float,
        default=FAIL_NMI_THRESHOLD,
        help="Mean best prior NMI below this triggers follow-up recommendation",
    )
    args = parser.parse_args()

    root = args.results_root
    df = summarize_subject_lab_style(root)
    done_folds = _count_metrics(root)
    status = assess_batch(root, fail_threshold=args.fail_nmi_threshold)
    write_report(root, status, df, done_folds)
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {SUMMARY_PATH}")

    cgmvae_done = status.n_complete_folds.get("cgmvae", 0)
    cgmvae_mean = status.mean_best_nmi.get("cgmvae", float("nan"))
    should_followup = cgmvae_done >= 1 and (
        cgmvae_mean != cgmvae_mean or cgmvae_mean < args.fail_nmi_threshold
    )

    if should_followup:
        print(
            f"Recommendation: subject_lab batch weak (cgmvae mean best NMI={cgmvae_mean:.4f} "
            f"over {cgmvae_done} fold(s)). Consider canceling remaining subject_lab jobs "
            "and running subject_tune_winners fold-4 ablation."
        )
    if args.submit_followup and should_followup:
        import subprocess

        subprocess.run(
            ["python3", "scripts/cv4fold/generate_configs.py", "--phase", "subject_tune_winners", "--folds", "4"],
            cwd=REPO_ROOT,
            check=True,
        )
        subprocess.run(
            ["bash", "hpc/submit/cv4fold/submit_subject_tune_winners_fold4.sh"],
            cwd=REPO_ROOT,
            check=True,
        )
        print("Submitted subject_tune_winners fold 4 (cgmvae + chmmgmvae).")
    elif args.submit_followup:
        print("Skip follow-up submit: batch does not meet failure criteria yet.")
    return 0


def summarize_subject_lab_style(root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for model in MODELS:
        for fold in FOLDS:
            rows.extend(collect_rows(model, fold, _need_latest(root, model, fold)))
    out = pd.DataFrame(rows)
    root.mkdir(parents=True, exist_ok=True)
    out.to_csv(SUMMARY_PATH, index=False)
    return out


def _need_latest(root: Path, model: str, fold: int) -> Path:
    latest = _latest_timestamp_dir(root / model / "joint" / f"fold_{fold}")
    if latest is None:
        return root / "_missing_"
    return latest


if __name__ == "__main__":
    raise SystemExit(main())
