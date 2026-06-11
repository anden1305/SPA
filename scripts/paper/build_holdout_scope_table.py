#!/usr/bin/env python3
"""Build joint vs within-lab holdout LaTeX tables (NMI + macro accuracy).

Joint scope: metrics on held-out mice from each lab only (split from pooled results.npz).
Within-lab scope: metrics on the same lab's held-out mice when training uses that lab only.
Uncertainty: mean over folds of best-of-3 seeds (VAE) or single run (HMM raw).
Table cells report the mean only; seed selection is described in the caption.
"""

from __future__ import annotations

import argparse
import csv
import statistics as stats
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.cv4fold.collect_hmm_raw_holdout import collect_joint_per_lab_rows, collect_rows as collect_hmm_rows
from scripts.cv4fold.manifest_utils import load_manifest
from scripts.cv4fold.metrics_paths import results_npz_path
from scripts.paper.scrape_holdout_accuracy import macro_map_accuracy
from scripts.paper.summarize_holdout_ladder import (
    DEFAULT_ROOTS,
    MODEL_LABEL,
    Row,
    collect_rows,
)
from src.helpers.nmi import calculate_nmi

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO / "paper/overleaf/tables/holdout_ladder.csv"

MODELS = ("hmm_raw", "cgmvae_locked", "hmmgmvae_locked", "chmmgmvae_locked")
MODEL_LABEL_TABLE = {
    "hmm_raw": "HMM (raw)",
    "cgmvae_locked": "cGMVAE",
    "hmmgmvae_locked": "HMMGMVAE",
    "chmmgmvae_locked": "cHMM--GMVAE",
}
LABS = ("lab_2", "lab_3", "lab_5")
LAB_LABEL = {"lab_2": "Lab 2", "lab_3": "Lab 3", "lab_5": "Lab 5", "all": "Joint"}
FOLDS = ("1", "2", "3", "4")
VAE_MODELS = ("cgmvae_locked", "hmmgmvae_locked", "chmmgmvae_locked")


def _load(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _mean_range(values: list[float]) -> tuple[float, float, float, int]:
    if not values:
        return float("nan"), float("nan"), float("nan"), 0
    return stats.mean(values), min(values), max(values), len(values)


def _fmt_mean(mean: float, n: int, *, bold: bool = False) -> str:
    if n == 0 or mean != mean:
        return "---"
    text = f"{mean:.3f}"
    return f"\\textbf{{{text}}}" if bold else text


def _fmt_pair(
    nmi: tuple[float, float, float, int],
    acc: tuple[float, float, float, int],
    *,
    bold_nmi: bool = False,
    bold_acc: bool = False,
) -> str:
    nmi_mean, _, _, n_nmi = nmi
    acc_mean, _, _, n_acc = acc
    return f"{_fmt_mean(nmi_mean, n_nmi, bold=bold_nmi)} & {_fmt_mean(acc_mean, n_acc, bold=bold_acc)}"


def _sub_id_to_lab(sub_id: int, lab_by_mouse: dict[str, str]) -> str | None:
    pid = f"sub-{int(sub_id):03d}"
    return lab_by_mouse.get(pid)


def _metrics_from_arrays(y_true: np.ndarray, y_hat: np.ndarray) -> tuple[float, float]:
    yt = y_true.astype(int).ravel()
    yh = y_hat.astype(int).ravel()
    return float(calculate_nmi(yh, yt)), float(macro_map_accuracy(yt, yh))


def _metrics_for_lab(
    npz_path: Path,
    target_lab: str,
    lab_by_mouse: dict[str, str],
) -> tuple[float, float] | None:
    data = np.load(npz_path)
    yt = data["y_true"].astype(int).ravel()
    yh = data["y_hat"].astype(int).ravel()
    sub_ids = data["sub_ids"].astype(int).ravel()
    lab_mask = np.array(
        [_sub_id_to_lab(int(s), lab_by_mouse) == target_lab for s in sub_ids],
        dtype=bool,
    )
    if not lab_mask.any():
        return None
    return _metrics_from_arrays(yt[lab_mask], yh[lab_mask])


def _metrics_pooled(npz_path: Path) -> tuple[float, float]:
    data = np.load(npz_path)
    return _metrics_from_arrays(data["y_true"], data["y_hat"])


def collect_vae_fair_metrics(rows: list[Row], manifest: dict) -> dict[tuple[str, str, str, str], dict[str, float]]:
    lab_by_mouse = {mid: info["lab"] for mid, info in manifest["inventory"].items()}
    out: dict[tuple[str, str, str, str], dict[str, float]] = {}

    for r in rows:
        if r.model not in VAE_MODELS:
            continue
        npz = results_npz_path(Path(r.result_root), r.selected_run)
        if npz is None:
            continue

        if r.scope == "joint":
            for lab in LABS:
                m = _metrics_for_lab(npz, lab, lab_by_mouse)
                if m is None:
                    continue
                nmi, acc = m
                out[("joint", r.fold, lab, r.model)] = {"nmi": nmi, "macro_accuracy": acc}
        elif r.scope == "within_lab" and r.lab in LABS:
            nmi, acc = _metrics_pooled(npz)
            out[("within_lab", r.fold, r.lab, r.model)] = {"nmi": nmi, "macro_accuracy": acc}

    return out


def collect_hmm_raw_fair_metrics(manifest: dict) -> dict[tuple[str, str, str, str], dict[str, float]]:
    out: dict[tuple[str, str, str, str], dict[str, float]] = {}
    for r in collect_joint_per_lab_rows(manifest):
        if r.status != "ok" or r.nmi is None or r.macro_accuracy is None:
            continue
        out[("joint", r.fold, r.lab, "hmm_raw")] = {
            "nmi": r.nmi,
            "macro_accuracy": r.macro_accuracy,
        }
    for r in collect_hmm_rows(manifest):
        if r.scope != "within_lab" or r.status != "ok":
            continue
        if r.nmi is None or r.macro_accuracy is None:
            continue
        out[("within_lab", r.fold, r.lab, "hmm_raw")] = {
            "nmi": r.nmi,
            "macro_accuracy": r.macro_accuracy,
        }
    return out


def collect_fair_metrics(rows: list[Row], manifest: dict) -> dict[tuple[str, str, str, str], dict[str, float]]:
    metrics = collect_vae_fair_metrics(rows, manifest)
    metrics.update(collect_hmm_raw_fair_metrics(manifest))
    return metrics


def _best_models(
    lab: str,
    metrics: dict[tuple[str, str, str, str], dict[str, float]],
    metric_key: str,
    scope: str,
) -> set[str]:
    scores: dict[str, float] = {}
    for model in MODELS:
        vals = [
            metrics[(scope, f, lab, model)][metric_key]
            for f in FOLDS
            if (scope, f, lab, model) in metrics
        ]
        if vals:
            scores[model] = stats.mean(vals)
    if not scores:
        return set()
    best = max(scores.values())
    return {m for m, v in scores.items() if abs(v - best) < 1e-9}


def write_fair_comparison_tex(path: Path, metrics: dict[tuple[str, str, str, str], dict[str, float]]) -> None:
    lines = [
        "% Auto-generated by scripts/paper/build_holdout_scope_table.py",
        "\\begin{tabular}{|l|l|r|r|r|r|}",
        "\\hline",
        "\\textbf{Lab} & \\textbf{Model} & \\textbf{Joint NMI} & \\textbf{Joint Acc.} "
        "& \\textbf{Within NMI} & \\textbf{Within Acc.} \\\\",
        "\\thickhline",
    ]
    for lab in LABS:
        best_joint_nmi = _best_models(lab, metrics, "nmi", "joint")
        best_joint_acc = _best_models(lab, metrics, "macro_accuracy", "joint")
        best_within_nmi = _best_models(lab, metrics, "nmi", "within_lab")
        best_within_acc = _best_models(lab, metrics, "macro_accuracy", "within_lab")

        for i, model in enumerate(MODELS):
            lab_label = LAB_LABEL[lab] if i == 0 else ""
            j_nmi = [metrics[("joint", f, lab, model)]["nmi"] for f in FOLDS if ("joint", f, lab, model) in metrics]
            j_acc = [
                metrics[("joint", f, lab, model)]["macro_accuracy"]
                for f in FOLDS
                if ("joint", f, lab, model) in metrics
            ]
            w_nmi = [
                metrics[("within_lab", f, lab, model)]["nmi"]
                for f in FOLDS
                if ("within_lab", f, lab, model) in metrics
            ]
            w_acc = [
                metrics[("within_lab", f, lab, model)]["macro_accuracy"]
                for f in FOLDS
                if ("within_lab", f, lab, model) in metrics
            ]
            joint_cells = _fmt_pair(
                _mean_range(j_nmi),
                _mean_range(j_acc),
                bold_nmi=model in best_joint_nmi and j_nmi,
                bold_acc=model in best_joint_acc and j_acc,
            )
            within_cells = _fmt_pair(
                _mean_range(w_nmi),
                _mean_range(w_acc),
                bold_nmi=model in best_within_nmi and w_nmi,
                bold_acc=model in best_within_acc and w_acc,
            )
            lines.append(
                f"{lab_label} & {MODEL_LABEL_TABLE[model]} & {joint_cells} & {within_cells} \\\\"
            )
        if lab != LABS[-1]:
            lines.append("\\hline")
    lines.extend(["\\hline", "\\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_detailed_tex(path: Path, metrics: dict[tuple[str, str, str, str], dict[str, float]]) -> None:
    lines = [
        "% Auto-generated by scripts/paper/build_holdout_scope_table.py",
        "\\begin{tabular}{|l|l|l|l|r|r|}",
        "\\hline",
        "\\textbf{Fold} & \\textbf{Train} & \\textbf{Lab} & \\textbf{Model} "
        "& \\textbf{NMI} & \\textbf{Acc.} \\\\",
        "\\thickhline",
    ]
    for fold in FOLDS:
        for lab in LABS:
            for model in MODELS:
                key = ("joint", fold, lab, model)
                if key not in metrics:
                    continue
                m = metrics[key]
                lines.append(
                    f"{fold} & Joint & {LAB_LABEL[lab]} & {MODEL_LABEL_TABLE[model]} "
                    f"& {m['nmi']:.3f} & {m['macro_accuracy']:.3f} \\\\"
                )
    lines.append("\\hline")
    for fold in FOLDS:
        for lab in LABS:
            for model in MODELS:
                key = ("within_lab", fold, lab, model)
                if key not in metrics:
                    continue
                m = metrics[key]
                lines.append(
                    f"{fold} & Within & {LAB_LABEL[lab]} & {MODEL_LABEL_TABLE[model]} "
                    f"& {m['nmi']:.3f} & {m['macro_accuracy']:.3f} \\\\"
                )
    lines.append("\\end{tabular}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _index_nmi(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, str], float]:
    out: dict[tuple[str, str, str, str], float] = {}
    for r in rows:
        out[(r["scope"], r["fold"], r["lab"], r["model"])] = float(r["nmi"])
    return out


def write_joint_fold_tex(path: Path, rows: list[dict[str, str]], hmm_metrics: dict) -> None:
    idx = _index_nmi(rows)
    vae_models = ("cgmvae_locked", "hmmgmvae_locked", "chmmgmvae_locked")
    header_models = ("hmm_raw",) + vae_models
    lines = [
        "% Auto-generated by scripts/paper/build_holdout_scope_table.py",
        "\\begin{tabular}{|l|r|r|r|r|}",
        "\\hline",
        "\\textbf{Fold} & \\textbf{HMM (raw)} & \\textbf{cGMVAE} & "
        "\\textbf{HMMGMVAE} & \\textbf{cHMM--GMVAE} \\\\",
        "\\thickhline",
    ]
    means: dict[str, list[float]] = defaultdict(list)
    for fold in FOLDS:
        vals = []
        for model in header_models:
            if model == "hmm_raw":
                m = hmm_metrics.get(("joint", fold, "all", "hmm_raw"))
                v = m["nmi"] if m else float("nan")
            else:
                v = idx.get(("joint", fold, "all", model), float("nan"))
            vals.append(v)
            if v == v:
                means[model].append(v)
        line_vals = []
        for v in vals:
            line_vals.append(f"{v:.3f}" if v == v else "---")
        lines.append(f"{fold} & {' & '.join(line_vals)} \\\\")
    lines.append("\\hline")
    mean_cells = []
    for model in header_models:
        mr, _, _, n = _mean_range(means[model])
        if n == 0:
            mean_cells.append("---")
        elif n == 1:
            mean_cells.append(f"{mr:.3f}")
        else:
            mean_cells.append(f"{mr:.3f}")
    lines.append(f"\\textbf{{Mean}} & {' & '.join(mean_cells)} \\\\")
    lines.extend(["\\hline", "\\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--fair-tex",
        type=Path,
        default=REPO / "docs/paper/assets/tables/T2_holdout_fair_comparison.tex",
    )
    parser.add_argument(
        "--detailed-tex",
        type=Path,
        default=REPO / "docs/paper/assets/tables/S2_within_lab_holdout.tex",
    )
    parser.add_argument(
        "--joint-fold-tex",
        type=Path,
        default=REPO / "docs/paper/assets/tables/T1_joint_holdout_ladder.tex",
    )
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    ladder_rows = collect_rows(DEFAULT_ROOTS)
    manifest = load_manifest(args.manifest)
    metrics = collect_fair_metrics(ladder_rows, manifest)

    hmm_pooled: dict[tuple[str, str, str, str], dict[str, float]] = {}
    for r in collect_hmm_rows(manifest):
        if r.scope == "joint" and r.status == "ok" and r.nmi is not None and r.macro_accuracy is not None:
            hmm_pooled[("joint", r.fold, "all", "hmm_raw")] = {
                "nmi": r.nmi,
                "macro_accuracy": r.macro_accuracy,
            }

    csv_rows = _load(args.csv)
    write_fair_comparison_tex(args.fair_tex, metrics)
    write_detailed_tex(args.detailed_tex, metrics)
    write_joint_fold_tex(args.joint_fold_tex, csv_rows, hmm_pooled)

    n_joint = sum(1 for k in metrics if k[0] == "joint")
    n_within = sum(1 for k in metrics if k[0] == "within_lab")
    print(f"Wrote {args.fair_tex} (joint per-lab cells: {n_joint}, within-lab: {n_within})")
    print(f"Wrote {args.detailed_tex}")
    print(f"Wrote {args.joint_fold_tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
