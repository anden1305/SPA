#!/usr/bin/env python3
"""Scrape macro-aligned accuracy (+ NMI) from holdout and K-sweep results.npz."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scripts.cv4fold.select_best_vae_run import select_best_run
from scripts.paper.summarize_holdout_ladder import (
    DEFAULT_ROOTS,
    MODEL_LABEL,
    MODEL_ORDER,
    Row,
    collect_rows,
)
from src.helpers.accuracy import accuracy
from src.helpers.nmi import calculate_nmi

REPO = Path(__file__).resolve().parents[2]
DEFAULT_K_ROOT = REPO / "results/cv4fold/paper_k_sweep/fold_4"
N_MACRO = 3


def macro_map_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_macro: int = N_MACRO) -> float:
    """Map each predicted cluster to dominant expert macro label, then accuracy."""
    yt = np.clip(y_true.astype(int), 0, n_macro - 1)
    yp = y_pred.astype(int)
    mapping: dict[int, int] = {}
    for p in np.unique(yp):
        mask = yp == p
        counts = np.bincount(yt[mask], minlength=n_macro)
        mapping[int(p)] = int(np.argmax(counts))
    yp_m = np.array([mapping[int(p)] for p in yp])
    return float(accuracy(yt, yp_m))


def metrics_from_npz(npz_path: Path) -> dict[str, float]:
    data = np.load(npz_path)
    yt = data["y_true"].ravel()
    yh = data["y_hat"].ravel()
    return {
        "nmi": float(calculate_nmi(yh, yt)),
        "macro_accuracy": macro_map_accuracy(yt, yh),
        "n_epochs": int(yt.size),
    }


def best_seed_npz(result_root: Path, selected_run: int) -> Path | None:
    p = result_root / "plots" / str(selected_run) / "results.npz"
    return p if p.is_file() else None


@dataclass
class AccRow:
    scope: str
    fold: str
    lab: str
    model: str
    k: int | None
    nmi: float
    macro_accuracy: float
    selected_run: int
    result_root: str


def scrape_holdout(rows: list[Row]) -> list[AccRow]:
    out: list[AccRow] = []
    for r in rows:
        if r.model not in MODEL_ORDER or r.model == "hmm_features":
            continue
        root = Path(r.result_root)
        npz = best_seed_npz(root, r.selected_run)
        if npz is None:
            continue
        acc = macro_map_accuracy(*[np.load(npz)[k].ravel() for k in ("y_true", "y_hat")])
        out.append(
            AccRow(
                scope=r.scope,
                fold=r.fold,
                lab=r.lab,
                model=r.model,
                k=3,
                nmi=r.nmi,
                macro_accuracy=acc,
                selected_run=r.selected_run,
                result_root=r.result_root,
            )
        )
    return out


def scrape_k_sweep(root: Path, k_min: int = 3, k_max: int = 15) -> list[AccRow]:
    out: list[AccRow] = []
    if not root.is_dir():
        return out
    for k in range(k_min, k_max + 1):
        k_dir = root / f"K{k}"
        if not k_dir.is_dir():
            continue
        runs = sorted(
            (p for p in k_dir.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for run in runs:
            try:
                sel = select_best_run(run)
            except FileNotFoundError:
                continue
            npz = best_seed_npz(Path(sel["result_root"]), sel["selected_run"])
            if npz is None:
                continue
            acc = macro_map_accuracy(*[np.load(npz)[key].ravel() for key in ("y_true", "y_hat")])
            out.append(
                AccRow(
                    scope="k_sweep",
                    fold="4",
                    lab="all",
                    model="chmmgmvae_locked",
                    k=k,
                    nmi=float(sel["nmi"]),
                    macro_accuracy=acc,
                    selected_run=sel["selected_run"],
                    result_root=sel["result_root"],
                )
            )
            break
    return out


def write_csv(path: Path, rows: list[AccRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "scope",
                "fold",
                "lab",
                "model",
                "model_label",
                "k",
                "nmi",
                "macro_accuracy",
                "selected_run",
                "result_root",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.scope,
                    r.fold,
                    r.lab,
                    r.model,
                    MODEL_LABEL.get(r.model, r.model),
                    "" if r.k is None else r.k,
                    f"{r.nmi:.4f}",
                    f"{r.macro_accuracy:.4f}",
                    r.selected_run,
                    r.result_root,
                ]
            )


def write_joint_latex(path: Path, rows: list[AccRow]) -> None:
    """LaTeX table: joint holdout accuracy by fold (cGMVAE vs cHMM)."""
    joint = [r for r in rows if r.scope == "joint" and r.k == 3]
    folds = sorted({r.fold for r in joint}, key=int)
    models = ["cgmvae_locked", "chmmgmvae_locked"]
    lines = [
        "% Auto-generated by scripts/paper/scrape_holdout_accuracy.py",
        "\\begin{tabular}{|l|r|r|r|r|}",
        "\\hline",
        "\\textbf{Fold} & \\textbf{cGMVAE NMI} & \\textbf{cGMVAE Acc.} "
        "& \\textbf{cHMM NMI} & \\textbf{cHMM Acc.} \\\\",
        "\\thickhline",
    ]
    for fold in folds:
        by_model = {r.model: r for r in joint if r.fold == fold}
        if not all(m in by_model for m in models):
            continue
        cg, ch = by_model["cgmvae_locked"], by_model["chmmgmvae_locked"]
        lines.append(
            f"{fold} & {cg.nmi:.3f} & {cg.macro_accuracy:.3f} "
            f"& {ch.nmi:.3f} & {ch.macro_accuracy:.3f} \\\\"
        )
    cg_nmis = [r.nmi for r in joint if r.model == "cgmvae_locked"]
    ch_nmis = [r.nmi for r in joint if r.model == "chmmgmvae_locked"]
    cg_accs = [r.macro_accuracy for r in joint if r.model == "cgmvae_locked"]
    ch_accs = [r.macro_accuracy for r in joint if r.model == "chmmgmvae_locked"]
    if cg_nmis and ch_nmis:
        lines.append("\\hline")
        lines.append(
            f"\\textbf{{Mean}} & {np.mean(cg_nmis):.3f} & {np.mean(cg_accs):.3f} "
            f"& {np.mean(ch_nmis):.3f} & {np.mean(ch_accs):.3f} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_k_sweep_latex(path: Path, rows: list[AccRow]) -> None:
    ks = sorted({r.k for r in rows if r.k is not None})
    by_k = {r.k: r for r in rows if r.scope == "k_sweep"}
    lines = [
        "% Auto-generated by scripts/paper/scrape_holdout_accuracy.py",
        "\\begin{tabular}{|r|r|r|}",
        "\\hline",
        "\\textbf{$K$} & \\textbf{NMI} & \\textbf{Macro accuracy} \\\\",
        "\\thickhline",
    ]
    for k in ks:
        r = by_k.get(k)
        if r is None:
            continue
        lines.append(f"{k} & {r.nmi:.3f} & {r.macro_accuracy:.3f} \\\\")
    lines.extend(["\\hline", "\\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=REPO / "paper/overleaf/tables/holdout_accuracy.csv",
    )
    parser.add_argument(
        "--k-root",
        type=Path,
        default=DEFAULT_K_ROOT,
    )
    parser.add_argument(
        "--latex-joint",
        type=Path,
        default=REPO / "docs/paper/assets/tables/S4_holdout_accuracy.tex",
    )
    parser.add_argument(
        "--latex-k-sweep",
        type=Path,
        default=REPO / "docs/paper/assets/tables/S5_k_sweep_accuracy.tex",
    )
    args = parser.parse_args()

    holdout_rows = scrape_holdout(collect_rows(DEFAULT_ROOTS))
    k_rows = scrape_k_sweep(args.k_root)
    all_rows = holdout_rows + k_rows
    write_csv(args.out_csv, all_rows)
    write_joint_latex(args.latex_joint, holdout_rows)
    write_k_sweep_latex(args.latex_k_sweep, k_rows)

    summary = {
        "n_holdout": len(holdout_rows),
        "n_k_sweep": len(k_rows),
        "joint_mean_accuracy": {
            "cgmvae_locked": float(
                np.mean([r.macro_accuracy for r in holdout_rows if r.scope == "joint" and r.model == "cgmvae_locked"])
            )
            if holdout_rows
            else None,
            "chmmgmvae_locked": float(
                np.mean([r.macro_accuracy for r in holdout_rows if r.scope == "joint" and r.model == "chmmgmvae_locked"])
            )
            if holdout_rows
            else None,
        },
    }
    json_path = args.out_csv.with_suffix(".json")
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {args.out_csv} ({len(all_rows)} rows)")
    print(f"Wrote {args.latex_joint}")
    print(f"Wrote {args.latex_k_sweep}")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
