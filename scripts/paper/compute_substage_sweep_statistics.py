#!/usr/bin/env python3
"""Substage validity metrics per K (population / holdout sweeps).

Replaces fold-level Wilcoxon for substage discovery: tests whether mixture
components are (i) active, (ii) latent-separated vs label permutation, and
(iii) associated with expert macros (Cramér's V, NMI).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.stats import chi2_contingency

from scripts.paper.k_sweep_metrics import best_npz_for_k_merged
from src.helpers.state_distinctness import compute_state_distinctness

REPO = Path(__file__).resolve().parents[2]
DEFAULT_POP = REPO / "results/cv4fold/paper_k_sweep/population"
DEFAULT_HOLDOUT = REPO / "results/cv4fold/paper_k_sweep/fold_4"
OUT_POP = REPO / "results/cv4fold/paper_figures/biology_meeting_population/00_overview"
OUT_HOLDOUT = REPO / "results/cv4fold/paper_figures/biology_meeting/00_overview"


def _cramers_v(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    classes_t = np.unique(y_true)
    classes_p = np.unique(y_pred)
    table = np.zeros((len(classes_t), len(classes_p)), dtype=np.int64)
    for i, t in enumerate(classes_t):
        for j, p in enumerate(classes_p):
            table[i, j] = int(np.sum((y_true == t) & (y_pred == p)))
    if table.sum() == 0 or min(table.shape) < 2:
        return float("nan")
    chi2, _, _, _ = chi2_contingency(table)
    n = table.sum()
    k = min(table.shape) - 1
    if k <= 0 or n == 0:
        return float("nan")
    return float(math.sqrt(chi2 / (n * k)))


def _occupancy_stats(y_pred: np.ndarray) -> dict[str, float]:
    counts = np.bincount(y_pred.astype(int))
    active = counts[counts > 0]
    if active.size == 0:
        return {"active": 0, "min_occ": 0.0, "entropy_norm": 0.0, "mean_macro_purity": float("nan")}
    occ = active / active.sum()
    n = y_pred.size
    raw = counts[counts > 0] / n
    h = -np.sum(raw * np.log(raw + 1e-12))
    h_max = math.log(len(active))
    return {
        "active": int(len(active)),
        "min_occ_pct": float(100.0 * raw.min()),
        "entropy_norm": float(h / h_max) if h_max > 0 else 0.0,
    }


def _macro_purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Occupancy-weighted mean dominant-macro fraction per predicted substage."""
    macros = np.unique(y_true)
    total = 0.0
    weight = 0.0
    for p in np.unique(y_pred):
        mask = y_pred == p
        w = mask.sum()
        if w == 0:
            continue
        counts = {int(m): int((y_true[mask] == m).sum()) for m in macros}
        weight += w
        total += w * max(counts.values()) / w
    return float(total / weight) if weight else float("nan")


def _subsample(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_points: int,
    seed: int = 124,
) -> tuple[np.ndarray, np.ndarray]:
    n = x.shape[0]
    if n <= max_points:
        return x, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    return x[idx], y[idx]


def _latent_distinctness(x: np.ndarray, y: np.ndarray, *, max_points: int = 6000) -> dict[str, float]:
    xs, ys = _subsample(x, y, max_points=max_points)
    out = compute_state_distinctness(torch.from_numpy(xs.astype(np.float32)), torch.from_numpy(ys.astype(np.int64)))
    return {
        "latent_mean_ed": float(out["mean_pairwise_energy"]),
        "latent_weighted_ed": float(out["weighted_mean_pairwise_energy"]),
        "latent_fisher": float(out["fisher_trace"]) if out["fisher_trace"] is not None else float("nan"),
    }


def _permutation_separation_p(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_perm: int = 50,
    max_points: int = 6000,
    seed: int = 124,
) -> float:
    xs, ys = _subsample(x, y, max_points=max_points, seed=seed)
    obs = _latent_distinctness(xs, ys, max_points=max_points)["latent_mean_ed"]
    rng = np.random.default_rng(seed + 1)
    ys = ys.astype(np.int64)
    exceed = 0
    for _ in range(n_perm):
        y_perm = rng.permutation(ys)
        null = _latent_distinctness(xs, y_perm, max_points=max_points)["latent_mean_ed"]
        if null >= obs:
            exceed += 1
    return float((exceed + 1) / (n_perm + 1))


def _load_validations_for_npz(npz_path: Path) -> dict | None:
    run_dir = npz_path.parents[2]
    dv_path = run_dir / "data_validations.json"
    if not dv_path.is_file():
        return None
    return json.loads(dv_path.read_text(encoding="utf-8"))


def analyze_k(root: Path, k: int, *, n_perm: int) -> dict | None:
    hit = best_npz_for_k_merged(root, k)
    if hit is None:
        return None
    npz_path, seed, nmi, run_name = hit
    data = np.load(npz_path)
    x = data["x_latent"].reshape(-1, data["x_latent"].shape[-1])
    y_true = data["y_true"].reshape(-1).astype(int)
    y_pred = data["y_hat"].reshape(-1).astype(int)

    occ = _occupancy_stats(y_pred)
    distinct = _latent_distinctness(x, y_pred)
    row = {
        "K": k,
        "configured_K": k,
        "best_seed": seed,
        "nmi": float(nmi),
        "cramers_v": _cramers_v(y_true, y_pred),
        "macro_purity": _macro_purity(y_true, y_pred),
        "perm_p_separation": _permutation_separation_p(x, y_pred, n_perm=n_perm),
        "run": run_name,
        "npz": str(npz_path.relative_to(REPO) if str(npz_path).startswith(str(REPO)) else npz_path),
        **occ,
        **distinct,
    }
    dv = _load_validations_for_npz(npz_path)
    if dv and "latent_mean_pairwise_energy" in dv:
        row["latent_mean_ed_logged"] = float(dv["latent_mean_pairwise_energy"])
    row["state_collapse"] = "yes" if row["active"] < k else "no"
    return row


def collect_sweep(root: Path, k_min: int, k_max: int, *, n_perm: int) -> list[dict]:
    rows: list[dict] = []
    for k in range(k_min, k_max + 1):
        row = analyze_k(root, k, n_perm=n_perm)
        if row:
            rows.append(row)
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_tex(rows: list[dict], path: Path, *, caption_tag: str) -> None:
    if not rows:
        return
    lines = [
        f"% Auto-generated by scripts/paper/compute_substage_sweep_statistics.py ({caption_tag})",
        r"\begin{tabular}{|r|r|r|r|r|r|r|r|}",
        r"\hline",
        r"\textbf{$K$} & \textbf{Active} & \textbf{Min occ.\ (\%)} & \textbf{Latent ED} & \textbf{Fisher} & \textbf{Cramér $V$} & \textbf{NMI} & \textbf{$p_{\mathrm{perm}}$} \\",
        r"\thickhline",
    ]
    for r in rows:
        lines.append(
            f"{r['K']} & {r['active']}/{r['configured_K']} & {r['min_occ_pct']:.2f} & "
            f"{r['latent_mean_ed']:.2f} & {r['latent_fisher']:.2f} & {r['cramers_v']:.3f} & "
            f"{r['nmi']:.3f} & {r['perm_p_separation']:.3f} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_POP)
    parser.add_argument("--out-dir", type=Path, default=OUT_POP)
    parser.add_argument("--k-min", type=int, default=3)
    parser.add_argument("--k-max", type=int, default=15)
    parser.add_argument("--n-perm", type=int, default=100)
    parser.add_argument("--tag", type=str, default="population")
    args = parser.parse_args()

    rows = collect_sweep(args.root, args.k_min, args.k_max, n_perm=args.n_perm)
    if not rows:
        print(f"No K with results.npz under {args.root}")
        return 1

    write_csv(rows, args.out_dir / f"k_sweep_substage_validity_{args.tag}.csv")
    write_tex(rows, REPO / f"docs/paper/assets/tables/S12_substage_validity_{args.tag}.tex", caption_tag=args.tag)
    (args.out_dir / f"k_sweep_substage_validity_{args.tag}.json").write_text(
        json.dumps({"tag": args.tag, "rows": rows}, indent=2), encoding="utf-8"
    )
    for r in rows:
        print(
            f"K={r['K']:2d} active={r['active']}/{r['configured_K']} "
            f"ED={r['latent_mean_ed']:.2f} V={r['cramers_v']:.3f} "
            f"p_perm={r['perm_p_separation']:.3f} NMI={r['nmi']:.3f}"
        )
    print(f"Wrote {args.out_dir}/k_sweep_substage_validity_{args.tag}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
