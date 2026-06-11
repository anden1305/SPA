#!/usr/bin/env python3
"""K substage sweep curve (S3 Fig) from paper_k_sweep holdout results."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.paper.plot_style import apply_paper_style, panel_label, save_figure

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO / "results/cv4fold/paper_k_sweep/fold_4"
K_VALUES = list(range(3, 16))


def _read_seed_nmis(run_dir: Path) -> list[float]:
    plots = run_dir / "plots"
    if not plots.is_dir():
        return []
    nmis: list[float] = []
    seed_dirs = sorted(p for p in plots.iterdir() if p.is_dir() and p.name.isdigit())
    if seed_dirs:
        for seed_dir in seed_dirs:
            metrics = seed_dir / "metrics.txt"
            if not metrics.is_file():
                continue
            text = metrics.read_text(encoding="utf-8", errors="replace")
            m = re.search(r"^NMI:\s*([0-9.]+)", text, re.I | re.M)
            if m:
                nmis.append(float(m.group(1)))
        return nmis
    metrics = plots / "metrics.txt"
    if metrics.is_file():
        text = metrics.read_text(encoding="utf-8", errors="replace")
        for pat in (r"prior[_\s]*nmi[:\s]+([0-9.]+)", r"^NMI:\s*([0-9.]+)"):
            m = re.search(pat, text, re.I | re.M)
            if m:
                return [float(m.group(1))]
    return []


def collect_k_sweep(root: Path) -> dict[int, dict[str, float]]:
    """Return per-K best/min/max NMI across all seeds (base + extra2 runs)."""
    from scripts.paper.k_sweep_metrics import collect_k_sweep_merged

    merged = collect_k_sweep_merged(root, min(K_VALUES), max(K_VALUES))
    out: dict[int, dict[str, float]] = {}
    for k, s in merged.items():
        nmis = [x["nmi"] for x in s["seeds"]]
        out[k] = {
            "best": max(nmis),
            "min": min(nmis),
            "max": max(nmis),
            "run": s["run"],
            "n_seeds": s["n_seeds"],
        }
    return out


def pick_chosen_k(scores: dict[int, dict[str, float]]) -> int | None:
    if not scores:
        return None
    return max(scores.keys(), key=lambda k: scores[k]["best"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=REPO / "docs/paper/figures/main/S3_k_sweep.pdf")
    parser.add_argument("--summary-json", type=Path, default=REPO / "docs/paper/figures/k_sweep/k_sweep_summary.json")
    parser.add_argument("--chosen-k", type=int, default=None, help="Mark chosen K (default: argmax best NMI)")
    args = parser.parse_args()

    scores = collect_k_sweep(args.root)
    if not scores:
        raise SystemExit(f"No K-sweep NMI found under {args.root}")

    chosen_k = args.chosen_k if args.chosen_k is not None else pick_chosen_k(scores)
    ks = sorted(scores.keys())
    bests = [scores[k]["best"] for k in ks]
    mins = [scores[k]["min"] for k in ks]
    maxs = [scores[k]["max"] for k in ks]
    yerr = np.array([np.array(bests) - np.array(mins), np.array(maxs) - np.array(bests)])

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(5.0, 2.35))
    ax.errorbar(
        ks, bests, yerr=yerr, fmt="o-", color="#7C3AED", linewidth=1.5,
        markersize=5, capsize=3, capthick=1, elinewidth=1, markerfacecolor="white",
        markeredgewidth=1.2, zorder=3,
    )
    if chosen_k is not None and chosen_k in scores:
        ax.axvline(chosen_k, color="#334155", linestyle="--", lw=1.0, alpha=0.75, zorder=1)
        ax.scatter([chosen_k], [scores[chosen_k]["best"]], s=55, color="#7C3AED", zorder=4)
        ax.text(
            chosen_k + 0.25, scores[chosen_k]["best"] + 0.015,
            f"$K$={chosen_k}\n({scores[chosen_k]['best']:.3f})",
            fontsize=7, va="bottom",
        )
    ax.set_xlabel("Mixture components $K$")
    ax.set_ylabel("Best prior NMI (holdout)")
    ax.set_xticks(K_VALUES)
    ax.set_xticklabels([str(k) if k in ks else str(k) for k in K_VALUES], fontsize=7)
    for i, tick in enumerate(ax.get_xticklabels()):
        if K_VALUES[i] not in ks:
            tick.set_color("#94A3B8")
    ax.set_ylim(max(0, min(mins) - 0.05), min(0.75, max(maxs) + 0.06))
    ax.tick_params(axis="both", labelsize=7)

    save_figure(fig, args.out)
    plt.close(fig)

    summary = {
        "root": str(args.root),
        "chosen_k": chosen_k,
        "by_k": {str(k): v for k, v in sorted(scores.items())},
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    staging = REPO / "paper/overleaf/figures/s3_k_sweep.pdf"
    staging.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(args.out, staging)
    print(f"Wrote {args.out} (n_K={len(scores)}, chosen_K={chosen_k})")
    print(f"Wrote {args.summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
