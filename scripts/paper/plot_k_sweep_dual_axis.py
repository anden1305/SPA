#!/usr/bin/env python3
"""K-sweep dual-axis plot (thesis Fig 23 style): Validation NMI + Predictive Likelihood vs K."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO / "results/cv4fold/paper_k_sweep/fold_4"
DEFAULT_OUT = REPO / "results/cv4fold/paper_figures/k_sweep/k_sweep_dual_axis.pdf"
# Locked fold-4 K-sweep: dataloader.sequence_length × model.params.latent_dim
SEQUENCE_LENGTH = 64
LATENT_DIM = 6


def _parse_metrics(path: Path) -> dict[str, float] | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    nmi_m = re.search(r"^NMI:\s*([0-9.+-eE]+)", text, re.M)
    ll_m = re.search(r"^log p\(z_1:T\):\s*([0-9.+-eE]+)", text, re.M)
    if not nmi_m:
        return None
    out = {"nmi": float(nmi_m.group(1))}
    if ll_m:
        out["log_p_z"] = float(ll_m.group(1))
    return out


def _read_seed_metrics(run_dir: Path) -> list[dict[str, float]]:
    plots = run_dir / "plots"
    if not plots.is_dir():
        return []
    seed_dirs = sorted(
        (p for p in plots.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    )
    rows: list[dict[str, float]] = []
    if seed_dirs:
        for seed_dir in seed_dirs:
            m = _parse_metrics(seed_dir / "metrics.txt")
            if m:
                rows.append(m)
        return rows
    m = _parse_metrics(plots / "metrics.txt")
    return [m] if m else []


def collect_k_sweep(root: Path, k_min: int = 3, k_max: int = 15) -> dict[int, dict]:
    """Per-K seed metrics merged across all run dirs (base + extra2 jobs)."""
    from scripts.paper.k_sweep_metrics import collect_k_sweep_merged

    return collect_k_sweep_merged(root, k_min, k_max)


def pick_chosen_k(scores: dict[int, dict], *, criterion: str = "best") -> int | None:
    if not scores:
        return None
    if criterion == "mean":
        return max(scores.keys(), key=lambda k: scores[k]["nmi_mean"])
    return max(scores.keys(), key=lambda k: scores[k]["nmi_best"])


def _ll_norm_divisor(mode: str) -> float:
    if mode == "none":
        return 1.0
    if mode == "T":
        return float(SEQUENCE_LENGTH)
    if mode == "TL":
        return float(SEQUENCE_LENGTH * LATENT_DIM)
    raise ValueError(f"Unknown ll_norm mode: {mode!r}")


def plot_dual_axis(
    scores: dict[int, dict],
    out_path: Path,
    *,
    title: str = "cHMM–GMVAE (fold 4 holdout)",
    chosen_k: int | None = None,
    ll_norm: str = "TL",
) -> None:
    """Thesis Fig 23 styling: blue circles = NMI (left), orange squares = log p(z) (right)."""
    div = _ll_norm_divisor(ll_norm)
    ks = sorted(scores.keys())
    nmi = np.array([scores[k]["nmi_mean"] for k in ks], dtype=float)
    nmi_std = np.array([scores[k]["nmi_std"] for k in ks], dtype=float)
    ll = [scores[k]["log_p_z_mean"] for k in ks]
    ll_std = [scores[k].get("log_p_z_std", 0.0) for k in ks]
    has_ll = any(v is not None for v in ll)
    ll_vals = np.array([(v / div) if v is not None else np.nan for v in ll], dtype=float)
    ll_std_n = np.array([(s / div) if s is not None else 0.0 for s in ll_std], dtype=float)
    x = np.array(ks, dtype=float)

    fig, ax_nmi = plt.subplots(figsize=(4.8, 3.0))
    fig.patch.set_facecolor("#F8F8F8")
    ax_nmi.set_facecolor("#F0F0F0")

    color_nmi = "#1f77b4"
    color_ll = "#ff7f0e"

    ax_nmi.set_xlabel("Number of substages", fontsize=11)
    ax_nmi.set_ylabel("Validation NMI", color=color_nmi, fontsize=11)
    ax_nmi.fill_between(x, nmi - nmi_std, nmi + nmi_std, color=color_nmi, alpha=0.22, linewidth=0)
    (line_nmi,) = ax_nmi.plot(
        x, nmi, "o-", color=color_nmi, linewidth=2, markersize=7, label="Validation NMI",
    )
    ax_nmi.tick_params(axis="y", labelcolor=color_nmi)
    ax_nmi.grid(True, alpha=0.35)
    ax_nmi.set_xticks(ks)

    lines = [line_nmi]
    labels = ["Validation NMI"]

    if has_ll:
        ax_ll = ax_nmi.twinx()
        ax_ll.set_facecolor("#F0F0F0")
        if ll_norm == "TL":
            ll_ylabel = f"Mean log prior density\nlog p(z₁:T) / (T·L), T={SEQUENCE_LENGTH}, L={LATENT_DIM}"
        elif ll_norm == "T":
            ll_ylabel = f"Mean log prior density\nlog p(z₁:T) / T, T={SEQUENCE_LENGTH}"
        else:
            ll_ylabel = "Predictive Likelihood\nlog p(z₁:T)"
        ax_ll.set_ylabel(ll_ylabel, color=color_ll, fontsize=10)
        ax_ll.fill_between(x, ll_vals - ll_std_n, ll_vals + ll_std_n, color=color_ll, alpha=0.18, linewidth=0)
        (line_ll,) = ax_ll.plot(
            x, ll_vals, "s-", color=color_ll, linewidth=2, markersize=7, label="Predictive Likelihood",
        )
        ax_ll.tick_params(axis="y", labelcolor=color_ll)
        lines.append(line_ll)
        labels.append("Predictive Likelihood")

    if chosen_k is not None and chosen_k in scores:
        ax_nmi.axvline(chosen_k, color="#555555", linestyle="--", lw=1.0, alpha=0.7)
        ax_nmi.text(
            chosen_k + 0.15, ax_nmi.get_ylim()[1] * 0.92,
            f"K={chosen_k}", fontsize=8, color="#374151",
        )

    ax_nmi.legend(
        lines, [f"{l} (mean ± SD)" for l in labels],
        loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=2, fontsize=8,
    )
    ax_nmi.set_title(title, fontsize=12, pad=28)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--chosen-k", type=int, default=None)
    parser.add_argument("--k-min", type=int, default=3)
    parser.add_argument("--k-max", type=int, default=15)
    parser.add_argument("--title", type=str, default="cHMM–GMVAE (fold 4 holdout)")
    parser.add_argument(
        "--ll-norm",
        choices=("none", "T", "TL"),
        default="TL",
        help="Scale log p(z₁:T): TL = per latent dim per timestep (default)",
    )
    args = parser.parse_args()

    scores = collect_k_sweep(args.root, args.k_min, args.k_max)
    if not scores:
        raise SystemExit(f"No K-sweep metrics found under {args.root}")

    chosen_k = args.chosen_k if args.chosen_k is not None else pick_chosen_k(scores)
    plot_dual_axis(scores, args.out, title=args.title, chosen_k=chosen_k, ll_norm=args.ll_norm)

    summary_path = args.summary_json or args.out.with_suffix(".json")
    div = _ll_norm_divisor(args.ll_norm)
    by_k = {}
    for k, v in sorted(scores.items()):
        row = dict(v)
        if row.get("log_p_z_mean") is not None:
            row["log_p_z_mean_norm"] = row["log_p_z_mean"] / div
            row["log_p_z_std_norm"] = row.get("log_p_z_std", 0.0) / div
        by_k[str(k)] = row
    summary = {
        "root": str(args.root),
        "chosen_k": chosen_k,
        "ll_norm": args.ll_norm,
        "ll_norm_divisor": div,
        "by_k": by_k,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {args.out} (+ .png), n_K={len(scores)}, chosen_K={chosen_k}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
