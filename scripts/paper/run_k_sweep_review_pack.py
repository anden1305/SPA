#!/usr/bin/env python3
"""Generate full K-sweep review pack for substage selection + professor biology discussion.

Outputs under results/cv4fold/paper_figures/k_sweep_review/:
  - K-selection curves (dual-axis + NMI-only)
  - Per-K biology panels (best seed per K)
  - metrics_table.csv + manifest.json
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.paper.plot_fig4_substage_compact import plot_compact as plot_biology_compact
from scripts.paper.plot_figure27_compact import run as run_publication_panels
from scripts.paper.k_sweep_metrics import best_npz_for_k_merged
from scripts.paper.plot_k_sweep_dual_axis import collect_k_sweep, pick_chosen_k, plot_dual_axis

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO / "results/cv4fold/paper_k_sweep/fold_4"
DEFAULT_OUT = REPO / "results/cv4fold/paper_figures/k_sweep_review"
CANDIDATE_K = (3, 4, 5, 7)


def best_npz_for_k(root: Path, k: int) -> tuple[Path, int, float] | None:
    hit = best_npz_for_k_merged(root, k)
    if hit is None:
        return None
    npz, seed, nmi, _run = hit
    return npz, seed, nmi


def write_metrics_table(scores: dict[int, dict], out_csv: Path) -> None:
    rows = []
    for k in sorted(scores):
        s = scores[k]
        row = {
            "K": k,
            "nmi_mean": f"{s['nmi_mean']:.4f}",
            "nmi_std": f"{s['nmi_std']:.4f}",
            "nmi_best": f"{s['nmi_best']:.4f}",
            "nmi_min": f"{s['nmi_min']:.4f}",
            "n_seeds": s["n_seeds"],
            "log_p_z_mean": "" if s.get("log_p_z_mean") is None else f"{s['log_p_z_mean']:.2f}",
            "run": s["run"],
            "candidate": "yes" if k in CANDIDATE_K else "",
        }
        rows.append(row)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_readme(out_dir: Path, chosen_k: int, scores: dict[int, dict]) -> None:
    lines = [
        "# K-sweep review pack",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Model selection (Fig 23 style)",
        "",
        "- `k_sweep_dual_axis.pdf` — Validation NMI (blue) + log p(z₁:T) (orange) vs K",
        "- `k_sweep_nmi_only.pdf` — Best prior NMI per K with seed range",
        "- `k_sweep_metrics_table.csv` — numeric summary",
        "",
        f"**Auto-picked K={chosen_k}** (argmax best-seed NMI = {scores[chosen_k]['nmi_best']:.4f}).",
        "",
        "## Recommended candidates for biology review",
        "",
        "| K | Best NMI | Mean ± SD | Notes |",
        "|---|---------|-----------|-------|",
    ]
    notes = {
        3: "Stable seeds; near-macro resolution",
        4: "**Current winner** — best holdout NMI",
        5: "Finer than macro; moderate seed spread",
        7: "Thesis operating point; higher variance",
    }
    for k in CANDIDATE_K:
        if k not in scores:
            continue
        s = scores[k]
        lines.append(
            f"| {k} | {s['nmi_best']:.4f} | {s['nmi_mean']:.4f} ± {s['nmi_std']:.4f} | {notes.get(k, '')} |"
        )
    lines += [
        "",
        "## Per-K biology folders (`K03/` … `K15/`)",
        "",
        "Each folder uses the **best-seed** run at that K:",
        "",
        "- `biology_compact.pdf` — 2×2 PSD / bout / macro-mix panel (main-text style)",
        "- `substage_panels.pdf` — full physiology grid",
        "- `transition_hypnogram.pdf` — dynamics",
        "- `metrics.json` — seed, NMI, npz path",
        "",
        "## Professor meeting",
        "",
        "Bring `K03/`, `K04/`, `K05/` biology panels + dual-axis curve.",
        "See `docs/paper/birgitte_interview_guide.md` for question script.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--k-min", type=int, default=3)
    parser.add_argument("--k-max", type=int, default=15)
    parser.add_argument("--chosen-k", type=int, default=None)
    parser.add_argument("--skip-per-k", action="store_true", help="Only K-selection curves")
    parser.add_argument("--k-list", type=int, nargs="*", default=None, help="Subset of K (default: all in range)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    scores = collect_k_sweep(args.root, args.k_min, args.k_max)
    if not scores:
        raise SystemExit(f"No K-sweep data under {args.root}")

    chosen_k = args.chosen_k if args.chosen_k is not None else pick_chosen_k(scores)

    dual_pdf = args.out_dir / "k_sweep_dual_axis.pdf"
    plot_dual_axis(scores, dual_pdf, chosen_k=chosen_k)
    summary = {"chosen_k": chosen_k, "by_k": {str(k): v for k, v in sorted(scores.items())}}
    (args.out_dir / "k_sweep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_metrics_table(scores, args.out_dir / "k_sweep_metrics_table.csv")

    # NMI-only curve (paper S3 style) into same folder
    import matplotlib.pyplot as plt
    import numpy as np
    from scripts.paper.plot_k_sweep_curve import collect_k_sweep as collect_nmi
    from scripts.paper.plot_style import apply_paper_style, save_figure

    nmi_scores = collect_nmi(args.root)
    ks = sorted(nmi_scores.keys())
    bests = [nmi_scores[k]["best"] for k in ks]
    mins = [nmi_scores[k]["min"] for k in ks]
    maxs = [nmi_scores[k]["max"] for k in ks]
    yerr = np.array([np.array(bests) - np.array(mins), np.array(maxs) - np.array(bests)])
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(5.0, 2.35))
    ax.errorbar(ks, bests, yerr=yerr, fmt="o-", color="#7C3AED", linewidth=1.5, markersize=5, capsize=3)
    if chosen_k in nmi_scores:
        ax.axvline(chosen_k, color="#334155", linestyle="--", lw=1.0, alpha=0.75)
        ax.scatter([chosen_k], [nmi_scores[chosen_k]["best"]], s=55, color="#7C3AED", zorder=4)
    ax.set_xlabel("Mixture components $K$")
    ax.set_ylabel("Best prior NMI (holdout)")
    ax.set_xticks(list(range(args.k_min, args.k_max + 1)))
    save_figure(fig, args.out_dir / "k_sweep_nmi_only.pdf")
    plt.close(fig)

    per_k_manifest: dict[str, dict] = {}
    k_values = args.k_list if args.k_list else list(range(args.k_min, args.k_max + 1))

    if not args.skip_per_k:
        for k in k_values:
            hit = best_npz_for_k(args.root, k)
            if hit is None:
                print(f"Skip K={k}: no results.npz")
                continue
            npz, seed, nmi = hit
            k_dir = args.out_dir / f"K{k:02d}"
            k_dir.mkdir(parents=True, exist_ok=True)
            plot_biology_compact(npz, k_dir / "biology_compact.pdf")
            run_publication_panels(
                npz,
                k_dir / "substage_panels.pdf",
                k_dir / "transition_matrix.pdf",
                k_dir / "hypnogram.pdf",
                k_dir / "transition_hypnogram.pdf",
            )
            meta = {"K": k, "best_seed": seed, "nmi": nmi, "npz": str(npz), "run": scores[k]["run"] if k in scores else ""}
            (k_dir / "metrics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            per_k_manifest[str(k)] = {**meta, "biology_compact": str(k_dir / "biology_compact.pdf")}
            print(f"K={k:2d} seed={seed} NMI={nmi:.4f} → {k_dir}")

    write_readme(args.out_dir, chosen_k, scores)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(args.root),
        "out_dir": str(args.out_dir),
        "chosen_k": chosen_k,
        "curves": {
            "dual_axis": str(dual_pdf),
            "nmi_only": str(args.out_dir / "k_sweep_nmi_only.pdf"),
            "metrics_csv": str(args.out_dir / "k_sweep_metrics_table.csv"),
        },
        "per_k": per_k_manifest,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Done → {args.out_dir} (chosen K={chosen_k}, n_K={len(scores)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
