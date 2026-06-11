#!/usr/bin/env python3
"""Biology meeting pack: full thesis-style substage plots for every K (fold-4 holdout sweep).

Output: results/cv4fold/paper_figures/biology_meeting/
  00_overview/     — K-selection curves + metrics table
  K03/ … K15/      — per-K thesis stack + publication panels

Also mirrors candidate K folders to docs/paper/figures/professor_meeting/.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from scripts.paper.plot_fig4_substage_compact import plot_compact as plot_biology_compact
from scripts.paper.plot_figure27_compact import run as run_publication_panels
from scripts.paper.plot_k_sweep_dual_axis import collect_k_sweep, pick_chosen_k, plot_dual_axis
from scripts.paper.run_k_sweep_review_pack import (
    CANDIDATE_K,
    best_npz_for_k,
    write_metrics_table,
)
from scripts.substage_analysis.fig27_from_npz import plot_fig27_gmm_predicted
from scripts.substage_analysis.full_analysis_from_npz import run_full_thesis_analysis, run_tsne_from_npz

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO / "results/cv4fold/paper_k_sweep/fold_4"
DEFAULT_OUT = REPO / "results/cv4fold/paper_figures/biology_meeting"
DEFAULT_PROFESSOR_DIR = REPO / "docs/paper/figures/professor_meeting"


def _write_k_readme(k_dir: Path, meta: dict) -> None:
    lines = [
        f"# K = {meta['K']}",
        "",
        f"- **Best seed:** {meta['best_seed']}",
        f"- **Prior NMI:** {meta['nmi']:.4f}",
        f"- **Run:** `{meta['run']}`",
        "",
        "## Show Birgitte (in order)",
        "",
        "1. `frequency_plot_gmm_predicted.png` — thesis Fig 27 physiology grid (all K rows)",
        "2. `pca_comparison_true_vs_predicted.png` — expert vs GM substages in latent space",
        "3. `tsne_scatter_true.png` / `tsne_scatter_predicted.png` — t-SNE of same latent subsample",
        "4. `transition_matrix_predicted.png` — switching dynamics",
        "5. `label_distribution.png` — occupancy / rare states",
        "6. `biology_compact.pdf` — compact summary panel",
        "7. `transition_hypnogram.pdf` — 2 h excerpt vs expert macro",
        "",
        "## Full stack",
        "",
        "| File | Content |",
        "|------|---------|",
        "| `pca_scatter_*.png` | PCA true / predicted / kmeans |",
        "| `tsne_scatter_*.png` | t-SNE true / predicted (shared embedding) |",
        "| `transition_matrix_*.png` | Transition matrices |",
        "| `latent_feature_boxplots.png` | Latent dim separation |",
        "| `substage_panels.pdf` | Publication physiology grid |",
        "",
        "**Ask:** stable substages vs transitions? Name each row? Keep this K for the paper?",
        "",
    ]
    (k_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _write_overview_readme(
    out_dir: Path,
    chosen_k: int,
    scores: dict,
    *,
    protocol: str,
) -> None:
    if protocol == "population":
        proto_line = "**Model:** cHMM–GMVAE locked, **all 20 mice** (incohort train+val, 5 seeds)"
        pick_line = f"**Incohort pick:** K={chosen_k} (best prior NMI across seeds; appendix only)"
        title = "# Population biology — substage K (appendix)"
    else:
        proto_line = "**Model:** cHMM–GMVAE locked, joint holdout **fold 4**"
        pick_line = f"**Stats pick:** K={chosen_k} (best holdout NMI)"
        title = "# Biology meeting — substage K selection"
    lines = [
        title,
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        proto_line,
        pick_line,
        "",
        "## Start here",
        "",
        "1. `00_overview/k_sweep_dual_axis.pdf` — NMI + log p(z) vs K (thesis Fig 23)",
        "2. Compare candidate folders: **K03, K04, K05, K07** (see table below)",
        "3. Per-K README in each folder lists what to show",
        "",
        "## Candidate comparison",
        "",
        "| K | Best NMI | Mean ± SD | Biology folder |",
        "|---|---------|-----------|----------------|",
    ]
    notes = {
        3: "Near-macro",
        4: "Best holdout NMI",
        5: "Slightly finer",
        7: "Thesis-like resolution",
    }
    for k in CANDIDATE_K:
        if k not in scores:
            continue
        s = scores[k]
        note = notes.get(k, "")
        lines.append(
            f"| {k} | {s['nmi_best']:.4f} | {s['nmi_mean']:.4f} ± {s['nmi_std']:.4f} | `K{k:02d}/` — {note} |"
        )
    lines += [
        "",
        "## All K",
        "",
        "Folders `K03/` … `K15/` each contain the full thesis plot stack.",
        "",
        "Interview script: `docs/paper/birgitte_interview_guide.md`",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--k-min", type=int, default=3)
    parser.add_argument("--k-max", type=int, default=15)
    parser.add_argument("--k-list", type=int, nargs="*", default=None)
    parser.add_argument("--skip-thesis-plots", action="store_true")
    parser.add_argument("--fig27-only", action="store_true", help="Only regenerate Fig 27 frequency grids")
    parser.add_argument("--tsne-only", action="store_true", help="Only regenerate t-SNE true/predicted scatters")
    parser.add_argument("--skip-publication", action="store_true")
    parser.add_argument(
        "--protocol",
        choices=("holdout", "population"),
        default="holdout",
        help="README text and default dual-axis title context",
    )
    parser.add_argument("--professor-dir", type=Path, default=None)
    parser.add_argument("--sync-professor-dir", action="store_true", default=True)
    parser.add_argument("--no-sync-professor-dir", action="store_false", dest="sync_professor_dir")
    args = parser.parse_args()

    professor_dir = args.professor_dir or (
        REPO / "docs/paper/figures/professor_meeting_population"
        if args.protocol == "population"
        else DEFAULT_PROFESSOR_DIR
    )
    dual_title = (
        "cHMM–GMVAE (all 20 mice, incohort)"
        if args.protocol == "population"
        else "cHMM–GMVAE (fold 4 holdout)"
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    overview = args.out_dir / "00_overview"
    overview.mkdir(parents=True, exist_ok=True)

    scores = collect_k_sweep(args.root, args.k_min, args.k_max)
    if not scores:
        raise SystemExit(f"No K-sweep data under {args.root}")

    chosen_k = pick_chosen_k(scores)
    dual_pdf = overview / "k_sweep_dual_axis.pdf"
    plot_dual_axis(scores, dual_pdf, chosen_k=chosen_k, title=dual_title)
    write_metrics_table(scores, overview / "k_sweep_metrics_table.csv")
    (overview / "k_sweep_summary.json").write_text(
        json.dumps({"chosen_k": chosen_k, "by_k": {str(k): v for k, v in sorted(scores.items())}}, indent=2),
        encoding="utf-8",
    )

    k_values = args.k_list if args.k_list else list(range(args.k_min, args.k_max + 1))
    per_k: dict[str, dict] = {}

    for k in k_values:
        hit = best_npz_for_k(args.root, k)
        if hit is None:
            print(f"Skip K={k}: no results.npz")
            continue
        npz, seed, nmi = hit
        k_dir = args.out_dir / f"K{k:02d}"
        k_dir.mkdir(parents=True, exist_ok=True)

        meta: dict = {
            "K": k,
            "best_seed": seed,
            "nmi": nmi,
            "npz": str(npz),
            "run": scores[k]["run"] if k in scores else "",
        }

        if args.fig27_only:
            meta["n_active_substages"] = plot_fig27_gmm_predicted(
                npz, k_dir / "frequency_plot_gmm_predicted.png", k=k, nmi=nmi
            )
        elif args.tsne_only:
            run_tsne_from_npz(npz, k_dir, k=k, nmi=nmi)
        elif not args.skip_thesis_plots:
            run_full_thesis_analysis(npz, k_dir, k=k, nmi=nmi)

        if not args.skip_publication and not args.fig27_only and not args.tsne_only:
            plot_biology_compact(npz, k_dir / "biology_compact.pdf")
            run_publication_panels(
                npz,
                k_dir / "substage_panels.pdf",
                k_dir / "transition_matrix.pdf",
                k_dir / "hypnogram.pdf",
                k_dir / "transition_hypnogram.pdf",
            )
        (k_dir / "metrics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        _write_k_readme(k_dir, meta)
        per_k[str(k)] = meta
        print(f"K={k:2d} seed={seed} NMI={nmi:.4f} → {k_dir}")

    _write_overview_readme(args.out_dir, chosen_k, scores, protocol=args.protocol)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(args.root),
        "out_dir": str(args.out_dir),
        "chosen_k": chosen_k,
        "per_k": per_k,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.sync_professor_dir:
        professor_dir.mkdir(parents=True, exist_ok=True)
        for name in ("k_sweep_dual_axis.pdf", "k_sweep_dual_axis.png", "k_sweep_metrics_table.csv"):
            src = overview / name
            if src.is_file():
                shutil.copy(src, professor_dir / name)
        for k_dir in sorted(args.out_dir.glob("K[0-9][0-9]")):
            if not k_dir.is_dir():
                continue
            dst = professor_dir / k_dir.name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(k_dir, dst)
        shutil.copy(args.out_dir / "README.md", professor_dir / "README.md")
        print(f"Synced all K folders + overview → {professor_dir}")

    print(f"Done → {args.out_dir} ({len(per_k)} K folders, chosen K={chosen_k})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
