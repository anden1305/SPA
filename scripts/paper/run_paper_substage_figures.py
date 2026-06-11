#!/usr/bin/env python3
"""Generate paper substage figures into results/cv4fold/paper_figures/ (isolated from thesis results/)."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from scripts.paper.plot_figure27_compact import run as run_compact_panels
from scripts.paper.plot_k_sweep_dual_axis import collect_k_sweep, pick_chosen_k, plot_dual_axis
from scripts.substage_analysis.full_analysis_from_npz import run_full_thesis_analysis

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "results/cv4fold/paper_figures"
DEFAULT_K_ROOT = REPO / "results/cv4fold/paper_k_sweep/fold_4"
DEFAULT_NPZ = (
    DEFAULT_K_ROOT
    / "K4/joint_k_sweep_f4_K4_20260609-183717/plots/3/results.npz"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--k-root", type=Path, default=DEFAULT_K_ROOT)
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--chosen-k", type=int, default=None)
    parser.add_argument("--skip-k-sweep", action="store_true")
    parser.add_argument("--skip-substage", action="store_true")
    parser.add_argument("--copy-to-paper", action="store_true", help="Also copy key PDFs to docs/paper/figures/")
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "k_root": str(args.k_root),
        "npz": str(args.npz),
        "outputs": {},
    }

    if not args.skip_k_sweep:
        k_dir = args.out_root / "k_sweep"
        scores = collect_k_sweep(args.k_root)
        if not scores:
            raise SystemExit(f"No K-sweep data under {args.k_root}")
        chosen_k = args.chosen_k if args.chosen_k is not None else pick_chosen_k(scores)
        dual_pdf = k_dir / "k_sweep_dual_axis.pdf"
        plot_dual_axis(scores, dual_pdf, chosen_k=chosen_k)
        summary = {"chosen_k": chosen_k, "by_k": {str(k): v for k, v in sorted(scores.items())}}
        (k_dir / "k_sweep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        manifest["outputs"]["k_sweep"] = {
            "dual_axis_pdf": str(dual_pdf),
            "dual_axis_png": str(dual_pdf.with_suffix(".png")),
            "summary_json": str(k_dir / "k_sweep_summary.json"),
            "chosen_k": chosen_k,
        }
        print(f"K-sweep: {dual_pdf} (chosen K={chosen_k})")

    if not args.skip_substage:
        if not args.npz.is_file():
            raise SystemExit(f"Missing npz: {args.npz}")
        tag = args.npz.parent.name  # seed dir e.g. "3"
        k_tag = args.npz.parents[3].name if len(args.npz.parents) > 3 else "K?"
        sub_dir = args.out_root / "substages" / f"{k_tag}_seed{tag}"
        sub_dir.mkdir(parents=True, exist_ok=True)

        analysis_out = run_full_thesis_analysis(args.npz, sub_dir / "analysis", k=4)
        manifest["outputs"]["substage_analysis"] = analysis_out

        pub_dir = sub_dir / "publication"
        pub_dir.mkdir(parents=True, exist_ok=True)
        run_compact_panels(
            args.npz,
            pub_dir / "fig3_substage_panels.pdf",
            pub_dir / "fig4_transition_matrix.pdf",
            pub_dir / "fig4_hypnogram.pdf",
            pub_dir / "fig4_combined.pdf",
        )
        manifest["outputs"]["publication"] = {
            "fig3": str(pub_dir / "fig3_substage_panels.pdf"),
            "fig4_combined": str(pub_dir / "fig4_combined.pdf"),
            "fig4_hypnogram": str(pub_dir / "fig4_hypnogram.pdf"),
        }
        print(f"Substage analysis: {sub_dir}")

    manifest_path = args.out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest: {manifest_path}")

    if args.copy_to_paper:
        paper_fig = REPO / "docs/paper/figures"
        overleaf = REPO / "paper/overleaf/figures"
        (paper_fig / "k_sweep").mkdir(parents=True, exist_ok=True)
        (paper_fig / "archive").mkdir(parents=True, exist_ok=True)
        overleaf.mkdir(parents=True, exist_ok=True)
        copies = []
        if "k_sweep" in manifest["outputs"]:
            copies.append(
                (Path(manifest["outputs"]["k_sweep"]["dual_axis_pdf"]), paper_fig / "k_sweep/k_sweep_dual_axis.pdf")
            )
        if "publication" in manifest["outputs"]:
            pub = manifest["outputs"]["publication"]
            copies += [
                (Path(pub["fig3"]), paper_fig / "archive/Fig3_substage_panels.pdf"),
                (Path(pub["fig4_combined"]), paper_fig / "archive/Fig4_combined.pdf"),
            ]
        for src, dst in copies:
            if src.is_file():
                shutil.copy(src, dst)
                shutil.copy(src, overleaf / dst.name)
        print(f"Copied {len(copies)} figures to docs/paper/figures/ and paper/overleaf/figures/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
