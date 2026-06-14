#!/usr/bin/env python3
"""Build main-text Fig 3: population K=8 PCA + hypnogram (two-panel stack)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from scripts.paper.plot_style import apply_paper_style, save_figure

REPO = Path(__file__).resolve().parents[2]
BIO_ROOT = REPO / "results/cv4fold/paper_figures/biology_meeting_population"
POP_ROOT = REPO / "results/cv4fold/paper_k_sweep/population"


def _panel_path(k: int, kind: str) -> Path:
    kdir = BIO_ROOT / f"K{k:02d}"
    if kind == "pca":
        path = kdir / f"K{k}_pca_comparison_true_vs_predicted.png"
    elif kind == "hypnogram":
        return _ensure_thesis_hypnogram(k)
    else:
        raise ValueError(kind)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    return path


def _ensure_thesis_hypnogram(k: int) -> Path:
    """30 min substage hypnogram with macro-group background bands (thesis Fig 29 style)."""
    from scripts.paper.k_sweep_metrics import best_npz_for_k_merged
    from scripts.paper.plot_thesis_substage_dynamics import plot_hypnogram_from_npz

    kdir = BIO_ROOT / f"K{k:02d}"
    out_pdf = kdir / "hypnogram.pdf"
    hit = best_npz_for_k_merged(POP_ROOT, k)
    if hit is None:
        metrics = kdir / "metrics.json"
        if metrics.is_file():
            import json

            npz = Path(json.loads(metrics.read_text())["npz"])
            if not npz.is_file():
                raise FileNotFoundError(npz)
        else:
            raise FileNotFoundError(f"No population NPZ for K={k}")
    else:
        npz = hit[0]
    plot_hypnogram_from_npz(npz, out_pdf, minutes=30.0)
    out_tif = out_pdf.with_suffix(".tif")
    if not out_tif.is_file():
        raise FileNotFoundError(out_tif)
    return out_tif


def _image_aspect(path: Path) -> float:
    img = mpimg.imread(path)
    h, w = img.shape[:2]
    return w / h


def _panel_label(ax, letter: str) -> None:
    ax.text(
        0.015,
        0.985,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        family="serif",
        bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.9},
        zorder=10,
    )


def build_k8_figure(out_path: Path, *, k: int = 8) -> None:
    panels: list[tuple[str, Path]] = [
        ("A", _panel_path(k, "pca")),
        ("B", _panel_path(k, "hypnogram")),
    ]

    apply_paper_style()
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
            "font.size": 10,
        }
    )

    fig_w = 4.2
    aspects = [_image_aspect(path) for _, path in panels]
    row_hs = [fig_w / asp for asp in aspects]
    row_gap = 0.14
    fig_h = sum(row_hs) + row_gap

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=row_hs,
        hspace=row_gap / min(row_hs),
        left=0.02,
        right=0.98,
        top=0.99,
        bottom=0.02,
    )

    for idx, (label, path) in enumerate(panels):
        ax = fig.add_subplot(gs[idx, 0])
        img = mpimg.imread(path)
        h, w = img.shape[:2]
        ax.imshow(img, aspect="equal", interpolation="antialiased")
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.axis("off")
        _panel_label(ax, label)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "docs/paper/figures/main/Fig_population_K8_latent.png",
    )
    parser.add_argument("--k", type=int, default=8)
    args = parser.parse_args()
    build_k8_figure(args.out, k=args.k)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
