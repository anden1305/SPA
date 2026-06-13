#!/usr/bin/env python3
"""Build Fig 3: 2×2 population latent grid (K=6, 9, 12 PCA + K=12 hypnogram)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from scripts.paper.plot_style import apply_paper_style, save_figure

REPO = Path(__file__).resolve().parents[2]
BIO_ROOT = REPO / "results/cv4fold/paper_figures/biology_meeting_population"


def _panel_path(k: int, kind: str) -> Path:
    kdir = BIO_ROOT / f"K{k:02d}"
    if kind == "pca":
        path = kdir / f"K{k}_pca_comparison_true_vs_predicted.png"
    elif kind == "hypnogram":
        for name in ("hypnogram.tif", "hypnogram.png", "hypnogram.pdf"):
            path = kdir / name
            if path.is_file():
                return path
        raise FileNotFoundError(f"Missing hypnogram panel under {kdir}")
    else:
        raise ValueError(kind)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    return path


def _image_aspect(path: Path) -> float:
    img = mpimg.imread(path)
    h, w = img.shape[:2]
    return w / h


def _panel_label_below(ax, letter: str) -> None:
    ax.text(
        0.5,
        -0.045,
        letter,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11,
        fontweight="normal",
        family="serif",
    )


def build_grid(out_path: Path) -> None:
    panels: list[tuple[str, Path]] = [
        ("A", _panel_path(6, "pca")),
        ("B", _panel_path(9, "pca")),
        ("C", _panel_path(12, "pca")),
        ("D", _panel_path(12, "hypnogram")),
    ]

    apply_paper_style()
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
            "font.size": 10,
        }
    )

    fig_w = 7.0
    col_w = fig_w * 0.47
    aspects = [_image_aspect(path) for _, path in panels]
    row_hs = [
        col_w / min(aspects[0], aspects[1]),
        col_w / min(aspects[2], aspects[3]),
    ]
    label_pad = 0.28
    row_gap = 0.10
    fig_h = row_hs[0] + row_hs[1] + row_gap + label_pad

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=row_hs,
        hspace=row_gap / min(row_hs),
        wspace=0.05,
        left=0.02,
        right=0.98,
        top=0.99,
        bottom=label_pad / fig_h,
    )

    for idx, (label, path) in enumerate(panels):
        r, c = divmod(idx, 2)
        ax = fig.add_subplot(gs[r, c])
        img = mpimg.imread(path)
        h, w = img.shape[:2]
        ax.imshow(img, aspect="equal", interpolation="antialiased")
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.axis("off")
        _panel_label_below(ax, label)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "docs/paper/figures/main/Fig_population_pca_2x2.png",
    )
    args = parser.parse_args()
    build_grid(args.out)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
