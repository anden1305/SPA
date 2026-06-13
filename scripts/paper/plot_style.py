"""Shared matplotlib style for PLOS / thesis paper figures."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Ladder palette (consistent across Fig 1C and Fig 2)
LADDER_COLORS = {
    "hmm_features": "#6B7280",
    "hmmgmvae_locked": "#059669",
    "cgmvae_locked": "#2563EB",
    "chmmgmvae_locked": "#7C3AED",
}
LADDER_LABELS = {
    "hmm_features": "HMM (features)",
    "hmmgmvae_locked": "HMMGMVAE",
    "cgmvae_locked": "cGMVAE",
    "chmmgmvae_locked": "cHMM–GMVAE",
}
# Thesis expert-feature HMM (~0.51) used a different protocol; not plotted as holdout baseline.
THESIS_HMM_REF: float | None = None

SUBSTAGE_COLORS = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
    "#86BCB6", "#D37295", "#A0CBE8", "#FABFD2", "#B6992D",
]

# Macro true-label colours used in physiology grids (Awake/NREM/REM/Artifact).
MACRO_LABEL_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#e41a1c"]

# Subject-mix column: avoid tab20 greens/blues that collide with REM/Wake in true-label mix.
SUBJECT_MIX_CMAP = "twilight_shifted"

# Default text scale for thesis Fig 27 / frequency_plot grids.
FREQUENCY_GRID_FONT_SCALE = 1.55


def apply_paper_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(ax, letter: str, x: float = -0.12, y: float = 1.08) -> None:
    ax.text(
        x, y, letter, transform=ax.transAxes,
        fontsize=12, fontweight="bold", va="top", ha="left",
    )


def save_figure(fig: plt.Figure, path, dpi: int = 300) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    if path.suffix.lower() == ".pdf":
        fig.savefig(path.with_suffix(".tif"), dpi=dpi, pil_kwargs={"compression": "tiff_lzw"})
    elif path.suffix.lower() in (".tif", ".tiff", ".png"):
        fig.savefig(path.with_suffix(".pdf"), dpi=min(dpi, 200))
