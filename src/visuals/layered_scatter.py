"""Per-class scatter z-order and alpha for latent PCA / t-SNE panels."""

from __future__ import annotations

from typing import Sequence

import numpy as np

MACRO_ALPHA: dict[str, float] = {
    "Artifact": 0.20,
    "NREM": 0.60,
    "REM": 0.55,
    "Awake": 0.70,
}

# Bottom (drawn first) to top
MACRO_DRAW_ORDER: tuple[str, ...] = ("Artifact", "NREM", "REM", "Awake")


def draw_order_and_alpha(
    labels: np.ndarray,
    label_names: Sequence[str] | None = None,
) -> list[tuple[int, float]]:
    """Return (class_id, alpha) in draw order (first entry is bottom layer)."""
    labels = np.asarray(labels)
    present = {int(c) for c in np.unique(labels)}

    if label_names and "Artifact" in label_names:
        name_to_idx = {name: i for i, name in enumerate(label_names)}
        ordered: list[tuple[int, float]] = []
        for name in MACRO_DRAW_ORDER:
            if name not in name_to_idx:
                continue
            c = name_to_idx[name]
            if c not in present:
                continue
            ordered.append((c, MACRO_ALPHA.get(name, 0.6)))
        seen = {c for c, _ in ordered}
        for c in sorted(present):
            if c not in seen:
                ordered.insert(0, (c, 0.25))
        return ordered

    counts = {int(c): int((labels == c).sum()) for c in present}
    total = max(sum(counts.values()), 1)
    ordered: list[tuple[int, float]] = []
    for c in sorted(counts, key=lambda k: counts[k]):
        frac = counts[c] / total
        if frac > 0.30:
            alpha = 0.30
        elif frac > 0.15:
            alpha = 0.45
        elif frac > 0.05:
            alpha = 0.55
        else:
            alpha = 0.70
        ordered.append((c, alpha))
    return ordered


def build_color_lut(
    labels: np.ndarray,
    palette: Sequence[str],
) -> dict[int, str]:
    unique = np.unique(labels)
    return {int(v): palette[i % len(palette)] for i, v in enumerate(unique)}


def scatter_layered(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    *,
    label_names: Sequence[str] | None,
    colors: dict[int, str] | None = None,
    palette: Sequence[str] | None = None,
    point_size: float = 6.0,
    zorder_base: int = 1,
) -> dict[int, str]:
    """Draw one scatter per class with class-specific alpha and z-order."""
    labels = np.asarray(labels)
    if colors is None:
        if palette is None:
            raise ValueError("Provide colors or palette")
        colors = build_color_lut(labels, palette)

    for z, (c, alpha) in enumerate(draw_order_and_alpha(labels, label_names)):
        mask = labels == c
        if not np.any(mask):
            continue
        ax.scatter(
            x[mask],
            y[mask],
            s=point_size,
            alpha=alpha,
            c=colors[int(c)],
            edgecolors="none",
            zorder=zorder_base + z,
        )
    return colors
