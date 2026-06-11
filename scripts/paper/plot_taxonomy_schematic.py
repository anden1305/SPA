#!/usr/bin/env python3
"""Thesis Fig 30-style substage taxonomy schematic from cHMM-GMVAE validation NPZ.

Nodes are placed at per-substage latent PCA centroids; directed edges use the
change-only transition matrix (same convention as ``transition_matrix.py``).
Arrow weight encodes transition probability tiers from the thesis caption.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

from scripts.paper.plot_style import apply_paper_style, save_figure

MACRO_NAMES = ["Awake", "NREM", "REM", "Artifact"]
ROLE_COLORS = {
    "awake": "#3B82F6",
    "nrem": "#F59E0B",
    "rem": "#10B981",
    "artifact": "#EF4444",
}
TRANSITION_BLEND = {
    ("awake", "nrem"): "#7C9FD6",
    ("nrem", "awake"): "#7C9FD6",
    ("nrem", "rem"): "#6BBF9A",
    ("rem", "nrem"): "#6BBF9A",
    ("rem", "awake"): "#5BA8C8",
    ("awake", "rem"): "#5BA8C8",
    ("awake", "artifact"): "#C08484",
    ("nrem", "artifact"): "#C08484",
    ("rem", "artifact"): "#C08484",
}


def remap_labels(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y.astype(int))
    ranked_old = np.argsort(-counts)
    ranked_old = ranked_old[counts[ranked_old] > 0]
    mapping = np.empty(counts.shape[0], dtype=int)
    mapping[ranked_old] = np.arange(len(ranked_old))
    return mapping[y.astype(int)]


def _bout_lengths(y: np.ndarray) -> np.ndarray:
    """Per-epoch bout length (consecutive same label)."""
    n = y.size
    if n == 0:
        return np.array([], dtype=np.float64)
    out = np.ones(n, dtype=np.float64)
    run_start = 0
    for i in range(1, n + 1):
        if i == n or y[i] != y[run_start]:
            out[run_start:i] = i - run_start
            run_start = i
    return out


def _transition_matrix(y: np.ndarray, n_states: int) -> np.ndarray:
    src, dst = y[:-1], y[1:]
    ch = src != dst
    src, dst = src[ch], dst[ch]
    counts = np.zeros((n_states, n_states), dtype=np.float64)
    if src.size:
        np.add.at(counts, (src, dst), 1.0)
    row_sums = counts.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = counts / row_sums
    m[row_sums[:, 0] == 0, :] = np.nan
    np.fill_diagonal(m, np.nan)
    return m


def _macro_mix(y_true: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    sub = y_true[mask]
    if sub.size == 0:
        return {name: 0.0 for name in MACRO_NAMES}
    counts = np.bincount(np.clip(sub.astype(int), 0, len(MACRO_NAMES) - 1), minlength=len(MACRO_NAMES))
    frac = counts / counts.sum()
    return {MACRO_NAMES[i]: float(frac[i]) for i in range(len(MACRO_NAMES))}


def _dominant_macro(mix: dict[str, float]) -> str:
    core = {k: v for k, v in mix.items() if k != "Artifact"}
    return max(core, key=core.get)


def _second_macro(mix: dict[str, float], first: str) -> str:
    core = {k: v for k, v in mix.items() if k not in ("Artifact", first)}
    return max(core, key=core.get) if core else first


def _role_key(name: str) -> str:
    return name.lower().replace(" ", "_")


def auto_taxonomy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    min_prevalence: float = 0.005,
    stable_bout_threshold: float = 4.0,
    stable_macro_threshold: float = 0.6,
) -> dict[int, dict[str, Any]]:
    n_states = int(y_pred.max()) + 1
    bouts = _bout_lengths(y_pred)
    total = y_pred.size
    taxonomy: dict[int, dict[str, Any]] = {}

    for s in range(n_states):
        mask = y_pred == s
        prev = float(mask.sum()) / total if total else 0.0
        mix = _macro_mix(y_true, mask)
        dom = _dominant_macro(mix)
        sec = _second_macro(mix, dom)
        med_bout = float(np.median(bouts[mask])) if mask.any() else 0.0

        entry: dict[str, Any] = {
            "prevalence": prev,
            "macro_mix": mix,
            "median_bout": med_bout,
            "exclude": prev < min_prevalence,
        }

        if entry["exclude"]:
            entry["role"] = "excluded"
        elif med_bout < stable_bout_threshold or mix[dom] < stable_macro_threshold:
            pair = tuple(sorted([_role_key(dom), _role_key(sec)]))
            entry["role"] = "transition"
            entry["transition_pair"] = pair
        else:
            entry["role"] = _role_key(dom)

        taxonomy[s + 1] = entry
    return taxonomy


def _node_color(role: str, transition_pair: tuple[str, str] | None = None) -> str:
    if role == "transition" and transition_pair:
        return TRANSITION_BLEND.get(transition_pair, "#9CA3AF")
    return ROLE_COLORS.get(role, "#9CA3AF")


def _pca_positions(x_latent: np.ndarray, y_pred: np.ndarray, n_states: int) -> np.ndarray:
    x = x_latent.reshape(-1, x_latent.shape[-1]).astype(np.float64)
    mu = x.mean(axis=0)
    xc = x - mu
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    scores = xc @ vt.T
    pos = np.zeros((n_states, 2), dtype=np.float64)
    for s in range(n_states):
        mask = y_pred == s
        if mask.any():
            pos[s] = scores[mask, :2].mean(axis=0)
    span = np.ptp(pos, axis=0)
    span[span == 0] = 1.0
    pos = (pos - pos.mean(axis=0)) / span
    return pos


def _arrow_style(p: float) -> tuple[str, float, float, float] | None:
    if not np.isfinite(p) or p <= 0.05:
        return None
    if p > 0.3:
        return ("black", 2.4, 0.035, 0.025)
    if p > 0.1:
        return ("black", 1.6, 0.028, 0.020)
    return ("#6B7280", 1.0, 0.022, 0.016)


def plot_taxonomy_schematic(
    npz_path: Path,
    out_path: Path,
    *,
    taxonomy_override: dict | None = None,
    title: str = "Substage taxonomy (latent layout + transitions)",
) -> dict[str, Any]:
    data = np.load(npz_path)
    y_true = data["y_true"].reshape(-1)
    y_pred = remap_labels(data["y_hat"].reshape(-1))
    x_latent = data["x_latent"]
    n_states = int(y_pred.max()) + 1

    taxonomy = auto_taxonomy(y_true, y_pred)
    if taxonomy_override:
        for key, val in taxonomy_override.items():
            idx = int(key)
            if idx in taxonomy and isinstance(val, dict):
                taxonomy[idx].update(val)

    active = [s for s in range(1, n_states + 1) if not taxonomy[s].get("exclude", False)]
    if len(active) < 2:
        raise SystemExit(
            f"Need ≥2 non-excluded substages for taxonomy plot; got {len(active)} at K={n_states}. "
            "Use a higher-K cHMM-GMVAE K-sweep run (e.g. K=7 or K=13)."
        )

    active_idx = [s - 1 for s in active]
    pos_full = _pca_positions(x_latent, y_pred, n_states)
    pos = pos_full[active_idx]
    trans = _transition_matrix(y_pred, n_states)

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    ax.set_aspect("equal")
    ax.axis("off")

    idx_map = {s: i for i, s in enumerate(active)}
    for i, s in enumerate(active):
        meta = taxonomy[s]
        color = _node_color(meta.get("role", "awake"), meta.get("transition_pair"))
        circle = Circle(pos[i], 0.08, facecolor=color, edgecolor="black", linewidth=1.2, zorder=3)
        ax.add_patch(circle)
        ax.text(pos[i, 0], pos[i, 1], str(s), ha="center", va="center", fontsize=9, fontweight="bold", zorder=4)
        role = meta.get("role", "")
        if role == "transition":
            label = f"S{s}\n(transition)"
        elif role == "excluded":
            label = f"S{s}\n(excluded)"
        else:
            label = f"S{s}\n({role})"
        ax.text(pos[i, 0], pos[i, 1] - 0.14, label, ha="center", va="top", fontsize=7, zorder=4)

    for src in active:
        for dst in active:
            if src == dst:
                continue
            p = trans[src - 1, dst - 1]
            style = _arrow_style(p)
            if style is None:
                continue
            color, lw, scale, shrink = style
            i, j = idx_map[src], idx_map[dst]
            arr = FancyArrowPatch(
                pos[i],
                pos[j],
                arrowstyle="-|>",
                mutation_scale=scale * 100,
                shrinkA=shrink * 100,
                shrinkB=shrink * 100,
                linewidth=lw,
                color=color,
                zorder=2,
            )
            ax.add_patch(arr)

    pad = 0.35
    ax.set_xlim(pos[:, 0].min() - pad, pos[:, 0].max() + pad)
    ax.set_ylim(pos[:, 1].min() - pad, pos[:, 1].max() + pad)
    ax.set_title(title, fontsize=10, pad=12)
    ax.text(
        0.02,
        0.02,
        "Edges: p>0.05 shown (black: >0.3 / >0.1; grey: 0.05–0.1)\n"
        "Layout: PCA centroids of latent μ per substage",
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
        color="#374151",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path)
    plt.close(fig)

    summary = {
        "npz": str(npz_path),
        "n_states": n_states,
        "active_substages": active,
        "taxonomy": taxonomy,
        "transition_matrix": np.nan_to_num(trans, nan=0.0).tolist(),
    }
    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Output PDF path")
    parser.add_argument(
        "--taxonomy-json",
        type=Path,
        default=None,
        help="Optional overrides: {\"3\": {\"role\": \"awake\", \"exclude\": false}, ...}",
    )
    parser.add_argument("--title", type=str, default=None)
    args = parser.parse_args()

    override = None
    if args.taxonomy_json and args.taxonomy_json.is_file():
        override = json.loads(args.taxonomy_json.read_text(encoding="utf-8"))

    title = args.title or f"Substage taxonomy — {args.npz.parent.parent.name}"
    summary = plot_taxonomy_schematic(args.npz, args.out, taxonomy_override=override, title=title)
    print(f"Wrote {args.out}")
    print(f"Wrote {args.out.with_suffix('.json')}")
    print(f"Active substages: {summary['active_substages']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
