#!/usr/bin/env python3
"""S2 Fig: per-mouse prior NMI (fold 4 holdout) — cGMVAE vs cHMM–GMVAE."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.paper.plot_style import (
    LADDER_COLORS,
    LADDER_LABELS,
    apply_paper_style,
    save_figure,
)

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CHMM = REPO / "results/cv4fold/paper_figures/per_mouse/fold_4/chmmgmvae_locked/seed_1"
DEFAULT_CGMVAE = REPO / "results/cv4fold/paper_figures/per_mouse/fold_4/cgmvae_locked/seed_1"

LAB_SHORT = {"lab_2": "Bern", "lab_3": "Cph", "lab_5": "Lyon"}
LAB_TINT = {"lab_2": "#FFF4E8", "lab_3": "#EFF8EF", "lab_5": "#F6EFF8"}
MOUSE_ORDER = ["id_80", "id_81", "id_56", "id_59", "id_60", "id_92"]


def _load_per_mouse(root: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    if not root.is_dir():
        return out
    for mouse_dir in sorted(root.iterdir()):
        mf = mouse_dir / "metrics.json"
        if not mf.is_file():
            continue
        payload = json.loads(mf.read_text(encoding="utf-8"))
        pid = str(payload.get("participant_id", mouse_dir.name))
        out[pid] = float(payload["nmi"])
    return out


def _lab_from_config(config_path: Path, participant_id: str) -> str:
    if not config_path.is_file():
        return "unknown"
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    for entry in cfg.get("val_datasets", []):
        if entry.get("id") == participant_id or entry.get("participant_id") == participant_id:
            lab = entry.get("lab") or entry.get("laboratory")
            if lab:
                return str(lab).replace("laboratory_", "lab_")
    return "unknown"


def _mouse_label(pid: str) -> str:
    m = re.search(r"(\d+)", pid)
    return m.group(1) if m else pid.replace("id_", "")


def plot_per_mouse(
    chmm_root: Path,
    cgmvae_root: Path,
    out_path: Path,
    *,
    chmm_config: Path | None = None,
    cgmvae_config: Path | None = None,
) -> None:
    apply_paper_style()
    chmm = _load_per_mouse(chmm_root)
    cgmvae = _load_per_mouse(cgmvae_root)
    if not chmm or not cgmvae:
        raise FileNotFoundError("Missing per-mouse metrics — run split_per_mouse.py first")

    cfg = chmm_config or chmm_root.parents[3] / "config.json"
    if not cfg.is_file():
        cfg = cgmvae_config or cgmvae_root.parents[3] / "config.json"

    mice = [m for m in MOUSE_ORDER if m in chmm or m in cgmvae]
    if not mice:
        mice = sorted(set(chmm) | set(cgmvae))

    labs = [_lab_from_config(cfg, m) for m in mice]
    cgmv_vals = [cgmvae.get(m, np.nan) for m in mice]
    chmm_vals = [chmm.get(m, np.nan) for m in mice]

    fig, ax = plt.subplots(figsize=(4.6, 2.05))
    x = np.arange(len(mice))
    width = 0.34

    ax.bar(
        x - width / 2,
        cgmv_vals,
        width,
        label=LADDER_LABELS["cgmvae_locked"],
        color=LADDER_COLORS["cgmvae_locked"],
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )
    ax.bar(
        x + width / 2,
        chmm_vals,
        width,
        label=LADDER_LABELS["chmmgmvae_locked"],
        color=LADDER_COLORS["chmmgmvae_locked"],
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )

    # Lab background bands
    lab_spans: list[tuple[str, int, int]] = []
    start = 0
    for i in range(1, len(labs) + 1):
        if i == len(labs) or labs[i] != labs[start]:
            lab_spans.append((labs[start], start, i - 1))
            start = i
    for lab, i0, i1 in lab_spans:
        ax.axvspan(i0 - 0.5, i1 + 0.5, color=LAB_TINT.get(lab, "#F8FAFC"), zorder=0, lw=0)
        mid = 0.5 * (i0 + i1)
        ax.text(
            mid,
            0.98,
            LAB_SHORT.get(lab, lab),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7,
            color="#64748B",
        )

    for i, (a, b) in enumerate(zip(cgmv_vals, chmm_vals)):
        if np.isfinite(a):
            ax.text(i - width / 2, a + 0.02, f"{a:.2f}", ha="center", va="bottom", fontsize=6, color="#334155")
        if np.isfinite(b):
            ax.text(i + width / 2, b + 0.02, f"{b:.2f}", ha="center", va="bottom", fontsize=6,
                    color=LADDER_COLORS["chmmgmvae_locked"], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([_mouse_label(m) for m in mice], fontsize=7.5)
    ax.set_xlabel("Held-out mouse", fontsize=8)
    ax.set_ylabel("Prior NMI", fontsize=8)
    ax.set_ylim(0, 0.98)
    ax.set_xlim(-0.6, len(mice) - 0.4)
    ax.tick_params(axis="both", labelsize=7)
    ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=2, handlelength=1.2, columnspacing=0.8)
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.35, zorder=1)

    save_figure(fig, out_path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chmm-root", type=Path, default=DEFAULT_CHMM)
    parser.add_argument("--cgmvae-root", type=Path, default=DEFAULT_CGMVAE)
    parser.add_argument("--out", type=Path, default=REPO / "docs/paper/figures/supplementary/S4_per_mouse_nmi.pdf")
    args = parser.parse_args()
    plot_per_mouse(args.chmm_root, args.cgmvae_root, args.out)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
