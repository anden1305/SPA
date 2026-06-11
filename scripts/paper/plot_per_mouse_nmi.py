#!/usr/bin/env python3
"""S4 Fig: per-mouse prior NMI (fold 4 holdout) — cGMVAE vs cHMM–GMVAE."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.paper.plot_style import LADDER_COLORS, apply_paper_style, panel_label, save_figure

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CHMM = REPO / "results/cv4fold/paper_figures/per_mouse/fold_4/chmmgmvae_locked/seed_1"
DEFAULT_CGMVAE = REPO / "results/cv4fold/paper_figures/per_mouse/fold_4/cgmvae_locked/seed_1"


def _load_per_mouse(root: Path) -> list[dict]:
    rows: list[dict] = []
    if not root.is_dir():
        return rows
    for mouse_dir in sorted(root.iterdir()):
        mf = mouse_dir / "metrics.json"
        if not mf.is_file():
            continue
        payload = json.loads(mf.read_text(encoding="utf-8"))
        pid = payload.get("participant_id", mouse_dir.name)
        rows.append({"participant_id": pid, "nmi": float(payload["nmi"])})
    return rows


def _lab_from_config(config_path: Path, participant_id: str) -> str:
    if not config_path.is_file():
        return "unknown"
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    for entry in cfg.get("val_datasets", []):
        if entry.get("id") == participant_id or entry.get("participant_id") == participant_id:
            lab = entry.get("lab") or entry.get("laboratory")
            if lab:
                return str(lab).replace("laboratory_", "lab_")
    # Fallback: infer from dataset path
    for entry in cfg.get("val_datasets", []):
        ds = str(entry.get("dataset", ""))
        m = re.search(r"lab[_-]?(\d+)", ds, re.I)
        if m and participant_id in str(entry.get("id", "")):
            return f"lab_{m.group(1)}"
    return "unknown"


def plot_per_mouse(
    chmm_root: Path,
    cgmvae_root: Path,
    out_path: Path,
    *,
    chmm_config: Path | None = None,
    cgmvae_config: Path | None = None,
) -> None:
    apply_paper_style()
    chmm_rows = _load_per_mouse(chmm_root)
    cgmvae_rows = _load_per_mouse(cgmvae_root)
    if not chmm_rows or not cgmvae_rows:
        raise FileNotFoundError("Missing per-mouse metrics — run split_per_mouse.py first")

    cfg = chmm_config or chmm_root.parents[3] / "config.json"
    if not cfg.is_file():
        cfg = cgmvae_config or cgmvae_root.parents[3] / "config.json"

    by_id: dict[str, dict[str, float]] = {}
    for r in cgmvae_rows:
        by_id.setdefault(r["participant_id"], {})["cgmvae"] = r["nmi"]
    for r in chmm_rows:
        by_id.setdefault(r["participant_id"], {})["chmm"] = r["nmi"]

    labs = sorted({_lab_from_config(cfg, pid) for pid in by_id})
    if labs == ["unknown"]:
        labs = ["all"]

    fig, axes = plt.subplots(1, len(labs), figsize=(2.4 * len(labs), 3.0), sharey=True)
    if len(labs) == 1:
        axes = [axes]

    for ax, lab in zip(axes, labs):
        mice = [pid for pid in by_id if lab == "all" or _lab_from_config(cfg, pid) == lab]
        if not mice:
            continue
        x = np.arange(len(mice))
        w = 0.35
        cgmv = [by_id[m].get("cgmvae", np.nan) for m in mice]
        chmm = [by_id[m].get("chmm", np.nan) for m in mice]
        ax.bar(x - w / 2, cgmv, w, label="cGMVAE", color=LADDER_COLORS["cgmvae_locked"])
        ax.bar(x + w / 2, chmm, w, label="cHMM–GMVAE", color=LADDER_COLORS["chmmgmvae_locked"])
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("id_", "M") for m in mice], rotation=45, ha="right", fontsize=7)
        ax.set_ylim(0, 0.95)
        ax.set_title(lab.replace("_", " "), fontsize=9)
        if ax is axes[0]:
            panel_label(ax, "A", x=-0.22, y=1.12)
        for i, (a, b) in enumerate(zip(cgmv, chmm)):
            if np.isfinite(a) and np.isfinite(b) and b > a:
                ax.annotate("", xy=(i + w / 2, b), xytext=(i - w / 2, a),
                            arrowprops=dict(arrowstyle="-|>", color="#64748B", lw=0.8))

    axes[0].set_ylabel("Prior NMI", fontsize=8)
    axes[-1].legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle("Fold 4 held-out mice", fontsize=9, y=1.02)
    save_figure(fig, out_path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chmm-root", type=Path, default=DEFAULT_CHMM)
    parser.add_argument("--cgmvae-root", type=Path, default=DEFAULT_CGMVAE)
    parser.add_argument("--out", type=Path, default=REPO / "docs/paper/figures/curated/S4_per_mouse_nmi.pdf")
    args = parser.parse_args()
    plot_per_mouse(args.chmm_root, args.cgmvae_root, args.out)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
