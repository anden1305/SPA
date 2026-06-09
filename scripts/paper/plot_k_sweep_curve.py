#!/usr/bin/env python3
"""K substage sweep curve (S3 Fig / Fig 3 backup) from paper_k_sweep or holdout results."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO / "results/cv4fold/paper_k_sweep/fold_4"
K_VALUES = [3, 5, 7, 9, 11, 13, 15]


def _read_best_nmi(run_dir: Path) -> float | None:
    metrics = run_dir / "plots" / "metrics.txt"
    if not metrics.is_file():
        return None
    text = metrics.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"prior[_\s]*nmi[:\s]+([0-9.]+)", text, re.I)
    if m:
        return float(m.group(1))
    m = re.search(r"checkpoint[_\s]*score[:\s]+([0-9.]+)", text, re.I)
    return float(m.group(1)) if m else None


def collect_k_sweep(root: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    if not root.is_dir():
        return out
    for k in K_VALUES:
        for pattern in (f"chmmgmvae_K{k}", f"K{k}"):
            for model_dir in root.glob(f"**/{pattern}"):
                if not model_dir.is_dir():
                    continue
                runs = sorted(model_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
                for run in runs:
                    if not run.is_dir():
                        continue
                    nmi = _read_best_nmi(run)
                    if nmi is not None:
                        out[k] = max(out.get(k, 0.0), nmi)
                        break
                if k in out:
                    break
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "docs/paper/figures/S3_k_sweep.pdf",
    )
    parser.add_argument("--chosen-k", type=int, default=None, help="Mark chosen K with vertical line")
    args = parser.parse_args()

    scores = collect_k_sweep(args.root)
    if not scores:
        # Placeholder curve for manuscript pipeline until jobs finish
        scores = {k: 0.45 + 0.02 * np.sin(k) for k in K_VALUES}
        placeholder = True
    else:
        placeholder = False

    ks = sorted(scores.keys())
    vals = [scores[k] for k in ks]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(ks, vals, "o-", color="#984ea3", linewidth=2, markersize=8)
    if args.chosen_k:
        ax.axvline(args.chosen_k, color="#333", linestyle="--", label=f"chosen K={args.chosen_k}")
        ax.legend()
    ax.set_xlabel("Mixture components K")
    ax.set_ylabel("Best prior NMI (validation)")
    title = "Substage resolution sweep (cHMM--GMVAE, joint fold 4)"
    if placeholder:
        title += " [placeholder — refresh after K-sweep jobs]"
    ax.set_title(title)
    ax.set_xticks(K_VALUES)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    fig.savefig(args.out.with_suffix(".tif"), dpi=300, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    staging = REPO / "paper/overleaf/figures/s3_k_sweep.pdf"
    staging.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(args.out, staging)
    print(f"Wrote {args.out} (placeholder={placeholder}, n_K={len(scores)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
