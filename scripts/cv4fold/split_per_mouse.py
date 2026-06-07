#!/usr/bin/env python3
"""Split validation results.npz into per-mouse metrics (optional heavy artifacts).

Default writes only tiny ``metrics.json`` files (~100 B each). Use ``--save-predictions``
or ``--save-plots`` only when you need full arrays or figures (large on disk).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.cv4fold.metrics_paths import results_npz_path
from src.helpers.align_labels import align_labels_hungarian
from src.helpers.nmi import calculate_nmi


def subject_id_map_from_config(config_path: Path) -> dict[int, str]:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    val_ids: list[str] = []
    for entry in cfg.get("val_datasets", []):
        pid = entry["id"]
        if pid not in val_ids:
            val_ids.append(pid)
    return {i: pid for i, pid in enumerate(val_ids)}


def split_per_mouse(
    result_root: Path,
    run_num: int,
    out_dir: Path,
    id_map: dict[int, str],
    *,
    save_predictions: bool = False,
    save_plots: bool = False,
) -> int:
    npz_file = results_npz_path(result_root, run_num)
    if npz_file is None:
        print(f"Skip run {run_num}: no results.npz under {result_root}")
        return 0

    data = np.load(npz_file, allow_pickle=True)
    y_true = data["y_true"].astype(int).ravel()
    y_hat = data["y_hat"].astype(int).ravel()
    x_latent = data["x_latent"]
    sub_ids = data["sub_ids"].astype(int).ravel()
    if x_latent.ndim == 3:
        x_latent = x_latent.reshape(-1, x_latent.shape[-1])

    out_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0
    for sid in np.unique(sub_ids):
        mouse_label = id_map.get(int(sid), f"id_{int(sid)}")
        mask = sub_ids == sid
        yt = y_true[mask]
        yp = y_hat[mask]
        xl = x_latent[mask]
        try:
            yp_aligned = align_labels_hungarian(yt, yp)
            nmi = float(calculate_nmi(yp_aligned, yt))
        except Exception:
            nmi = float(calculate_nmi(yp, yt))

        mouse_dir = out_dir / mouse_label
        mouse_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "nmi": nmi,
            "n_epochs": int(len(yt)),
            "participant_id": mouse_label,
        }
        (mouse_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        n_written += 1

        if save_predictions:
            np.savez(
                mouse_dir / "predictions.npz",
                y_true=yt,
                y_hat=yp,
                x_latent=xl,
                participant_id=mouse_label,
            )

        if save_plots and xl.shape[0] >= 3 and xl.shape[1] >= 2:
            _save_minimal_pca(mouse_dir / "pca_pc1_pc2.png", xl, yt)

    print(f"Wrote {n_written} per-mouse metrics under {out_dir}")
    return n_written


def _save_minimal_pca(out_path: Path, x_latent: np.ndarray, y_true: np.ndarray) -> None:
    """Lightweight 2D PCA scatter — no tripanel dependency."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    xy = pca.fit_transform(x_latent)
    fig, ax = plt.subplots(figsize=(5, 4))
    scatter = ax.scatter(xy[:, 0], xy[:, 1], c=y_true, cmap="tab10", s=8, alpha=0.7)
    ax.set_title("Latent PCA (PC1 vs PC2)")
    fig.colorbar(scatter, ax=ax, label="true stage")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--run-num", type=int, default=1, choices=(1, 2, 3))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Also write predictions.npz per mouse (large)",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Write minimal pca_pc1_pc2.png per mouse",
    )
    args = parser.parse_args()
    config_path = args.result_root / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    id_map = subject_id_map_from_config(config_path)
    out_dir = args.out_dir or (args.result_root / "per_mouse" / f"run_{args.run_num}")
    split_per_mouse(
        args.result_root,
        args.run_num,
        out_dir,
        id_map,
        save_predictions=args.save_predictions,
        save_plots=args.save_plots,
    )


if __name__ == "__main__":
    main()
