#!/usr/bin/env python3
"""Split results.npz into per-mouse folders using val_datasets from config.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.helpers.align_labels import align_labels_hungarian
from src.helpers.model_display_name import model_display_name_from_dict
from src.helpers.nmi import calculate_nmi
from src.visuals.pca_tripanel import save_pca_tripanel_arrays


def subject_id_map_from_config(config_path: Path) -> dict[int, str]:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    val_ids: list[str] = []
    for entry in cfg.get("val_datasets", []):
        pid = entry["id"]
        if pid not in val_ids:
            val_ids.append(pid)
    return {i: pid for i, pid in enumerate(val_ids)}


def split_per_mouse(
    plots_dir: Path,
    out_dir: Path,
    id_map: dict[int, str],
    config: dict | None = None,
) -> None:
    npz_path = plots_dir / "results.npz"
    if not npz_path.exists():
        print(f"Skip (no results.npz): {plots_dir}")
        return

    display_name = model_display_name_from_dict(config or {})
    data = np.load(npz_path, allow_pickle=True)
    y_true = data["y_true"].astype(int).ravel()
    y_hat = data["y_hat"].astype(int).ravel()
    x_latent = data["x_latent"]
    sub_ids = data["sub_ids"].astype(int).ravel()
    if x_latent.ndim == 3:
        x_latent = x_latent.reshape(-1, x_latent.shape[-1])

    out_dir.mkdir(parents=True, exist_ok=True)
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
        np.savez(
            mouse_dir / "predictions.npz",
            y_true=yt,
            y_hat=yp,
            x_latent=xl,
            participant_id=mouse_label,
        )
        with (mouse_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump({"nmi": nmi, "n_epochs": int(len(yt)), "participant_id": mouse_label}, f, indent=2)

        if xl.shape[0] >= 3 and xl.shape[1] >= 2:
            save_pca_tripanel_arrays(
                xl,
                yt,
                out_dir=mouse_dir,
                display_name=display_name,
                y_pred=yp,
            )

    print(f"Wrote per-mouse outputs under {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plots-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    id_map = subject_id_map_from_config(args.config)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    split_per_mouse(args.plots_dir, args.out_dir, id_map, config=cfg)


if __name__ == "__main__":
    main()
