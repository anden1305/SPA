"""Thesis Figure 27: substage physiology grid from results.npz (GMM/HMM predicted states)."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.substage_analysis.frequency_plot import plot_label_channel_frequency_grid

REPO = Path(__file__).resolve().parents[2]
METADATA_CSV = REPO / "data/ds006366_processed/metadata.csv"

LABEL_COLORS_TRUE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#e41a1c"]
LABEL_COLORS_PRED = [
    "#6A3D9A", "#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E",
    "#E6AB02", "#A6761D", "#666666", "#8DD3C7", "#BEBADA", "#FB8072",
    "#80B1D3", "#FDB462", "#B3B3B3", "#17BECF", "#BCBD22",
]
LAB_COLORS = {
    "lab_1": "#4E79A7",
    "lab_2": "#F28E2B",
    "lab_3": "#59A14F",
    "lab_4": "#E15759",
    "lab_5": "#B07AA1",
}
EPOCH_SEC = 4.0
WINDOW_LEN = 64


@lru_cache(maxsize=1)
def load_subject_num_to_lab() -> dict[int, str]:
    """Map legacy sub_id integer (e.g. 56 from sub-056) → lab label."""
    if not METADATA_CSV.is_file():
        return {}
    df = pd.read_csv(METADATA_CSV)
    out: dict[int, str] = {}
    for pid, lab in df.drop_duplicates("participant_id")[["participant_id", "lab"]].itertuples(index=False):
        m = re.search(r"\d+", str(pid))
        if m:
            out[int(m.group())] = str(lab)
    return out


def prepare_active_sorted_labels(y_pred: np.ndarray) -> np.ndarray:
    """
    Keep only HMM/GMM states with assignments; sort by count descending.
    Substage 1 = largest occupancy, then 2, …
    """
    y = y_pred.astype(int)
    counts = np.bincount(y)
    active = np.flatnonzero(counts > 0)
    if active.size == 0:
        return y[:0]
    order = active[np.argsort(-counts[active])]
    mapping = {int(old): new for new, old in enumerate(order)}
    return np.array([mapping[int(v)] for v in y], dtype=int)


def _sequence_start_mask(n_epochs: int, sub_ids: np.ndarray) -> np.ndarray:
    mask = np.zeros(n_epochs, dtype=bool)
    mask[0] = True
    if n_epochs > 1:
        mask[1:] = sub_ids[1:] != sub_ids[:-1]
    mask[np.arange(n_epochs) % WINDOW_LEN == 0] = True
    return mask


def plot_fig27_gmm_predicted(
    npz_path: Path,
    save_path: Path,
    *,
    k: int | None = None,
    epoch_sec: float = EPOCH_SEC,
    nmi: float | None = None,
) -> int:
    """
    Thesis Fig 27 grid: active substates only, sorted by occupancy (high → low).

    Returns number of active substates plotted.
    """
    data = np.load(npz_path)
    y_raw = data["y_hat"].reshape(-1).astype(int)
    y_pred = prepare_active_sorted_labels(y_raw)
    if y_pred.size == 0:
        raise ValueError(f"No active substates in {npz_path}")

    n_active = int(y_pred.max()) + 1
    labels_true = data["y_true"].reshape(-1).astype(int)
    raw = data["x"].reshape(-1, data["x"].shape[2], data["x"].shape[3])
    sub_ids = data["sub_ids"].reshape(-1)

    sub_to_lab = load_subject_num_to_lab()
    lab_ids = np.array([sub_to_lab.get(int(s), "unknown") for s in sub_ids], dtype=object)

    n_true = int(labels_true.max()) + 1
    label_names_true = ["Awake", "NREM", "REM", "Artifact"][:n_true]
    label_names_pred = [f"Substage {i + 1}" for i in range(n_active)]
    colors = (LABEL_COLORS_PRED * ((n_active // len(LABEL_COLORS_PRED)) + 1))[:n_active]

    title = "Features of GMM Predicted Substages"
    parts = []
    if k is not None:
        parts.append(f"configured K={k}")
    parts.append(f"{n_active} active")
    if nmi is not None:
        parts.append(f"holdout NMI={nmi:.3f}")
    if parts:
        title += f" ({', '.join(parts)})"

    plot_label_channel_frequency_grid(
        raw_datapoints=raw,
        sample_rate=128.0,
        y_pred=y_pred,
        sub_ids=sub_ids,
        lab_ids=lab_ids,
        label_names=label_names_pred,
        channel_names=["EEG1", "EEG2", "EMG"],
        channel_freq_ranges=[(0.0, 30), (0.0, 30), (5, 60)],
        plot_title=title,
        save_path=str(save_path),
        input_scale="linear",
        y_scale="linear",
        y_lim_mode="minmax",
        labels_true=labels_true,
        label_names_true=label_names_true,
        label_colors_true=LABEL_COLORS_TRUE[:n_true],
        label_colors=colors,
        lab_colors=LAB_COLORS,
        sequence_start_mask=_sequence_start_mask(raw.shape[0], sub_ids),
        bout_length_epoch_sec=epoch_sec,
        feature_titles=(
            "EEG1 theta/delta",
            "EEG2 theta/delta",
            "EMG Power",
            "Bout length (seconds)",
        ),
        subject_dist_col_title="Subject mix (inv-total weighted)",
        lab_dist_col_title="Lab mix (inv-total weighted)",
        figsize_per_row=2.6,
    )
    return n_active
