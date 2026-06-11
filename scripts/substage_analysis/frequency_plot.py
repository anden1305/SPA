from __future__ import annotations

from typing import Sequence, Optional, Tuple, List, Union, Literal
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms

ColorLike = Union[str, Tuple[float, float, float], Tuple[float, float, float, float]]


def plot_label_channel_frequency_grid(
    raw_datapoints: np.ndarray,                 # (N, C, F) log(power) by default
    sample_rate: float,
    y_pred: np.ndarray,                         # (N,) predicted label per sample (row groups)
    label_names: Sequence[str],                 # length = n_pred_labels (rows)
    channel_names: Sequence[str],               # length = C
    plot_title: str,
    *,
    # colors for predicted labels (rows)
    label_colors: Sequence[ColorLike],          # length = len(label_names)

    # true labels + names + colors
    labels_true: np.ndarray,                    # (N,)
    label_names_true: Sequence[str],            # length = T
    label_colors_true: Sequence[ColorLike],     # length = T

    # subject IDs for subject distribution column (can be int/str/object)
    sub_ids: np.ndarray,                        # (N,)

    channel_freq_ranges: Sequence[Tuple[float, float]],  # length C
    agg: str = "mean",
    input_scale: str = "log",
    log_eps: float = 0.0,
    normalize_per_row: bool = False,

    # shared y-axis per *spectrum* column
    y_scale: str = "log",
    y_lim_mode: str = "nice_pow10",
    y_clip_percentiles: Tuple[float, float] = (1.0, 99.0),

    # row alpha
    row_alpha: float = 0.95,

    # EEG band overlays on first two channels
    eeg_band_cols: Tuple[int, int] = (0, 1),
    eeg_bands: Sequence[Tuple[str, float, float]] = (
        ("δ", 0.5, 4.0),
        ("θ", 6.0, 9.0),
        ("α", 9.0, 15.0),
        ("β", 15.0, 30.0),
    ),
    eeg_line_alpha: float = 0.35,
    eeg_line_lw: float = 0.8,
    eeg_label_alpha: float = 0.80,
    eeg_label_fontsize: int = 9,
    eeg_label_y: float = 0.985,

    # true-label distribution column
    true_dist_col_title: str = "True label mix",
    true_dist_alpha: float = 0.85,
    true_dist_edgecolor: str = "none",
    show_true_dist_legend: bool = True,

    # subject distribution column
    subject_dist_col_title: str = "Subject mix",
    subject_dist_alpha: float = 0.85,
    subject_dist_edgecolor: str = "none",
    show_subject_dist_legend: bool = True,
    subject_max_legend_items: int = 30,
    subject_sort_legend: bool = True,

    # lab distribution column (optional, right of subject mix)
    lab_ids: Optional[np.ndarray] = None,
    lab_dist_col_title: str = "Lab mix",
    lab_dist_alpha: float = 0.85,
    lab_dist_edgecolor: str = "none",
    show_lab_dist_legend: bool = True,
    lab_colors: Optional[dict[str, ColorLike]] = None,

    # NEW: how to compute subject distribution
    # - "count": standard fraction by sample counts within each label
    # - "inv_total": each sample is weighted by 1 / (total samples of that subject across ALL data)
    #               so each subject has equal total mass globally and cannot dominate
    subject_dist_mode: Literal["count", "inv_total"] = "inv_total",

    # feature columns (horizontal violin plots)
    feature_titles: Tuple[str, str, str, str] = (
        "EEG1 theta/delta",
        "EEG2 theta/delta",
        "EMG Power",
        "Bout length (epochs)",
    ),
    emg_channel_index: Optional[int] = None,    # if None, uses last channel (C-1)
    ratio_eps: float = 1e-12,
    emg_power_scale: str = "log",               # "log" or "linear"
    bout_length_scale: str = "linear",          # "linear" or "log"
    violin_trim_percent: float = 5.0,           # remove bottom/top X% per label per feature
    feature_xlim_pad_frac: float = 0.03,        # small pad on shared x-lims for violins

    # Optional: prevent bouts from crossing concatenated sequence boundaries.
    # True at indices that are the FIRST element of a new sequence.
    sequence_start_mask: Optional[np.ndarray] = None,
    bout_length_epoch_sec: float = 1.0,

    figsize_per_row: float = 2.6,
    save_path: Optional[str] = None,
) -> None:
    """
    Grid: rows = predicted labels.

    Columns:
      - First C columns: 1 Hz binned spectra per channel (row-colored bars)
      - Next 4 columns: horizontal violin plots of derived features:
            1) EEG1 theta/delta (channel 0)
            2) EEG2 theta/delta (channel 1)
            3) EMG Power (channel emg_channel_index)
            4) Bout length (timesteps): run lengths of consecutive identical y_pred values
      - Next column: stacked % distribution of true labels within each predicted label.
      - Second-to-last column: stacked % distribution of subjects within each predicted label.
      - Last column (optional): stacked % lab composition when lab_ids is provided.

    Subject distribution modes:
      - subject_dist_mode="count":
            dist[r,s] = count(y_pred=r & sub=s) / count(y_pred=r)
      - subject_dist_mode="inv_total":
            Each sample i contributes weight w_i = 1 / total_count(subject_i) (global total across ALL N).
            dist[r,s] = sum_{i: y_pred=r & sub=s} w_i  /  sum_{i: y_pred=r} w_i
        => subjects with lots of samples do not dominate simply due to volume.
    """

    # -----------------
    # Validate inputs
    # -----------------
    if not isinstance(raw_datapoints, np.ndarray) or raw_datapoints.ndim != 3:
        raise ValueError(f"raw_datapoints must be a numpy array of shape (N,C,F). Got {getattr(raw_datapoints,'shape',None)}.")
    if not isinstance(y_pred, np.ndarray) or y_pred.ndim != 1:
        raise ValueError(f"y_pred must be a numpy array of shape (N,). Got {getattr(y_pred,'shape',None)}.")
    if not isinstance(labels_true, np.ndarray) or labels_true.ndim != 1:
        raise ValueError(f"labels_true must be a numpy array of shape (N,). Got {getattr(labels_true,'shape',None)}.")
    if not isinstance(sub_ids, np.ndarray) or sub_ids.ndim != 1:
        raise ValueError(f"sub_ids must be a numpy array of shape (N,). Got {getattr(sub_ids,'shape',None)}.")

    N, C, F = raw_datapoints.shape
    if y_pred.shape[0] != N:
        raise ValueError(f"y_pred length must equal N={N}. Got {y_pred.shape[0]}.")
    if labels_true.shape[0] != N:
        raise ValueError(f"labels_true length must equal N={N}. Got {labels_true.shape[0]}.")
    if sub_ids.shape[0] != N:
        raise ValueError(f"sub_ids length must equal N={N}. Got {sub_ids.shape[0]}.")
    if lab_ids is not None:
        if not isinstance(lab_ids, np.ndarray) or lab_ids.ndim != 1 or lab_ids.shape[0] != N:
            raise ValueError(f"lab_ids must be shape (N,). Got {getattr(lab_ids, 'shape', None)}.")
    if len(channel_names) != C:
        raise ValueError(f"channel_names length must equal C={C}. Got {len(channel_names)}.")
    if len(label_names) == 0:
        raise ValueError("label_names must be non-empty.")
    if len(label_names_true) == 0:
        raise ValueError("label_names_true must be non-empty.")
    if len(label_colors) != len(label_names):
        raise ValueError(f"label_colors must have same length as label_names (expected {len(label_names)}, got {len(label_colors)}).")
    if len(label_colors_true) != len(label_names_true):
        raise ValueError(f"label_colors_true must have same length as label_names_true (expected {len(label_names_true)}, got {len(label_colors_true)}).")

    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0.")
    if agg not in ("mean", "median"):
        raise ValueError('agg must be "mean" or "median".')
    if channel_freq_ranges is None or len(channel_freq_ranges) != C:
        raise ValueError(f"channel_freq_ranges must have length C={C}. Got {None if channel_freq_ranges is None else len(channel_freq_ranges)}.")
    if log_eps < 0:
        raise ValueError("log_eps must be >= 0.")
    if y_scale not in ("log", "linear"):
        raise ValueError('y_scale must be "log" or "linear".')
    if y_lim_mode not in ("nice_pow10", "minmax"):
        raise ValueError('y_lim_mode must be "nice_pow10" or "minmax".')
    if emg_power_scale not in ("log", "linear"):
        raise ValueError('emg_power_scale must be "log" or "linear".')
    if bout_length_scale not in ("log", "linear"):
        raise ValueError('bout_length_scale must be "log" or "linear".')
    if not (0.0 <= violin_trim_percent < 50.0):
        raise ValueError("violin_trim_percent must be in [0, 50).")
    if len(feature_titles) != 4:
        raise ValueError("feature_titles must have length 4 (EEG1 ratio, EEG2 ratio, EMG power, Bout length).")
    if subject_dist_mode not in ("count", "inv_total"):
        raise ValueError('subject_dist_mode must be "count" or "inv_total".')

    def _ensure_int(arr: np.ndarray, name: str) -> np.ndarray:
        if np.issubdtype(arr.dtype, np.integer):
            return arr
        if np.all(np.isfinite(arr)) and np.all(np.equal(arr, np.round(arr))):
            return arr.astype(int)
        raise ValueError(f"{name} must be integer dtype (or safely castable to integers).")

    y_pred = _ensure_int(y_pred, "y_pred")
    labels_true = _ensure_int(labels_true, "labels_true")

    L = len(label_names)
    T = len(label_names_true)

    if y_pred.min() < 0 or y_pred.max() >= L:
        raise ValueError(f"y_pred must be in [0, {L-1}]. Got min={y_pred.min()}, max={y_pred.max()}.")
    if labels_true.min() < 0 or labels_true.max() >= T:
        raise ValueError(f"labels_true must be in [0, {T-1}]. Got min={labels_true.min()}, max={labels_true.max()}.")

    if C < 2:
        raise ValueError("Need at least 2 channels to compute EEG1/EEG2 theta/delta features.")

    if sequence_start_mask is not None:
        if not isinstance(sequence_start_mask, np.ndarray) or sequence_start_mask.ndim != 1 or sequence_start_mask.shape[0] != N:
            raise ValueError(f"sequence_start_mask must be a numpy array of shape (N,). Got {getattr(sequence_start_mask,'shape',None)}.")
        sequence_start_mask = sequence_start_mask.astype(bool, copy=False)

    # Validate ranges
    ranges: List[Tuple[float, float]] = []
    for (lo, hi) in channel_freq_ranges:
        lo = float(lo)
        hi = float(hi)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            raise ValueError(f"Invalid channel range (low, high)=({lo},{hi}). Must be finite with high>low.")
        if hi <= 0:
            raise ValueError(f"Invalid channel range (low, high)=({lo},{hi}). high must be > 0.")
        ranges.append((lo, hi))

    if emg_channel_index is None:
        emg_channel_index = C - 1
    if not (0 <= emg_channel_index < C):
        raise ValueError(f"emg_channel_index must be in [0, {C-1}]. Got {emg_channel_index}.")

    # % contribution per predicted label (row)
    pred_counts = np.bincount(y_pred, minlength=L).astype(np.float64)
    pred_pct = 100.0 * pred_counts / max(float(N), 1.0)

    # -----------------
    # Frequency axis (assume rFFT)
    # -----------------
    n_time = 2 * (F - 1) if F > 1 else 1
    freqs = np.fft.rfftfreq(n_time, d=1.0 / sample_rate)  # length F

    # -----------------
    # Remap input -> linear power
    # -----------------
    X = raw_datapoints.astype(np.float64, copy=False)
    scale = input_scale.lower()
    if scale in ("log", "ln"):
        P = np.exp(X)
        if log_eps != 0.0:
            P = P - log_eps
    elif scale == "log10":
        P = np.power(10.0, X)
        if log_eps != 0.0:
            P = P - log_eps
    elif scale == "linear":
        P = X
    else:
        raise ValueError('input_scale must be one of: "log"/"ln", "log10", "linear".')

    P = np.maximum(P, 0.0)

    # -----------------
    # Helper: per-label trimming + shared limits from trimmed union
    # -----------------
    def _trim_vals(vals: np.ndarray, *, trim_percent: float, xscale: str) -> np.ndarray:
        vals = np.asarray(vals, dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if xscale == "log":
            vals = vals[vals > 0]

        if vals.size < 5 or trim_percent <= 0:
            return vals

        lo = float(np.percentile(vals, trim_percent))
        hi = float(np.percentile(vals, 100.0 - trim_percent))
        if hi <= lo:
            return vals
        return vals[(vals >= lo) & (vals <= hi)]

    def _compute_xlim_from_union(trimmed_lists: List[np.ndarray], *, xscale: str, pad_frac: float) -> Tuple[float, float]:
        union = np.concatenate([v for v in trimmed_lists if v.size > 0], axis=0) if trimmed_lists else np.array([], dtype=np.float64)
        if union.size == 0:
            return (0.0, 1.0)

        vmin = float(union.min())
        vmax = float(union.max())

        if np.isclose(vmin, vmax):
            if xscale == "log":
                return (max(vmin * 0.8, 1e-12), vmax * 1.2)
            return (vmin - 1.0, vmax + 1.0)

        if xscale == "log":
            lo = max(vmin / (1.0 + pad_frac), 1e-12)
            hi = vmax * (1.0 + pad_frac)
            return (lo, hi)

        pad = (vmax - vmin) * pad_frac
        return (vmin - pad, vmax + pad)

    # -----------------
    # Bout lengths (run-length encoding on y_pred)
    # -----------------
    def _bout_lengths_by_label(
        y: np.ndarray,
        n_labels: int,
        seq_start_mask: Optional[np.ndarray],
    ) -> List[np.ndarray]:
        y = np.asarray(y, dtype=int)
        if y.size == 0:
            return [np.array([], dtype=np.float64) for _ in range(n_labels)]

        reset = np.zeros_like(y, dtype=bool) if seq_start_mask is None else np.asarray(seq_start_mask, dtype=bool)
        reset = reset.copy()
        reset[0] = True

        breaks = np.where((np.diff(y) != 0) | reset[1:])[0] + 1
        starts = np.r_[0, breaks]
        ends = np.r_[breaks, y.size]

        run_labels = y[starts]
        run_lengths = (ends - starts).astype(np.float64)

        out: List[np.ndarray] = []
        for lbl in range(n_labels):
            out.append(run_lengths[run_labels == lbl])
        return out

    bout_lengths_per_label = _bout_lengths_by_label(y_pred, L, sequence_start_mask)
    if bout_length_epoch_sec != 1.0:
        bout_lengths_per_label = [
            arr * bout_length_epoch_sec for arr in bout_lengths_per_label
        ]

    # -----------------
    # Feature masks + per-sample features
    # -----------------
    delta_mask = (freqs >= 0.5) & (freqs < 4.0)
    theta_mask = (freqs >= 4.0) & (freqs < 8.0)

    emg_lo, emg_hi = ranges[emg_channel_index]
    emg_mask = (freqs >= emg_lo) & (freqs < emg_hi)

    delta_pow_ch0 = P[:, 0, :][:, delta_mask].sum(axis=1)
    theta_pow_ch0 = P[:, 0, :][:, theta_mask].sum(axis=1)
    ratio_ch0 = theta_pow_ch0 / np.maximum(delta_pow_ch0, ratio_eps)

    delta_pow_ch1 = P[:, 1, :][:, delta_mask].sum(axis=1)
    theta_pow_ch1 = P[:, 1, :][:, theta_mask].sum(axis=1)
    ratio_ch1 = theta_pow_ch1 / np.maximum(delta_pow_ch1, ratio_eps)

    emg_power = P[:, emg_channel_index, :][:, emg_mask].sum(axis=1)

    feature_scales: List[str] = ["linear", "linear", emg_power_scale, bout_length_scale]
    feature_all: List[np.ndarray] = [ratio_ch0, ratio_ch1, emg_power]

    feature_vals_by_label: List[List[np.ndarray]] = [[np.array([], dtype=np.float64) for _ in range(4)] for _ in range(L)]
    for lbl in range(L):
        idx = np.where(y_pred == lbl)[0]
        if idx.size == 0:
            feature_vals_by_label[lbl][3] = bout_lengths_per_label[lbl]
            continue
        feature_vals_by_label[lbl][0] = feature_all[0][idx]
        feature_vals_by_label[lbl][1] = feature_all[1][idx]
        feature_vals_by_label[lbl][2] = feature_all[2][idx]
        feature_vals_by_label[lbl][3] = bout_lengths_per_label[lbl]

    trimmed_feature_vals_by_label: List[List[np.ndarray]] = [[np.array([], dtype=np.float64) for _ in range(4)] for _ in range(L)]
    for lbl in range(L):
        for k in range(4):
            trimmed_feature_vals_by_label[lbl][k] = _trim_vals(
                feature_vals_by_label[lbl][k],
                trim_percent=violin_trim_percent,
                xscale=feature_scales[k],
            )

    feature_xlims: List[Tuple[float, float]] = []
    for k in range(4):
        per_label_trimmed = [trimmed_feature_vals_by_label[lbl][k] for lbl in range(L)]
        feature_xlims.append(_compute_xlim_from_union(per_label_trimmed, xscale=feature_scales[k], pad_frac=feature_xlim_pad_frac))

    # -----------------
    # Per-channel integer Hz bins and masks (for spectra bars)
    # -----------------
    channel_bins: List[np.ndarray] = []
    channel_bin_masks: List[List[np.ndarray]] = []

    for c_idx, (lo, hi) in enumerate(ranges):
        k_start = int(np.ceil(lo))
        k_stop = int(np.floor(hi))  # bins: k_start .. k_stop-1
        bins = np.arange(k_start, k_stop, dtype=int) if (k_stop > k_start) else np.array([], dtype=int)
        channel_bins.append(bins)
        channel_bin_masks.append([(freqs >= k) & (freqs < (k + 1)) for k in bins])

    # -----------------
    # Aggregate per predicted label+channel -> binned bars
    # -----------------
    bars: List[List[np.ndarray]] = [[np.zeros(len(channel_bins[c]), dtype=np.float64) for c in range(C)] for _ in range(L)]

    for lbl in range(L):
        idx = np.where(y_pred == lbl)[0]
        if idx.size == 0:
            continue

        P_lbl = P[idx]  # (n_lbl, C, F)
        spec = P_lbl.mean(axis=0) if agg == "mean" else np.median(P_lbl, axis=0)  # (C, F)

        for c_idx in range(C):
            masks_for_c = channel_bin_masks[c_idx]
            ybars = np.zeros(len(masks_for_c), dtype=np.float64)
            for b, mask in enumerate(masks_for_c):
                ybars[b] = spec[c_idx, mask].mean() if np.any(mask) else 0.0

            if normalize_per_row:
                s = ybars.sum()
                if s > 0:
                    ybars = ybars / s

            bars[lbl][c_idx] = ybars

    # -----------------
    # True-label distribution per predicted label
    # -----------------
    true_dist = np.zeros((L, T), dtype=np.float64)
    for lbl in range(L):
        idx = np.where(y_pred == lbl)[0]
        if idx.size == 0:
            continue
        counts = np.bincount(labels_true[idx], minlength=T).astype(np.float64)
        s = counts.sum()
        if s > 0:
            true_dist[lbl] = counts / s

    # -----------------
    # Subject distribution per predicted label (count vs inverse-total weighting)
    # -----------------
    # Map arbitrary subject IDs -> contiguous indices
    sub_ids_obj = np.asarray(sub_ids)
    if sub_ids_obj.dtype.kind in ("U", "S"):
        sub_ids_obj = sub_ids_obj.astype(object)

    unique_subs, sub_inv = np.unique(sub_ids_obj, return_inverse=True)
    S = int(unique_subs.size)

    # Global totals per subject (across ALL data)
    subj_totals = np.bincount(sub_inv, minlength=S).astype(np.float64)
    subj_totals = np.maximum(subj_totals, 1.0)  # safety

    # counts per (label, subject) (still useful for ordering + "count" mode)
    counts_rs = np.zeros((L, S), dtype=np.float64)
    for lbl in range(L):
        idx = np.where(y_pred == lbl)[0]
        if idx.size == 0:
            continue
        counts_rs[lbl] = np.bincount(sub_inv[idx], minlength=S).astype(np.float64)

    if subject_dist_mode == "count":
        subject_dist = np.zeros((L, S), dtype=np.float64)
        row_sums = counts_rs.sum(axis=1, keepdims=True)
        nz = row_sums[:, 0] > 0
        subject_dist[nz] = counts_rs[nz] / row_sums[nz]
    else:
        # inv_total: weight each sample by 1 / total(subject)
        # => dist[r,s] ∝ counts_rs[r,s] / subj_totals[s]
        weighted = counts_rs / subj_totals[None, :]
        subject_dist = np.zeros((L, S), dtype=np.float64)
        row_sums = weighted.sum(axis=1, keepdims=True)
        nz = row_sums[:, 0] > 0
        subject_dist[nz] = weighted[nz] / row_sums[nz]

    # Legend ordering (usually by global count)
    if subject_sort_legend:
        sub_order = np.argsort(-subj_totals)  # descending totals
    else:
        sub_order = np.arange(S)

    # Subject colors
    cmap = plt.get_cmap("tab20" if S <= 20 else "hsv")
    subject_colors: List[Tuple[float, float, float, float]] = [cmap(i / max(S - 1, 1)) for i in range(S)]

    # Lab distribution (optional)
    has_lab_col = lab_ids is not None
    lab_legend_handles = None
    lab_legend_labels = None
    if has_lab_col:
        lab_ids_obj = np.asarray(lab_ids, dtype=object)
        unique_labs, lab_inv = np.unique(lab_ids_obj, return_inverse=True)
        n_labs = int(unique_labs.size)
        lab_totals = np.bincount(lab_inv, minlength=n_labs).astype(np.float64)
        lab_totals = np.maximum(lab_totals, 1.0)
        counts_rl = np.zeros((L, n_labs), dtype=np.float64)
        for lbl in range(L):
            idx = np.where(y_pred == lbl)[0]
            if idx.size:
                counts_rl[lbl] = np.bincount(lab_inv[idx], minlength=n_labs).astype(np.float64)
        if subject_dist_mode == "count":
            lab_dist = np.zeros((L, n_labs), dtype=np.float64)
            row_sums = counts_rl.sum(axis=1, keepdims=True)
            nz = row_sums[:, 0] > 0
            lab_dist[nz] = counts_rl[nz] / row_sums[nz]
        else:
            weighted = counts_rl / lab_totals[None, :]
            lab_dist = np.zeros((L, n_labs), dtype=np.float64)
            row_sums = weighted.sum(axis=1, keepdims=True)
            nz = row_sums[:, 0] > 0
            lab_dist[nz] = weighted[nz] / row_sums[nz]
        default_lab_colors = {
            "lab_1": "#4E79A7", "lab_2": "#F28E2B", "lab_3": "#59A14F",
            "lab_4": "#E15759", "lab_5": "#B07AA1", "unknown": "#BAB0AC",
        }
        palette = {**default_lab_colors, **(lab_colors or {})}
        lab_color_list = [palette.get(str(lab), "#999999") for lab in unique_labs]
        lab_order = np.argsort(-lab_totals)

    # -----------------
    # Compute shared y-limits per spectrum column (channel)
    # -----------------
    col_ylims: List[Tuple[float, float]] = []
    for c_idx in range(C):
        vals = np.concatenate([bars[lbl][c_idx] for lbl in range(L)]) if L > 0 else np.array([], dtype=np.float64)

        if vals.size == 0:
            col_ylims.append((0.0, 1.0))
            continue

        if y_scale == "log":
            pos = vals[vals > 0]
            if pos.size == 0:
                col_ylims.append((0.0, 1.0))
                continue

            if y_lim_mode == "minmax":
                vmin, vmax = float(pos.min()), float(pos.max())
            else:
                lo_p, hi_p = y_clip_percentiles
                vmin = float(np.percentile(pos, lo_p))
                vmax = float(np.percentile(pos, hi_p))
                vmin = max(vmin, float(pos.min()))
                vmax = min(max(vmax, vmin * 1.01), float(pos.max()))

            lo_exp = int(np.floor(np.log10(vmin)))
            hi_exp = int(np.ceil(np.log10(vmax)))
            col_ylims.append((10.0 ** lo_exp, 10.0 ** hi_exp))
        else:
            if y_lim_mode == "minmax":
                vmin, vmax = float(vals.min()), float(vals.max())
            else:
                lo_p, hi_p = y_clip_percentiles
                vmin = float(np.percentile(vals, lo_p))
                vmax = float(np.percentile(vals, hi_p))
                vmin = min(vmin, float(vals.min()))
                vmax = max(vmax, float(vals.max()))
            if np.isclose(vmax, vmin):
                vmax = vmin + 1.0
            col_ylims.append((vmin, vmax))

    # -----------------
    # EEG band overlay helper
    # -----------------
    def _add_eeg_bands(ax, lo: float, hi: float) -> None:
        tx = transforms.blended_transform_factory(ax.transData, ax.transAxes)

        segments = []
        boundaries = set()
        for name, b_lo, b_hi in eeg_bands:
            seg_lo = max(lo, b_lo)
            seg_hi = min(hi, b_hi)
            if seg_hi > seg_lo:
                segments.append((name, seg_lo, seg_hi))
                boundaries.add(seg_lo)
                boundaries.add(seg_hi)

        for x0 in sorted(boundaries):
            if lo <= x0 <= hi:
                ax.axvline(x0, linewidth=eeg_line_lw, alpha=eeg_line_alpha, zorder=3)

        for name, seg_lo, seg_hi in segments:
            xmid = 0.5 * (seg_lo + seg_hi)
            ax.text(
                xmid,
                eeg_label_y,
                name,
                transform=tx,
                ha="center",
                va="top",
                fontsize=eeg_label_fontsize,
                alpha=eeg_label_alpha,
                zorder=4,
            )

    # -----------------
    # Helper: horizontal violin in one axis (expects already-trimmed vals)
    # -----------------
    def _plot_single_violin_h(ax, vals: np.ndarray, color: ColorLike, xscale: str, xlim: Tuple[float, float]) -> None:
        vals = np.asarray(vals, dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if xscale == "log":
            vals = vals[vals > 0]

        if vals.size == 0:
            ax.text(0.5, 0.5, "n=0", ha="center", va="center", transform=ax.transAxes, alpha=0.7)
        else:
            vp = ax.violinplot(
                dataset=[vals],
                positions=[0.0],
                widths=0.85,
                showmeans=False,
                showmedians=True,
                showextrema=False,
                vert=False,
            )
            for body in vp["bodies"]:
                body.set_facecolor(color)
                body.set_edgecolor("none")
                body.set_alpha(0.75)
            if "cmedians" in vp:
                vp["cmedians"].set_color("black")
                vp["cmedians"].set_linewidth(1.0)
                vp["cmedians"].set_alpha(0.7)

        ax.set_ylim(-0.9, 0.9)
        ax.set_yticks([])
        if xscale == "log":
            ax.set_xscale("log")
        ax.set_xlim(*xlim)
        ax.grid(True, axis="x", linewidth=0.4, alpha=0.35)

    # -----------------
    # Plot grid: L rows x (C + 4 + 1 + 1) cols
    # -----------------
    n_feature_cols = 4
    true_dist_col_idx = C + n_feature_cols
    subj_dist_col_idx = C + n_feature_cols + 1
    lab_dist_col_idx = C + n_feature_cols + 2 if has_lab_col else None
    C_plot = C + n_feature_cols + 2 + (1 if has_lab_col else 0)

    fig_w = 5.2 * C_plot
    fig_h = max(2.8, figsize_per_row * L)
    fig, axes = plt.subplots(L, C_plot, figsize=(fig_w, fig_h), sharey=False)

    if L == 1 and C_plot == 1:
        axes = np.array([[axes]])
    elif L == 1:
        axes = np.array([axes])
    elif C_plot == 1:
        axes = axes[:, None]

    true_legend_handles = None
    true_legend_labels = None

    subj_legend_handles = None
    subj_legend_labels = None

    for r in range(L):
        # ---- Spectrum columns ----
        for c_idx in range(C):
            ax = axes[r, c_idx]
            x = channel_bins[c_idx]
            yv = bars[r][c_idx]

            ax.bar(
                x, yv,
                width=0.95,
                align="center",
                color=label_colors[r],
                alpha=row_alpha,
                edgecolor="none",
            )

            lo, hi = ranges[c_idx]
            ax.set_xlim(lo - 0.5, hi + 0.5)

            if y_scale == "log":
                ax.set_yscale("log")
            ax.set_ylim(*col_ylims[c_idx])

            ax.grid(True, linewidth=0.4, alpha=0.35)

            if r == 0:
                ax.set_title(str(channel_names[c_idx]))

            if c_idx == 0:
                ax.set_ylabel(f"{label_names[r]}\n{pred_pct[r]:.1f}% ({int(pred_counts[r])})")

            if r == L - 1:
                ax.set_xlabel("Frequency (Hz)")
            else:
                ax.tick_params(labelbottom=False)

            tick_start = int(np.ceil(lo / 5.0) * 5)
            tick_end = int(np.floor(hi / 5.0) * 5)
            if tick_end >= tick_start:
                ax.set_xticks(np.arange(tick_start, tick_end + 1, 5))

            if c_idx in eeg_band_cols:
                _add_eeg_bands(ax, lo=lo, hi=hi)

        # ---- Feature columns ----
        for k in range(n_feature_cols):
            axf = axes[r, C + k]
            vals = trimmed_feature_vals_by_label[r][k]

            _plot_single_violin_h(
                ax=axf,
                vals=vals,
                color=label_colors[r],
                xscale=feature_scales[k],
                xlim=feature_xlims[k],
            )

            if r == 0:
                axf.set_title(feature_titles[k])

            if r != L - 1:
                axf.tick_params(labelbottom=False)

        # ---- True label distribution ----
        axd = axes[r, true_dist_col_idx]
        dist = true_dist[r]

        left = 0.0
        for t_idx in range(T):
            w = float(dist[t_idx])
            if w <= 0:
                continue
            axd.barh(
                y=0,
                width=w,
                left=left,
                height=0.72,
                color=label_colors_true[t_idx],
                alpha=true_dist_alpha,
                edgecolor=true_dist_edgecolor,
                label=str(label_names_true[t_idx]),
            )
            left += w

        axd.set_xlim(0.0, 1.0)
        axd.set_ylim(-1.0, 1.0)
        axd.grid(True, axis="x", linewidth=0.4, alpha=0.35)

        if r == 0:
            axd.set_title(true_dist_col_title)

        axd.set_yticks([])
        axd.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        axd.set_xticklabels(["0%", "25%", "50%", "75%", "100%"] if r == L - 1 else [])
        if r == L - 1:
            axd.set_xlabel("Percent")

        if r == 0:
            true_legend_handles = [
                plt.Rectangle((0, 0), 1, 1, color=label_colors_true[t], alpha=true_dist_alpha)
                for t in range(T)
            ]
            true_legend_labels = [str(n) for n in label_names_true]

        # ---- Subject distribution (inv_total-weighted) ----
        axs = axes[r, subj_dist_col_idx]
        sdist = subject_dist[r]

        left = 0.0
        for s_idx in sub_order:
            w = float(sdist[s_idx])
            if w <= 0:
                continue
            axs.barh(
                y=0,
                width=w,
                left=left,
                height=0.72,
                color=subject_colors[s_idx],
                alpha=subject_dist_alpha,
                edgecolor=subject_dist_edgecolor,
                label=str(unique_subs[s_idx]),
            )
            left += w

        axs.set_xlim(0.0, 1.0)
        axs.set_ylim(-1.0, 1.0)
        axs.grid(True, axis="x", linewidth=0.4, alpha=0.35)

        if r == 0:
            subtitle = "inv-total weighted" if subject_dist_mode == "inv_total" else "count-weighted"
            axs.set_title(f"{subject_dist_col_title}\n({subtitle})")

        axs.set_yticks([])
        axs.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        axs.set_xticklabels(["0%", "25%", "50%", "75%", "100%"] if r == L - 1 else [])
        if r == L - 1:
            axs.set_xlabel("Percent")

        if r == 0:
            ordered_labels = [str(unique_subs[i]) for i in sub_order]
            ordered_handles = [
                plt.Rectangle((0, 0), 1, 1, color=subject_colors[i], alpha=subject_dist_alpha)
                for i in sub_order
            ]

            if subject_max_legend_items is not None and S > int(subject_max_legend_items):
                k = int(subject_max_legend_items)
                ordered_labels = ordered_labels[:k]
                ordered_handles = ordered_handles[:k]
                ordered_labels.append(f"+{S - k} more")
                ordered_handles.append(plt.Rectangle((0, 0), 1, 1, color=(0, 0, 0, 0), alpha=0.0))

            subj_legend_handles = ordered_handles
            subj_legend_labels = ordered_labels

        # ---- Lab distribution ----
        if has_lab_col and lab_dist_col_idx is not None:
            axl = axes[r, lab_dist_col_idx]
            ldist = lab_dist[r]
            left = 0.0
            for li in lab_order:
                w = float(ldist[li])
                if w <= 0:
                    continue
                axl.barh(
                    y=0,
                    width=w,
                    left=left,
                    height=0.72,
                    color=lab_color_list[li],
                    alpha=lab_dist_alpha,
                    edgecolor=lab_dist_edgecolor,
                    label=str(unique_labs[li]),
                )
                left += w
            axl.set_xlim(0.0, 1.0)
            axl.set_ylim(-1.0, 1.0)
            axl.grid(True, axis="x", linewidth=0.4, alpha=0.35)
            if r == 0:
                axl.set_title(lab_dist_col_title)
            axl.set_yticks([])
            axl.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
            axl.set_xticklabels(["0%", "25%", "50%", "75%", "100%"] if r == L - 1 else [])
            if r == L - 1:
                axl.set_xlabel("Percent")
            if r == 0:
                lab_legend_handles = [
                    plt.Rectangle((0, 0), 1, 1, color=lab_color_list[i], alpha=lab_dist_alpha)
                    for i in lab_order
                ]
                lab_legend_labels = [str(unique_labs[i]) for i in lab_order]

    fig.suptitle(plot_title, y=1.02, fontsize=14)

    right_margin = 0.965 if not has_lab_col else 0.94
    fig.tight_layout(rect=[0.0, 0.0, right_margin, 0.98])

    if show_true_dist_legend and true_legend_handles is not None:
        fig.legend(
            true_legend_handles,
            true_legend_labels,
            loc="upper left",
            bbox_to_anchor=(0.968, 0.98),
            frameon=False,
            title="True labels",
        )

    if show_subject_dist_legend and subj_legend_handles is not None:
        fig.legend(
            subj_legend_handles,
            subj_legend_labels,
            loc="lower left",
            bbox_to_anchor=(0.968 if not has_lab_col else 0.945, 0.02),
            frameon=False,
            title="Subjects",
            fontsize=7,
        )

    if show_lab_dist_legend and lab_legend_handles is not None:
        fig.legend(
            lab_legend_handles,
            lab_legend_labels,
            loc="center left",
            bbox_to_anchor=(0.968, 0.50),
            frameon=False,
            title="Labs",
            fontsize=8,
        )

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
