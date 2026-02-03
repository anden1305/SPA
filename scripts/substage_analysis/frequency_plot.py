from __future__ import annotations

from typing import Sequence, Optional, Tuple, List
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_label_channel_frequency_grid(
    raw_datapoints: np.ndarray,                 # (N, C, F) log(power) by default
    sample_rate: float,
    y_pred: np.ndarray,                         # (N,)
    label_names: Sequence[str],                 # length = n_labels
    channel_names: Sequence[str],               # length = C
    plot_title: str,
    *,
    channel_freq_ranges: Sequence[Tuple[float, float]],  # length C, e.g. [(0,20),(0,20),(0,100)]
    agg: str = "mean",                          # "mean" or "median"
    input_scale: str = "log",                   # "log"/"ln", "log10", "linear"
    log_eps: float = 0.0,                       # if you did log(power + eps), set eps here
    normalize_per_row: bool = False,            # if True, disables meaningful cross-row amplitude compare
    # ---- new: shared y-axis per column ----
    y_scale: str = "log",                       # "log" or "linear"
    y_lim_mode: str = "nice_pow10",             # "nice_pow10" or "minmax"
    y_clip_percentiles: Tuple[float, float] = (1.0, 99.0),  # robust limits for "nice_pow10"
    # ---------------------------------------
    figsize_per_row: float = 2.6,
    save_path: Optional[str] = None,
) -> None:
    """
    Plots a grid of 1 Hz binned spectra per label (rows) and channel (cols), restricted to per-channel frequency bands.

    Defaults assume:
      - raw_datapoints contains log(power) values (log applied AFTER FFT power).
      - We remap back to linear power BEFORE aggregating/binning/plotting.

    NEW requirement:
      - All subplots in the same column share the same y-axis scaling and limits,
        so you can compare labels across rows for that channel.

    Note: If normalize_per_row=True, amplitudes become relative per (label, channel),
          which can undermine cross-row magnitude comparisons. Leave False if you want amplitude comparison.
    """
    # -----------------
    # Validate inputs
    # -----------------
    if not isinstance(raw_datapoints, np.ndarray) or raw_datapoints.ndim != 3:
        raise ValueError(f"raw_datapoints must be a numpy array of shape (N,C,F). Got {getattr(raw_datapoints,'shape',None)}.")
    if not isinstance(y_pred, np.ndarray) or y_pred.ndim != 1:
        raise ValueError(f"y_pred must be a numpy array of shape (N,). Got {getattr(y_pred,'shape',None)}.")

    N, C, F = raw_datapoints.shape
    if y_pred.shape[0] != N:
        raise ValueError(f"y_pred length must equal N={N}. Got {y_pred.shape[0]}.")
    if len(channel_names) != C:
        raise ValueError(f"channel_names length must equal C={C}. Got {len(channel_names)}.")
    if len(label_names) == 0:
        raise ValueError("label_names must be non-empty.")
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

    # Ensure integer labels
    if not np.issubdtype(y_pred.dtype, np.integer):
        if np.all(np.isfinite(y_pred)) and np.all(np.equal(y_pred, np.round(y_pred))):
            y_pred = y_pred.astype(int)
        else:
            raise ValueError("y_pred must be integer dtype (or safely castable to integers).")

    L = len(label_names)
    if y_pred.min() < 0 or y_pred.max() >= L:
        raise ValueError(f"y_pred must be in [0, {L-1}]. Got min={y_pred.min()}, max={y_pred.max()}.")

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

    # guard against tiny negatives from eps subtraction
    P = np.maximum(P, 0.0)

    # -----------------
    # Per-channel integer Hz bins and masks
    # -----------------
    channel_bins: List[np.ndarray] = []
    channel_bin_masks: List[List[np.ndarray]] = []

    for c_idx, (lo, hi) in enumerate(ranges):
        k_start = int(np.ceil(lo))
        k_stop = int(np.floor(hi))  # bins: k_start .. k_stop-1, representing [k,k+1)
        bins = np.arange(k_start, k_stop, dtype=int) if (k_stop > k_start) else np.array([], dtype=int)

        channel_bins.append(bins)
        masks_for_c: List[np.ndarray] = [(freqs >= k) & (freqs < (k + 1)) for k in bins]
        channel_bin_masks.append(masks_for_c)

    # -----------------
    # Aggregate per label+channel -> binned bars
    # -----------------
    bars: List[List[np.ndarray]] = [[np.zeros(len(channel_bins[c]), dtype=np.float64) for c in range(C)] for _ in range(L)]

    for lbl in range(L):
        idx = np.where(y_pred == lbl)[0]
        if idx.size == 0:
            continue

        P_lbl = P[idx]  # (n_lbl, C, F)

        # aggregate across samples in linear power domain
        spec = P_lbl.mean(axis=0) if agg == "mean" else np.median(P_lbl, axis=0)  # (C, F)

        for c_idx in range(C):
            bins = channel_bins[c_idx]
            masks_for_c = channel_bin_masks[c_idx]
            ybars = np.zeros(len(bins), dtype=np.float64)

            for b, mask in enumerate(masks_for_c):
                ybars[b] = spec[c_idx, mask].mean() if np.any(mask) else 0.0

            if normalize_per_row:
                s = ybars.sum()
                if s > 0:
                    ybars = ybars / s

            bars[lbl][c_idx] = ybars

    # -----------------
    # Compute shared y-limits per column (channel)
    # -----------------
    col_ylims: List[Tuple[float, float]] = []
    for c_idx in range(C):
        # collect all values for this channel across labels
        vals = np.concatenate([bars[lbl][c_idx] for lbl in range(L)]) if L > 0 else np.array([], dtype=np.float64)

        if vals.size == 0:
            col_ylims.append((0.0, 1.0))
            continue

        if y_scale == "log":
            # ignore zeros for log-limits
            pos = vals[vals > 0]
            if pos.size == 0:
                # nothing positive; fall back to linear-ish
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

            # expand to “nice” powers of 10: 1e-6 .. 1e-3 etc
            lo_exp = int(np.floor(np.log10(vmin)))
            hi_exp = int(np.ceil(np.log10(vmax)))
            col_ylims.append((10.0 ** lo_exp, 10.0 ** hi_exp))
        else:
            # linear
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
    # Plot grid: L rows x C cols (per-channel x-limits differ, y-limits shared per column)
    # -----------------
    fig_w = 5.2 * C
    fig_h = max(2.8, figsize_per_row * L)
    fig, axes = plt.subplots(L, C, figsize=(fig_w, fig_h), sharey=False)

    # Normalize axes indexing
    if L == 1 and C == 1:
        axes = np.array([[axes]])
    elif L == 1:
        axes = np.array([axes])
    elif C == 1:
        axes = axes[:, None]

    for r in range(L):
        for c_idx in range(C):
            ax = axes[r, c_idx]
            x = channel_bins[c_idx]
            yv = bars[r][c_idx]

            ax.bar(x, yv, width=0.95, align="center")

            # x-range per channel band
            lo, hi = ranges[c_idx]
            ax.set_xlim(lo - 0.5, hi + 0.5)

            # shared y-scale + y-lims per column
            if y_scale == "log":
                ax.set_yscale("log")
            ax.set_ylim(*col_ylims[c_idx])

            ax.grid(True, linewidth=0.4, alpha=0.35)

            if r == 0:
                ax.set_title(str(channel_names[c_idx]))
            if c_idx == 0:
                ax.set_ylabel(str(label_names[r]))

            if r == L - 1:
                ax.set_xlabel("Frequency (Hz)")
            else:
                ax.tick_params(labelbottom=False)

            # ticks every 5 Hz within the band
            tick_start = int(np.ceil(lo / 5.0) * 5)
            tick_end = int(np.floor(hi / 5.0) * 5)
            if tick_end >= tick_start:
                ax.set_xticks(np.arange(tick_start, tick_end + 1, 5))

    fig.suptitle(plot_title, y=1.02, fontsize=14)
    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
