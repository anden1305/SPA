import numpy as np
import pandas as pd

def _load_npy_1d(path, dtype=None):
    """Load a .npy file and return a 1D numpy array (raveled if needed)."""
    arr = np.load(path, allow_pickle=False)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return np.ravel(arr)

def split_eeg_by_label(eeg_path, labels_path, as_lists=True):
    """
    Load EEG and label arrays from .npy files, split into contiguous segments
    per label, and return:
      - segments_by_label: {label: [segment1, segment2, ...]}
      - indices_by_label:  {label: [(start_idx, end_idx), ...]} with end exclusive

    Parameters
    ----------
    eeg_path : str
        Path to the .npy file containing the EEG signal (1D floats).
    labels_path : str
        Path to the .npy file containing the labels (1D ints).
    as_lists : bool, default True
        If True, each EEG segment is returned as a Python list; if False, as a numpy array.

    Returns
    -------
    segments_by_label : dict[int, list[list[float] or np.ndarray]]
        Dictionary mapping each label to a list of EEG segments for each contiguous run.
    indices_by_label : dict[int, list[tuple[int, int]]]
        Dictionary mapping each label to a list of (start, end) index tuples (end is exclusive).

    Raises
    ------
    ValueError
        If the EEG and labels differ in length.
    """
    eeg = _load_npy_1d(eeg_path, dtype=float)
    labels = _load_npy_1d(labels_path, dtype=int)

    if eeg.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Length mismatch: EEG has {eeg.shape[0]} samples but labels has {labels.shape[0]}."
        )

    n = labels.shape[0]
    if n == 0:
        return {}, {}

    # Find boundaries where the label changes
    change_points = np.flatnonzero(np.diff(labels) != 0) + 1
    starts = np.r_[0, change_points]
    ends   = np.r_[change_points, n]

    unique_labels = np.unique(labels).astype(int)
    segments_by_label = {int(lbl): [] for lbl in unique_labels}
    indices_by_label  = {int(lbl): [] for lbl in unique_labels}

    for s, e in zip(starts, ends):
        lbl = int(labels[s])
        segment = eeg[s:e]
        segments_by_label[lbl].append(segment.tolist() if as_lists else segment)
        indices_by_label[lbl].append((int(s), int(e)))  # end is exclusive

    return segments_by_label, indices_by_label

from typing import Iterable, Tuple, Dict, List, Any

def combine_splits_from_path_tuples(
    path_tuples: Iterable[Tuple[str, str]],
    as_lists: bool = True,
):
    """
    Call `split_eeg_by_label` for each (eeg_path, labels_path) pair and
    combine the end results by label.

    Parameters
    ----------
    path_tuples : iterable of (eeg_path, labels_path)
        Each is a pair of .npy file paths.
    as_lists : bool
        Passed through to `split_eeg_by_label`.

    Returns
    -------
    segments_by_label : dict[int, list[list[float] or np.ndarray]]
        Concatenated segments per label across all path tuples.
    indices_by_label : dict[int, list[tuple[int,int]]]
        Concatenated (start, end) index pairs per label (end exclusive).
        NOTE: indices are **relative to their source file**, since we are
        “just combining the end results”.
    """
    combined_segments: Dict[int, List[Any]] = {}
    combined_indices:  Dict[int, List[Tuple[int, int]]] = {}

    for eeg_path, labels_path in path_tuples:
        print(eeg_path)
        segs, idxs = split_eeg_by_label(eeg_path, labels_path, as_lists=as_lists)
        for lbl, seg_list in segs.items():
            combined_segments.setdefault(int(lbl), []).extend(seg_list)
        for lbl, idx_list in idxs.items():
            combined_indices.setdefault(int(lbl), []).extend(idx_list)
    
    return combined_segments, combined_indices


import numpy as np
from typing import Iterable, Tuple

def _next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()

def average_psd(
    signals: Iterable[np.ndarray],
    fs: float,
    n_fft: int | None = None,
    use_hann: bool = True,
    demean: bool = True,
    band: tuple[float, float] | None = (1.0, 60.0),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the average one-sided PSD across signals and (optionally) return only a band.

    Parameters
    ----------
    signals : iterable of 1D arrays
        Time-domain EEG segments.
    fs : float
        Sampling rate in Hz.
    n_fft : int, optional
        FFT length. If None, uses next power of 2 of the longest signal.
        Tip: to get ≤1 Hz bins, set n_fft >= fs.
    use_hann : bool
        Apply Hann window before FFT.
    demean : bool
        Subtract mean from each signal before windowing.
    band : (fmin, fmax) or None
        If provided, returns only frequencies within [fmin, fmax] Hz.

    Returns
    -------
    f : np.ndarray
        Frequency vector (Hz).
    psd_avg : np.ndarray
        Average PSD (power/Hz) aligned to f.
    """
    # Collect and sanitize signals
    sigs, max_len = [], 0
    for s in signals:
        x = np.ravel(np.asarray(s, dtype=float))
        if x.size == 0:
            continue
        if demean:
            x = x - np.nanmean(x)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        sigs.append(x)
        max_len = max(max_len, x.size)
    if not sigs:
        raise ValueError("No non-empty signals provided.")

    if n_fft is None:
        n_fft = _next_pow2(max_len)
    if n_fft < 2:
        raise ValueError("n_fft must be >= 2.")

    # Window
    w = np.hanning(n_fft) if use_hann else np.ones(n_fft)
    w2_sum = np.sum(w**2)

    # Frequencies (one-sided)
    f = np.fft.rfftfreq(n_fft, d=1.0/fs)
    nyq = fs / 2.0

    # PSDs per signal
    psds = []
    for x in sigs:
        # Truncate/pad
        if x.size >= n_fft:
            xw = x[:n_fft] * w
        else:
            xw = np.zeros(n_fft, dtype=float)
            xw[:x.size] = x * w[:x.size]

        X = np.fft.rfft(xw, n=n_fft)
        Pxx = (np.abs(X) ** 2) / (fs * w2_sum)

        # Double non-DC/non-Nyquist bins (one-sided real signal)
        if n_fft % 2 == 0:
            if Pxx.size > 2:
                Pxx[1:-1] *= 2.0
        else:
            if Pxx.size > 1:
                Pxx[1:] *= 2.0

        psds.append(Pxx)

    psd_avg = np.mean(np.stack(psds, axis=0), axis=0)

    # Band-limit to [1, 60] Hz (or custom band)
    if band is not None:
        fmin, fmax = float(band[0]), float(band[1])
        if fmin < 0:
            raise ValueError("fmin must be >= 0.")
        if fmax > nyq + 1e-9:
            raise ValueError(f"fmax ({fmax} Hz) exceeds Nyquist ({nyq:.3f} Hz). Increase fs or lower fmax.")
        mask = (f >= fmin) & (f <= fmax)
        f, psd_avg = f[mask], psd_avg[mask]

    return f, psd_avg


import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable, Dict, Any, Tuple, List, Optional

def _moving_average(y: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return y
    # 'same' length moving average (handles edges reasonably)
    kernel = np.ones(win, dtype=float) / win
    return np.convolve(y, kernel, mode="same")

def _normalize_curve(y: np.ndarray, f: np.ndarray, mode: str) -> np.ndarray:
    if mode is None or mode == "none":
        return y
    if mode == "area":
        area = np.trapz(y, f)
        return y / area if area > 0 else y
    if mode == "max":
        m = np.max(y)
        return y / m if m > 0 else y
    if mode == "zscore":
        mu, sd = np.mean(y), np.std(y)
        return (y - mu) / sd if sd > 0 else y
    raise ValueError(f"Unknown normalize mode: {mode}")

def plot_stage_psd_comparison(
    spectra: Iterable[Dict[str, Any]],
    title: Optional[str] = "Sleep Stage Spectra (1–60 Hz)",
    band: Tuple[float, float] = (1.0, 60.0),
    normalize: str = "area",   # 'area' (recommended), 'max', 'zscore', or 'none'
    smooth_hz: float = 0.0,    # e.g. 0.5–1.0 to tame variance; 0 disables
    logy: bool = True,
    annotate_bands: bool = True,   # draw canonical EEG band boundaries
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot multiple averaged PSDs for different sleep stages.

    Parameters
    ----------
    spectra : iterable of dicts
        Each dict must contain:
          - "name": label (e.g., 'N1', 'N2', 'N3', 'REM', 'Wake')
          - "f": 1D array of frequencies (Hz)
          - "psd_avg": 1D array of PSD values aligned to f (power/Hz)
        Arrays for each item need not share the same frequency grid.
    title : str
        Figure title.
    band : (fmin, fmax)
        Frequency window to display (and to use for normalization).
    normalize : str
        'area' normalizes each curve by its area within `band` (shape-only comparison),
        'max' scales to unit peak, 'zscore' standardizes, 'none' keeps raw units.
    smooth_hz : float
        Optional moving-average smoothing window (in Hz). Uses the median df to pick
        an integer window length per curve.
    logy : bool
        Log-scale y-axis (typical for PSDs).
    annotate_bands : bool
        If True, draws vertical lines at canonical EEG band boundaries.
    save_path : str
        If provided, saves the figure (e.g., 'stages_psd.png').
    show : bool
        If True, displays the figure.

    Returns
    -------
    fig, ax : Matplotlib figure and axes.
    """
    spectra = list(spectra)
    if len(spectra) == 0:
        raise ValueError("`spectra` is empty.")

    fmin, fmax = float(band[0]), float(band[1])
    if fmin >= fmax:
        raise ValueError("band must satisfy fmin < fmax.")

    fig, ax = plt.subplots()

    for item in spectra:
        if not all(k in item for k in ("name", "f", "psd_avg")):
            raise ValueError("Each item must have 'name', 'f', and 'psd_avg' keys.")

        name = str(item["name"])
        f = np.ravel(np.asarray(item["f"], dtype=float))
        psd = np.ravel(np.asarray(item["psd_avg"], dtype=float))
        if f.size != psd.size:
            raise ValueError(f"Size mismatch in '{name}': f({f.size}) != psd_avg({psd.size}).")

        # Ensure ascending frequency
        if not np.all(np.diff(f) >= 0):
            idx = np.argsort(f)
            f, psd = f[idx], psd[idx]

        # Restrict to display/normalization band
        mask = (f >= fmin) & (f <= fmax)
        f_band = f[mask]
        psd_band = psd[mask]
        if f_band.size == 0:
            continue  # nothing in band

        # Optional smoothing in Hz (use median df to choose window length)
        if smooth_hz and smooth_hz > 0:
            df = np.median(np.diff(f_band)) if f_band.size > 1 else np.inf
            win_pts = int(max(1, round(smooth_hz / df))) if np.isfinite(df) and df > 0 else 1
            psd_band = _moving_average(psd_band, win_pts)

        # Optional normalization (within the band)
        psd_band = _normalize_curve(psd_band, f_band, normalize)

        # Protect log-scale from zeros
        psd_plot = np.where(psd_band > 0, psd_band, np.finfo(float).tiny)

        # Plot
        if logy:
            ax.semilogy(f_band, psd_plot, label=name)
        else:
            ax.plot(f_band, psd_plot, label=name)

    # Cosmetics
    ax.set_xlim(fmin, fmax)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (power/Hz, log scale)" if logy and normalize in ("none", None) else
                  ("Normalized PSD (log scale)" if logy else "Normalized PSD"))
    if title:
        ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(title="Stage", loc="best")

    if annotate_bands:
        # Canonical boundaries (Hz) within 1–60: delta/theta/alpha/sigma/beta
        boundaries = [4, 8, 12, 16, 30]
        labels =   [("δ", 1, 4), ("θ", 4, 8), ("α", 8, 12), ("σ", 12, 16), ("β", 16, 30)]
        # Vertical lines
        for b in boundaries:
            if fmin < b < fmax:
                ax.axvline(b, linestyle=":", alpha=0.6)
        # Band labels (lightweight annotation at midpoints)
        ylim = ax.get_ylim()
        y_text = np.exp(np.mean(np.log(ylim))) if logy else (ylim[0] + 0.85*(ylim[1]-ylim[0]))
        for sym, a, b in labels:
            if b <= fmin or a >= fmax:
                continue
            xmid = max(fmin, min(fmax, (a + b) / 2.0))
            ax.text(xmid, y_text, sym, ha="center", va="center", alpha=0.7)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()

    return fig, ax


import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Iterable, Any, Optional, List

# Default EEG bands (Hz) commonly used in sleep analysis
DEFAULT_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 16.0),  # spindles
    "low-beta":  (16.0, 22.0),
    "high-beta":  (22.0, 30.0),
    "low-gamma": (30.0, 40.0),
    "lowmid-gamma": (40.0, 48.0),
    "highmid-gamma": (48.0, 52.0),
    "high-gamma": (52.0, 60.0),
    # Uncomment if you want higher bands:
    # "low-gamma": (30.0, 45.0),
    # "high-gamma": (45.0, 60.0),
}

def _band_integral_linear_psd(f: np.ndarray, psd: np.ndarray, a: float, b: float) -> float:
    """
    Integrate linear PSD over [a, b] Hz with trapezoidal rule.
    Handles frequency grids that don't align with band edges by interpolating endpoints.
    """
    f = np.ravel(np.asarray(f, dtype=float))
    psd = np.ravel(np.asarray(psd, dtype=float))
    if f.size == 0 or psd.size == 0 or f.size != psd.size:
        return 0.0
    if not np.all(np.diff(f) >= 0):
        idx = np.argsort(f)
        f, psd = f[idx], psd[idx]

    aa, bb = max(a, f[0]), min(b, f[-1])
    if aa >= bb:
        return 0.0

    mask = (f >= aa) & (f <= bb)
    fi = f[mask]
    pi = psd[mask]

    # Interpolate endpoints to ensure we integrate exactly over [aa, bb]
    p_aa = np.interp(aa, f, psd)
    p_bb = np.interp(bb, f, psd)

    if fi.size == 0:
        fi2 = np.array([aa, bb])
        pi2 = np.array([p_aa, p_bb])
    else:
        fi2 = fi
        pi2 = pi
        if fi2[0] > aa:
            fi2 = np.concatenate(([aa], fi2))
            pi2 = np.concatenate(([p_aa], pi2))
        elif fi2[0] < aa:
            fi2[0] = aa
            pi2[0] = p_aa
        if fi2[-1] < bb:
            fi2 = np.concatenate((fi2, [bb]))
            pi2 = np.concatenate((pi2, [p_bb]))
        elif fi2[-1] > bb:
            fi2[-1] = bb
            pi2[-1] = p_bb

    return float(np.trapz(pi2, fi2))

def _bandpowers_from_psd(
    f: np.ndarray,
    psd_avg: np.ndarray,
    bands: Dict[str, Tuple[float, float]],
    total_band: Tuple[float, float],
    relative: bool,
) -> Dict[str, float]:
    abs_powers = {name: _band_integral_linear_psd(f, psd_avg, lo, hi)
                  for name, (lo, hi) in bands.items()}
    if not relative:
        return abs_powers

    denom = _band_integral_linear_psd(f, psd_avg, total_band[0], total_band[1])
    if denom <= 0:
        return {k: np.nan for k in abs_powers}
    return {k: v / denom for k, v in abs_powers.items()}

def plot_stage_bandpowers(
    spectra: Iterable[Dict[str, Any]],
    bands: Dict[str, Tuple[float, float]] = DEFAULT_BANDS,
    total_band: Tuple[float, float] = (1.0, 60.0),
    relative: bool = True,
    title: Optional[str] = "Bandpower by Sleep Stage",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Compute and plot (relative or absolute) bandpowers per sleep stage from spectra.

    Parameters
    ----------
    spectra : iterable of dict
        Each dict: {"name": str, "f": 1D array (Hz), "psd_avg": 1D array (power/Hz)}.
    bands : dict
        Mapping band_name -> (low_Hz, high_Hz).
    total_band : (low, high)
        Band used as denominator for relative power (ignored if relative=False).
    relative : bool
        If True, plot proportion of total power in `total_band`. If False, plot absolute power.
    title : str
        Figure title.
    save_path : str
        If provided, saves the figure.
    show : bool
        If True, displays the figure.

    Returns
    -------
    fig, ax, values : (matplotlib.figure.Figure, matplotlib.axes.Axes, dict)
        `values` is {stage_name: {band_name: value, ...}, ...}
    """
    items = list(spectra)
    if len(items) == 0:
        raise ValueError("`spectra` is empty.")
    for it in items:
        if not all(k in it for k in ("name", "f", "psd_avg")):
            raise ValueError("Each item must contain 'name', 'f', and 'psd_avg'.")

    stage_names: List[str] = [str(it["name"]) for it in items]
    band_names: List[str] = list(bands.keys())

    # Compute bandpowers per stage
    values: Dict[str, Dict[str, float]] = {}
    M = np.zeros((len(items), len(band_names)), dtype=float)
    for i, it in enumerate(items):
        bp = _bandpowers_from_psd(
            it["f"], it["psd_avg"], bands=bands, total_band=total_band, relative=relative
        )
        values[stage_names[i]] = bp
        M[i, :] = [bp[b] for b in band_names]

    # Plot grouped bars: bands on x-axis, stage series offset within each band
    n_stages, n_bands = M.shape
    x = np.arange(n_bands)
    width = 0.8 / max(1, n_stages)

    fig, ax = plt.subplots()
    for i in range(n_stages):
        ax.bar(x + i * width, M[i], width=width, label=stage_names[i])

    ax.set_xticks(x + (n_stages - 1) * width / 2.0)
    ax.set_xticklabels(band_names)
    ax.set_xlabel("Band")
    ax.set_ylabel("Relative power" if relative else "Power")
    if title:
        ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(title="Stage", loc="best")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()

    return fig, ax, values


import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable, Dict, Any, Optional, Tuple, List

def plot_stage_mean_durations(
    items: Iterable[Dict[str, Any]],
    fs: float,
    ci_level: float = 0.95,
    ci_method: str = "bootstrap",  # 'bootstrap' (default) or 'normal'
    n_boot: int = 2000,
    min_duration_sec: float = 0.0, # filter out micro-segments if desired
    title: Optional[str] = "Average Segment Duration by Sleep Stage",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot average duration (in seconds) of contiguous segments per stage with a CI.

    Parameters
    ----------
    items : iterable of dict
        Each dict: {"name": str, "indices": list[(start, end), ...]}
        where (start, end) are sample indices with end exclusive.
    fs : float
        Sampling rate in Hz.
    ci_level : float
        Confidence level for intervals (e.g., 0.95).
    ci_method : {'bootstrap', 'normal'}
        - 'bootstrap': percentile bootstrap CI for the mean (no SciPy; robust)
        - 'normal'   : mean ± z * (sd / sqrt(n)) with z≈1.96
    n_boot : int
        Bootstrap resamples if ci_method='bootstrap'.
    min_duration_sec : float
        Drop segments shorter than this duration (in seconds).
    title : str
        Figure title.
    save_path : str
        If provided, saves the figure.
    show : bool
        If True, display the plot.

    Returns
    -------
    fig, ax, stats : (matplotlib.figure.Figure, matplotlib.axes.Axes, dict)
        stats = {stage_name: {'n_segments', 'mean_sec', 'ci_lower', 'ci_upper',
                              'std_sec', 'median_sec'}}
    """
    items = list(items)
    if len(items) == 0:
        raise ValueError("`items` is empty.")
    if fs <= 0:
        raise ValueError("`fs` must be > 0 (Hz).")

    stage_names: List[str] = []
    means: List[float] = []
    lowers: List[float] = []
    uppers: List[float] = []
    counts: List[int] = []
    stds: List[float] = []
    medians: List[float] = []

    alpha = 1.0 - ci_level
    z = 1.96  # for 'normal' method at ~95%; close enough for most n

    stats: Dict[str, Dict[str, float]] = {}

    for it in items:
        if "name" not in it or "indices" not in it:
            raise ValueError("Each item must contain 'name' and 'indices'.")
        name = str(it["name"])
        idxs = list(it["indices"])

        # Convert to durations in seconds
        durs = []
        for s, e in idxs:
            dur = (int(e) - int(s)) / float(fs)
            if dur > 0 and dur >= min_duration_sec:
                durs.append(dur)
        durs = np.asarray(durs, dtype=float)

        if durs.size == 0:
            # No valid segments: use NaNs and zero count
            stage_names.append(name)
            means.append(np.nan)
            lowers.append(np.nan)
            uppers.append(np.nan)
            counts.append(0)
            stds.append(np.nan)
            medians.append(np.nan)
            stats[name] = {
                "n_segments": 0,
                "mean_sec": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "std_sec": np.nan,
                "median_sec": np.nan,
            }
            continue

        m = float(np.mean(durs))
        sd = float(np.std(durs, ddof=1)) if durs.size > 1 else 0.0
        med = float(np.median(durs))
        n = int(durs.size)

        if ci_method == "bootstrap":
            rng = np.random.default_rng(42)  # fixed seed for reproducibility
            if n == 1:
                ci_lo, ci_hi = m, m
            else:
                # Percentile bootstrap on the mean
                boot_means = np.empty(n_boot, dtype=float)
                for b in range(n_boot):
                    sample = durs[rng.integers(0, n, size=n)]
                    boot_means[b] = np.mean(sample)
                lo_q = 100.0 * (alpha / 2.0)
                hi_q = 100.0 * (1.0 - alpha / 2.0)
                ci_lo, ci_hi = np.percentile(boot_means, [lo_q, hi_q])
        elif ci_method == "normal":
            se = sd / np.sqrt(n) if n > 0 else 0.0
            ci_lo, ci_hi = m - z * se, m + z * se
        else:
            raise ValueError("ci_method must be 'bootstrap' or 'normal'.")

        stage_names.append(name)
        means.append(m)
        lowers.append(ci_lo)
        uppers.append(ci_hi)
        counts.append(n)
        stds.append(sd)
        medians.append(med)
        stats[name] = {
            "n_segments": n,
            "mean_sec": m,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "std_sec": sd,
            "median_sec": med,
        }

    # Prepare error bars (asymmetric)
    means = np.array(means, dtype=float)
    lowers = np.array(lowers, dtype=float)
    uppers = np.array(uppers, dtype=float)
    yerr = np.vstack((np.clip(means - lowers, 0, None), np.clip(uppers - means, 0, None)))

    # Plot
    x = np.arange(len(stage_names))
    fig, ax = plt.subplots()
    ax.bar(x, means, yerr=yerr, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(stage_names)
    ax.set_ylabel("Average segment duration (s)")
    if title:
        ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    # Optional annotation: number of segments on top of bars
    for xi, m, n in zip(x, means, counts):
        if np.isfinite(m):
            ax.text(xi, m, f" n={n}", ha="center", va="bottom", fontsize=9)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()

    return fig, ax, stats


import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable, Dict, Any, List, Optional, Sequence


import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable, Dict, Any, Optional, Sequence, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, Optional, Dict, Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, Optional, List, Tuple



import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, Optional, List, Tuple

def plot_transition_matrix_from_label_paths(
    label_paths: Sequence[str],
    order_labels: Optional[Sequence[int]] = None,  # e.g. [1,2,3,4]
    connect_runs: bool = False,
    title: str = "Stage Transition Probabilities (sample-wise)",
    fs: int = 128,
    decimals: int = 3,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Build and plot a stage→stage transition matrix from label .npy files.
    Each file is a 1D array of integer labels (one run). Transitions are counted
    between consecutive samples within a file; optionally link runs end→start.

    Returns
    -------
    counts : (S,S) int array
    probs  : (S,S) float array with row-normalized P(next=j | current=i)
    labels_order : list[int] label IDs matching rows/cols
    """
    # Load runs (ignore empty)
    runs: List[np.ndarray] = []
    for p in label_paths:
        x = np.load(p, allow_pickle=False)
        x = np.ravel(x).astype(int)
        x = x[::4*128]
        if x.size >= 1:
            runs.append(x)
    if not runs:
        raise ValueError("No non-empty label sequences found.")

    # Label order (rows/cols)
    if order_labels is None:
        labels_order = sorted({int(v) for r in runs for v in np.unique(r)})[:3]
    else:
        labels_order = [int(v) for v in order_labels]
    S = len(labels_order)
    idx = {lbl: i for i, lbl in enumerate(labels_order)}

    counts = np.zeros((S, S), dtype=int)

    # Count transitions
    prev_last = None
    for r, x in enumerate(runs):
        # Optionally connect the last label of previous run to first of current
        if connect_runs and prev_last is not None and x.size > 0:
            if prev_last in idx and x[0] in idx:
                counts[idx[prev_last], idx[x[0]]] += 1

        if x.size >= 2:
            a, b = x[:-1], x[1:]
            # keep only pairs where both labels are known
            mask = np.isin(a, labels_order) & np.isin(b, labels_order)
            if np.any(mask):
                a2, b2 = a[mask], b[mask]
                # map labels to indices
                ai = np.fromiter((idx[int(v)] for v in a2), dtype=int, count=a2.size)
                bi = np.fromiter((idx[int(v)] for v in b2), dtype=int, count=b2.size)
                # accumulate counts
                flat = ai * S + bi
                binc = np.bincount(flat, minlength=S * S)
                counts += binc.reshape(S, S)

        prev_last = x[-1]

    # Row-normalize
    row_sums = counts.sum(axis=1, keepdims=True).astype(float)
    probs = np.divide(counts, row_sums, out=np.zeros_like(counts, dtype=float), where=row_sums > 0)
    
    # Plot
    fig, ax = plt.subplots()
    im = ax.imshow(probs, origin="upper", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(S)); ax.set_yticks(np.arange(S))
    ax.set_xticklabels([NAME_MAP[label] for label in labels_order], rotation=45, ha="right")
    ax.set_yticklabels([NAME_MAP[label] for label in labels_order])
    ax.set_xlabel("To stage"); ax.set_ylabel("From stage"); ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax); cbar.set_label("P(next | current)")

    # annotate with more precision so small off-diagonals are visible
    fmt = f"{{:.{decimals}f}}"
    for i in range(S):
        for j in range(S):
            ax.text(j, i, fmt.format(probs[i, j]), ha="center", va="center", fontsize=9)

    fig.tight_layout()
    plt.show()

    return counts, probs, labels_order


def plot_stage_psd_with_confidence_intervals(
    path_tuples: Iterable[Tuple[str, str]],
    stage_labels: Optional[Sequence[int]] = None,
    stage_names: Optional[Dict[int, str]] = None,
    fs: float = 128.0,
    n_fft: int = 128,
    band: Tuple[float, float] = (1.0, 60.0),
    normalize: str = "area",
    smooth_hz: float = 0.0,
    ci_level: float = 0.95,
    logy: bool = True,
    annotate_bands: bool = True,
    title: Optional[str] = "Sleep Stage Spectra with 95% CI",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot average stage PSDs across runs with confidence intervals.
    
    For each run (eeg_path, labels_path), computes PSDs for each stage,
    then plots the mean PSD ± confidence interval across runs.

    Parameters
    ----------
    path_tuples : iterable of (eeg_path, labels_path)
        Each tuple contains paths to EEG and labels .npy files for one run.
    stage_labels : sequence of int, optional
        Which stage labels to include. If None, uses all found stages.
    stage_names : dict[int, str], optional
        Mapping from stage labels to display names.
    fs : float
        Sampling rate in Hz.
    n_fft : int
        FFT length for PSD computation.
    band : (fmin, fmax)
        Frequency range to display and use for normalization.
    normalize : str
        Normalization method: 'area', 'max', 'zscore', or 'none'.
    smooth_hz : float
        Optional smoothing window in Hz.
    ci_level : float
        Confidence level (e.g., 0.95 for 95% CI).
    logy : bool
        Use log scale for y-axis.
    annotate_bands : bool
        Draw EEG frequency band boundaries.
    title : str
        Plot title.
    save_path : str, optional
        Save figure to this path.
    show : bool
        Display the figure.

    Returns
    -------
    fig, ax : matplotlib figure and axes
    run_data : dict
        Statistics per stage: {stage_name: {'f': freq_array, 'mean': mean_psd, 
                                           'ci_lower': lower_bound, 'ci_upper': upper_bound}}
    """
    from typing import Dict, List
    
    path_tuples = list(path_tuples)
    if len(path_tuples) == 0:
        raise ValueError("No path tuples provided")
    
    # Default stage names
    if stage_names is None:
        stage_names = {1: 'Wake', 2: 'NREM', 3: 'REM', 4: 'Artifact'}
    
    # Collect PSDs per run per stage
    run_psds: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}  # stage -> [(f, psd), ...]
    
    print(f"Processing {len(path_tuples)} runs...")
    for i, (eeg_path, labels_path) in enumerate(path_tuples):
        print(f"  Run {i+1}/{len(path_tuples)}: {eeg_path}")
        
        try:
            segments_by_label, _ = split_eeg_by_label(eeg_path, labels_path, as_lists=False)
            
            for stage, segments in segments_by_label.items():
                if stage_labels is not None and stage not in stage_labels:
                    continue
                if stage == 4:  # Skip artifacts by default
                    continue
                    
                if len(segments) == 0:
                    continue
                    
                # Compute PSD for this stage in this run
                f, psd_avg = average_psd(signals=segments, fs=fs, n_fft=n_fft, band=band)
                
                if stage not in run_psds:
                    run_psds[stage] = []
                run_psds[stage].append((f, psd_avg))
                
        except Exception as e:
            print(f"    Warning: Failed to process run {i+1}: {e}")
            continue
    
    if not run_psds:
        raise ValueError("No valid PSDs computed from any run")
    
    # Process each stage
    run_data = {}
    fmin, fmax = float(band[0]), float(band[1])
    alpha = 1.0 - ci_level
    
    # Set up professional styling
    plt.style.use('default')  # Reset to default first
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Professional color palette for sleep stages
    stage_colors = {
        'Wake': '#E74C3C',     # Red - alert/active state
        'NREM': '#3498DB',     # Blue - calm/deep sleep
        'REM': '#9B59B6',      # Purple - dream state
        'Artifact': '#95A5A6'  # Gray - neutral
    }
    
    # Line styles for distinction
    line_styles = ['-', '--', '-.', ':']
    
    for stage_idx, (stage, psd_list) in enumerate(run_psds.items()):
        stage_name = stage_names.get(stage, f"Stage_{stage}")
        
        if len(psd_list) < 2:
            print(f"Warning: Only {len(psd_list)} run(s) for {stage_name}, skipping CI")
            continue
            
        # Get common frequency grid (use the first one as reference)
        f_ref = psd_list[0][0]
        
        # Collect and align PSDs to common frequency grid
        aligned_psds = []
        for f, psd in psd_list:
            # Interpolate to reference frequency grid if needed
            if not np.array_equal(f, f_ref):
                psd_interp = np.interp(f_ref, f, psd)
            else:
                psd_interp = psd
            aligned_psds.append(psd_interp)
        
        psds_array = np.stack(aligned_psds, axis=0)  # shape: (n_runs, n_freqs)
        
        # Apply normalization and smoothing to each run separately
        for i in range(psds_array.shape[0]):
            # Restrict to band for normalization
            mask = (f_ref >= fmin) & (f_ref <= fmax)
            f_band = f_ref[mask]
            psd_band = psds_array[i, mask]
            
            # Smoothing using simple moving average
            if smooth_hz > 0:
                df = np.median(np.diff(f_band)) if f_band.size > 1 else 1.0
                win_pts = max(1, int(smooth_hz / df))
                psd_band = _moving_average(psd_band, win_pts)
            
            # Normalization
            psd_band = _normalize_curve(psd_band, f_band, normalize)
            psds_array[i, mask] = psd_band
        
        # Compute statistics
        mean_psd = np.mean(psds_array, axis=0)
        std_psd = np.std(psds_array, axis=0, ddof=1)
        n_runs = psds_array.shape[0]
        
        # Confidence interval using t-distribution approximation
        if n_runs > 1:
            # Use approximation: for 95% CI and reasonable n, t ≈ 2.0
            if ci_level == 0.95:
                if n_runs >= 30:
                    t_val = 1.96  # Normal approximation
                elif n_runs >= 10:
                    t_val = 2.1   # Conservative t-value
                else:
                    t_val = 2.5   # More conservative for small samples
            else:
                # Simple approximation for other confidence levels
                t_val = 2.0
            
            margin = t_val * std_psd / np.sqrt(n_runs)
            ci_lower = mean_psd - margin
            ci_upper = mean_psd + margin
        else:
            ci_lower = mean_psd
            ci_upper = mean_psd
        
        # Restrict to display band
        mask = (f_ref >= fmin) & (f_ref <= fmax)
        f_plot = f_ref[mask]
        mean_plot = mean_psd[mask]
        ci_lower_plot = ci_lower[mask]
        ci_upper_plot = ci_upper[mask]
        
        # Ensure positive values for log scale
        if logy:
            mean_plot = np.maximum(mean_plot, np.finfo(float).tiny)
            ci_lower_plot = np.maximum(ci_lower_plot, np.finfo(float).tiny)
            ci_upper_plot = np.maximum(ci_upper_plot, np.finfo(float).tiny)
        
        # Get color and style
        color = stage_colors.get(stage_name, f'C{stage_idx}')
        line_style = line_styles[stage_idx % len(line_styles)]
        
        # Plot with enhanced styling
        if logy:
            line = ax.semilogy(f_plot, mean_plot, 
                             label=f"{stage_name} (n={n_runs})",
                             color=color, linewidth=2.5, linestyle=line_style,
                             alpha=0.9)
            ax.fill_between(f_plot, ci_lower_plot, ci_upper_plot, 
                          alpha=0.2, color=color, label=f'{int(ci_level*100)}% CI')
        else:
            line = ax.plot(f_plot, mean_plot, 
                         label=f"{stage_name} (n={n_runs})",
                         color=color, linewidth=2.5, linestyle=line_style,
                         alpha=0.9)
            ax.fill_between(f_plot, ci_lower_plot, ci_upper_plot, 
                          alpha=0.2, color=color, label=f'{int(ci_level*100)}% CI')
        
        # Store results
        run_data[stage_name] = {
            'f': f_plot,
            'mean': mean_plot,
            'ci_lower': ci_lower_plot,
            'ci_upper': ci_upper_plot,
            'n_runs': n_runs
        }
    
    # Enhanced formatting
    ax.set_xlim(fmin, fmax)
    ax.set_xlabel("Frequency (Hz)", fontsize=14, fontweight='bold')
    ylabel = "PSD (power/Hz)" if normalize == "none" else "Normalized PSD"
    if logy:
        ylabel += " (log scale)"
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    
    if title:
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Enhanced grid
    ax.grid(True, which="major", linestyle="-", alpha=0.3, color='gray')
    ax.grid(True, which="minor", linestyle=":", alpha=0.2, color='gray')
    
    # Professional legend
    legend = ax.legend(title="Sleep Stage", loc="best", frameon=True, 
                      fancybox=True, shadow=True, fontsize=11)
    legend.set_title("Sleep Stage", prop={'size': 12, 'weight': 'bold'})
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # Annotate frequency bands with improved styling
    if annotate_bands:
        boundaries = [4, 8, 12, 16, 30]
        band_labels = [("δ", 1, 4), ("θ", 4, 8), ("α", 8, 12), ("σ", 12, 16), ("β", 16, 30), ("γ", 30, 60)]
        band_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        
        for i, b in enumerate(boundaries):
            if fmin < b < fmax:
                ax.axvline(b, linestyle="--", alpha=0.5, color='darkgray', linewidth=1)
        
        ylim = ax.get_ylim()
        if logy:
            y_text = np.exp((np.log(ylim[0]) + np.log(ylim[1])) / 2.0)
        else:
            y_text = (ylim[0] + ylim[1]) / 2.0
        
        for i, (sym, a, b) in enumerate(band_labels):
            if b <= fmin or a >= fmax:
                continue
            xmid = max(fmin, min(fmax, (a + b) / 2.0))
            color = band_colors[i % len(band_colors)]
            ax.text(xmid, y_text, sym, ha="center", va="center", 
                   fontsize=12, fontweight='bold', color='white',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=color, 
                           alpha=0.8, edgecolor='white', linewidth=1))
    
    # Enhance spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('gray')
    
    # Improve tick formatting
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.2)
    ax.tick_params(axis='both', which='minor', width=0.8)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300, facecolor='white')
    if show:
        plt.show()
    
    return fig, ax, run_data


def plot_stage_bandpowers_with_confidence_intervals(
    path_tuples: Iterable[Tuple[str, str]],
    stage_labels: Optional[Sequence[int]] = None,
    stage_names: Optional[Dict[int, str]] = None,
    bands: Dict[str, Tuple[float, float]] = DEFAULT_BANDS,
    total_band: Tuple[float, float] = (1.0, 60.0),
    relative: bool = True,
    fs: float = 128.0,
    n_fft: int = 128,
    ci_level: float = 0.95,
    title: Optional[str] = "Sleep Stage Bandpowers with 95% CI Across Runs",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot average stage bandpowers across runs with confidence intervals.
    
    For each run (eeg_path, labels_path), computes bandpowers for each stage,
    then plots the mean bandpower ± confidence interval across runs.

    Parameters
    ----------
    path_tuples : iterable of (eeg_path, labels_path)
        Each tuple contains paths to EEG and labels .npy files for one run.
    stage_labels : sequence of int, optional
        Which stage labels to include. If None, uses all found stages.
    stage_names : dict[int, str], optional
        Mapping from stage labels to display names.
    bands : dict[str, tuple[float, float]]
        Frequency bands to compute. Keys are band names, values are (low_Hz, high_Hz).
    total_band : (fmin, fmax)
        Total frequency range for relative power normalization.
    relative : bool
        If True, compute relative bandpowers (proportion of total power).
        If False, compute absolute bandpowers.
    fs : float
        Sampling rate in Hz.
    n_fft : int
        FFT length for PSD computation.
    ci_level : float
        Confidence level (e.g., 0.95 for 95% CI).
    title : str
        Plot title.
    save_path : str, optional
        Save figure to this path.
    show : bool
        Display the figure.

    Returns
    -------
    fig, ax : matplotlib figure and axes
    run_data : dict
        Statistics per stage and band: {stage_name: {band_name: {'mean': float, 
                                                                'ci_lower': float, 
                                                                'ci_upper': float,
                                                                'n_runs': int}}}
    """
    from typing import Dict, List
    
    path_tuples = list(path_tuples)
    if len(path_tuples) == 0:
        raise ValueError("No path tuples provided")
    
    # Default stage names
    if stage_names is None:
        stage_names = {1: 'Wake', 2: 'NREM', 3: 'REM', 4: 'Artifact'}
    
    # Collect bandpowers per run per stage
    run_bandpowers: Dict[int, List[Dict[str, float]]] = {}  # stage -> [bandpower_dict, ...]
    
    print(f"Processing {len(path_tuples)} runs for bandpower analysis...")
    for i, (eeg_path, labels_path) in enumerate(path_tuples):
        print(f"  Run {i+1}/{len(path_tuples)}: {eeg_path}")
        
        try:
            segments_by_label, _ = split_eeg_by_label(eeg_path, labels_path, as_lists=False)
            
            for stage, segments in segments_by_label.items():
                if stage_labels is not None and stage not in stage_labels:
                    continue
                if stage == 4:  # Skip artifacts by default
                    continue
                    
                if len(segments) == 0:
                    continue
                    
                # Compute PSD for this stage in this run
                f, psd_avg = average_psd(signals=segments, fs=fs, n_fft=n_fft, band=total_band)
                
                # Compute bandpowers from the PSD
                bandpowers = _bandpowers_from_psd(f, psd_avg, bands, total_band, relative)
                
                if stage not in run_bandpowers:
                    run_bandpowers[stage] = []
                run_bandpowers[stage].append(bandpowers)
                
        except Exception as e:
            print(f"    Warning: Failed to process run {i+1}: {e}")
            continue
    
    if not run_bandpowers:
        raise ValueError("No valid bandpowers computed from any run")
    
    # Process statistics for each stage and band
    run_data = {}
    alpha = 1.0 - ci_level
    
    # Prepare data for plotting
    stage_names_list = []
    band_names = list(bands.keys())
    n_bands = len(band_names)
    
    # Collect data for each stage that has enough runs
    plot_data = {}
    for stage, bandpower_list in run_bandpowers.items():
        stage_name = stage_names.get(stage, f"Stage_{stage}")
        
        if len(bandpower_list) < 2:
            print(f"Warning: Only {len(bandpower_list)} run(s) for {stage_name}, skipping CI")
            continue
            
        stage_names_list.append(stage_name)
        plot_data[stage_name] = {}
        run_data[stage_name] = {}
        
        # Convert list of dicts to array: rows=runs, cols=bands
        n_runs = len(bandpower_list)
        bandpower_matrix = np.zeros((n_runs, n_bands))
        
        for run_idx, bp_dict in enumerate(bandpower_list):
            for band_idx, band_name in enumerate(band_names):
                bandpower_matrix[run_idx, band_idx] = bp_dict.get(band_name, np.nan)
        
        # Compute statistics for each band
        for band_idx, band_name in enumerate(band_names):
            band_values = bandpower_matrix[:, band_idx]
            # Remove NaN values
            band_values = band_values[~np.isnan(band_values)]
            
            if len(band_values) == 0:
                continue
                
            mean_val = np.mean(band_values)
            std_val = np.std(band_values, ddof=1) if len(band_values) > 1 else 0.0
            n_valid = len(band_values)
            
            # Confidence interval using t-distribution approximation
            if n_valid > 1:
                if ci_level == 0.95:
                    if n_valid >= 30:
                        t_val = 1.96  # Normal approximation
                    elif n_valid >= 10:
                        t_val = 2.1   # Conservative t-value
                    else:
                        t_val = 2.5   # More conservative for small samples
                else:
                    t_val = 2.0
                
                margin = t_val * std_val / np.sqrt(n_valid)
                ci_lower = mean_val - margin
                ci_upper = mean_val + margin
            else:
                ci_lower = mean_val
                ci_upper = mean_val
            
            plot_data[stage_name][band_name] = {
                'mean': mean_val,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'n_runs': n_valid
            }
            
            run_data[stage_name][band_name] = {
                'mean': mean_val,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'std': std_val,
                'n_runs': n_valid
            }
    
    if not plot_data:
        raise ValueError("No stages with sufficient runs for confidence intervals")
    
    # Create the plot with professional styling
    plt.style.use('default')  # Reset to default first
    n_stages = len(stage_names_list)
    x = np.arange(n_bands)
    width = 0.8 / max(1, n_stages)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Professional color palette for sleep stages
    stage_colors = {
        'Wake': '#E74C3C',     # Red - alert/active state
        'NREM': '#3498DB',     # Blue - calm/deep sleep  
        'REM': '#9B59B6',      # Purple - dream state
        'Artifact': '#95A5A6'  # Gray - neutral
    }
    
    # Enhanced bar patterns for distinction
    bar_patterns = ['', '///', '...', '+++']
    edge_colors = ['white', 'white', 'white', 'white']
    
    for stage_idx, stage_name in enumerate(stage_names_list):
        means = []
        ci_lowers = []
        ci_uppers = []
        
        for band_name in band_names:
            if band_name in plot_data[stage_name]:
                data = plot_data[stage_name][band_name]
                means.append(data['mean'])
                ci_lowers.append(data['ci_lower'])
                ci_uppers.append(data['ci_upper'])
            else:
                means.append(0)
                ci_lowers.append(0)
                ci_uppers.append(0)
        
        means = np.array(means)
        ci_lowers = np.array(ci_lowers)
        ci_uppers = np.array(ci_uppers)
        
        # Calculate error bars (asymmetric)
        yerr_lower = np.maximum(0, means - ci_lowers)
        yerr_upper = np.maximum(0, ci_uppers - means)
        yerr = np.vstack([yerr_lower, yerr_upper])
        
        # Get colors and styling
        color = stage_colors.get(stage_name, f'C{stage_idx}')
        pattern = bar_patterns[stage_idx % len(bar_patterns)]
        edge_color = edge_colors[stage_idx % len(edge_colors)]
        
        # Plot bars with enhanced styling
        x_pos = x + stage_idx * width
        bars = ax.bar(x_pos, means, width=width, label=stage_name, 
                     yerr=yerr, capsize=4, alpha=0.85,
                     color=color, hatch=pattern, 
                     edgecolor=edge_color, linewidth=1.5,
                     error_kw={'linewidth': 2, 'capthick': 2})
        
        # Add number of runs as professional annotations
        for j, (bar, mean_val) in enumerate(zip(bars, means)):
            if mean_val > 0 and band_names[j] in plot_data[stage_name]:
                n_runs = plot_data[stage_name][band_names[j]]['n_runs']
                # Position text above error bars
                text_y = bar.get_height() + yerr_upper[j] + max(means) * 0.02
                ax.text(bar.get_x() + bar.get_width()/2, text_y,
                       f'n={n_runs}', ha='center', va='bottom', 
                       fontsize=9, fontweight='bold', 
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', 
                               alpha=0.8, edgecolor='gray', linewidth=0.5))
    
    # Enhanced formatting
    ax.set_xticks(x + (n_stages - 1) * width / 2.0)
    ax.set_xticklabels([band.capitalize() for band in band_names], 
                      fontsize=12, fontweight='bold')
    ax.set_xlabel("Frequency Band", fontsize=14, fontweight='bold')
    
    y_label = "Relative Bandpower" if relative else "Absolute Bandpower"
    if relative:
        y_label += " (proportion)"
    ax.set_ylabel(y_label, fontsize=14, fontweight='bold')
    
    if title:
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Professional grid
    ax.grid(True, axis="y", linestyle="-", alpha=0.3, color='gray')
    ax.grid(True, axis="y", which='minor', linestyle=":", alpha=0.2, color='gray')
    
    # Enhanced legend
    legend = ax.legend(title="Sleep Stage", loc="upper right", frameon=True,
                      fancybox=True, shadow=True, fontsize=12)
    legend.set_title("Sleep Stage", prop={'size': 13, 'weight': 'bold'})
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.95)
    legend.get_frame().set_edgecolor('gray')
    
    # Add percentage formatting for relative bandpower
    if relative:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    
    # Enhance spines and ticks
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('gray')
    
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.2)
    ax.tick_params(axis='both', which='minor', width=0.8)
    
    # Add subtle background color
    ax.set_facecolor('#FAFAFA')
    
    # Improve layout
    plt.tight_layout()
    
    # Add band frequency information as subplot text
    band_info = []
    for band_name, (low, high) in bands.items():
        band_info.append(f"{band_name.capitalize()}: {low}-{high} Hz")
    
    if len(band_info) <= 6:  # Only show if not too many bands
        info_text = " | ".join(band_info)
        fig.text(0.5, 0.02, info_text, ha='center', va='bottom', 
                fontsize=10, style='italic', color='gray')
        plt.subplots_adjust(bottom=0.1)
    
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300, facecolor='white')
    if show:
        plt.show()
    
    return fig, ax, run_data


def get_all_paths(n: int = None):
    df = pd.read_csv("data/ds006366_processed/metadata.csv")
    eeg_paths = []
    labels_paths = []
    for row in df.iloc:
        path = row["path"]
        eeg_paths.append(path + "/EEG1.npy")
        labels_paths.append(path + "/labels.npy")
    if n:
        return eeg_paths[:n], labels_paths[:n]
    return eeg_paths, labels_paths

if __name__ == "__main__":
    
    eeg_paths, labels_paths = get_all_paths(n=50)  # Use smaller subset for testing
    
    segments_by_label, indices_by_label = combine_splits_from_path_tuples(zip(eeg_paths, labels_paths))
    
    # segments_by_label,indices_by_label = split_eeg_by_label(
    #     eeg_path="data/ds006366_processed/sub-001/1/EEG1.npy",
    #     labels_path="data/ds006366_processed/sub-001/1/labels.npy",
    # )
    
    NAME_MAP = {
        1: 'Wake',
        2: 'NREM',
        3: 'REM',
        4: 'Artifact'
    }
    
    stage_frequency_map = []
    for stage in segments_by_label:
        if stage == 4:
            continue
        f, psd_avg = average_psd(signals=segments_by_label[stage], fs=128, n_fft=128, band=(1,60))
        stage_frequency_map.append(
            {
                "name": NAME_MAP[stage],
                "f": f,
                "psd_avg": psd_avg
            }
        )
    
    
    stage_indices_map = []
    for stage in indices_by_label:
        if stage == 4:
            continue
        stage_indices_map.append(
            {
                "name": NAME_MAP[stage],
                "indices": indices_by_label[stage]
            }
        )
    
    # plot_stage_psd_comparison(spectra=stage_frequency_map)
    
    # plot_stage_bandpowers(spectra=stage_frequency_map)    
    
    # plot_stage_mean_durations(items=stage_indices_map, fs = 128)
    
    # plot_transition_matrix_from_label_paths(labels_paths)
    
    # Test the new confidence interval function
    print("Plotting stage PSDs with confidence intervals across runs...")
    plot_stage_psd_with_confidence_intervals(
        path_tuples=list(zip(eeg_paths, labels_paths)),
        stage_names=NAME_MAP,
        fs=128,
        n_fft=256,
        band=(0.1, 60),
        normalize="area",
        smooth_hz=0.5,  # Small amount of smoothing
        ci_level=0.95,
        title="Sleep Stage Spectra with 95% CI Across Runs"
    )
    
    # Test the new bandpower confidence interval function
    print("\nPlotting stage bandpowers with confidence intervals across runs...")
    plot_stage_bandpowers_with_confidence_intervals(
        path_tuples=list(zip(eeg_paths, labels_paths)),
        stage_names=NAME_MAP,
        bands=DEFAULT_BANDS,
        total_band=(1.0, 60.0),
        relative=True,
        fs=128,
        n_fft=256,
        ci_level=0.95,
        title="Sleep Stage Bandpowers with 95% CI Across Runs"
    )