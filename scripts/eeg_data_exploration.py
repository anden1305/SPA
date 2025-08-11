import numpy as np
from pathlib import Path
import numpy as np

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
        # Use newer numpy integration function (trapz deprecated)
        area = np.trapezoid(y, f)
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
    close: bool = False,
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

    fig, ax = plt.subplots(figsize=(12, 5))

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
    if close:
        plt.close(fig)

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
    "mid-gamma": (40.0, 50.0),
    "high-gamma": (50.0, 60.0),
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

    return float(np.trapezoid(pi2, fi2))

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
    close: bool = False,
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

    fig, ax = plt.subplots(figsize=(12, 5))
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
    if close:
        plt.close(fig)

    return fig, ax, values



if __name__ == "__main__":
    # Mapping from numeric stage codes to human-readable names
    NAME_MAP = {
        1: 'Wake',
        2: 'NREM',
        3: 'REM'
    }

    # Root directory for subjectwise outputs
    subjectwise_root = Path("results/data_exploration/subjectwise")
    subjectwise_root.mkdir(parents=True, exist_ok=True)

    base = Path("data") / "ds006366_processed"
    for subj_dir in sorted(base.glob("sub-*")):
        subj_id = subj_dir.name
        subject_dir = subjectwise_root / subj_id
        runs_dir = subject_dir / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)

        per_stage_segments_agg = {}

        # Iterate runs for this subject
        for run_dir in sorted(subj_dir.glob("*")):
            if not run_dir.is_dir():
                continue
            eeg_file = run_dir / "EEG1.npy"
            labels_file = run_dir / "labels.npy"
            if not eeg_file.exists() or not labels_file.exists():
                continue
            try:
                segs, _ = split_eeg_by_label(eeg_file, labels_file, as_lists=False)
            except Exception as e:
                print(f"Skipping {run_dir} due to error: {e}")
                continue
            # Build per-run stage frequency map
            run_stage_frequency_map = []
            for stage_code, seg_list in segs.items():
                if stage_code == 4 or not seg_list:  # skip Artifact
                    continue
                per_stage_segments_agg.setdefault(stage_code, []).extend(seg_list)
                f_run, psd_run = average_psd(signals=seg_list, fs=128, n_fft=512)
                run_stage_frequency_map.append({
                    "name": NAME_MAP.get(stage_code, f"Stage{stage_code}"),
                    "f": f_run,
                    "psd_avg": psd_run
                })
            if run_stage_frequency_map:
                plot_stage_psd_comparison(
                    spectra=run_stage_frequency_map,
                    save_path=str(runs_dir / f"{subj_id}_{run_dir.name}_psd.png"),
                    title=f"Sleep Stage Spectra ({subj_id} run {run_dir.name})",
                    show=False,
                    close=True
                )
                plot_stage_bandpowers(
                    spectra=run_stage_frequency_map,
                    save_path=str(runs_dir / f"{subj_id}_{run_dir.name}_bandpowers.png"),
                    title=f"Bandpower by Sleep Stage ({subj_id} run {run_dir.name})",
                    show=False,
                    close=True
                )

        # Aggregated subject plots
        if per_stage_segments_agg:
            subj_stage_frequency_map = []
            for stage_code, seg_list in per_stage_segments_agg.items():
                if not seg_list:
                    continue
                f_agg, psd_agg = average_psd(signals=seg_list, fs=128, n_fft=512)
                subj_stage_frequency_map.append({
                    "name": NAME_MAP.get(stage_code, f"Stage{stage_code}"),
                    "f": f_agg,
                    "psd_avg": psd_agg
                })
            if subj_stage_frequency_map:
                plot_stage_psd_comparison(
                    spectra=subj_stage_frequency_map,
                    save_path=str(subject_dir / f"{subj_id}_aggregated_psd.png"),
                    title=f"Sleep Stage Spectra ({subj_id} aggregated)",
                    show=False,
                    close=True
                )
                plot_stage_bandpowers(
                    spectra=subj_stage_frequency_map,
                    save_path=str(subject_dir / f"{subj_id}_aggregated_bandpowers.png"),
                    title=f"Bandpower by Sleep Stage ({subj_id} aggregated)",
                    show=False,
                    close=True
                )
        else:
            print(f"No valid segments for {subj_id}; skipping aggregated plots.")

