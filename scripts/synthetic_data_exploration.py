"""Quick exploration of generated synthetic sleep data.

Loads a saved synthetic run from data/synthetic (default prefix 'synthetic_default')
and produces basic summaries:
- Label counts & transition matrix
- Per-epoch standard deviation distribution
- Mean power spectrum per class
Outputs figures into results/synthetic_exploration.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Iterable, Tuple, Optional

SAVE_DIR = Path("data") / "synthetic"
RESULTS = Path("results") / "synthetic_exploration"
RESULTS.mkdir(parents=True, exist_ok=True)

def load_run(prefix: str = "synthetic_default"):
    eeg_path = SAVE_DIR / f"{prefix}_eeg.npy"
    labels_path = SAVE_DIR / f"{prefix}_labels.npy"
    meta_path = SAVE_DIR / f"{prefix}_metadata.json"
    if not eeg_path.exists():
        raise FileNotFoundError(eeg_path)
    eeg = np.load(eeg_path)
    labels = np.load(labels_path)
    import json
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    return eeg, labels, meta

def summarize_labels(labels: np.ndarray, meta: dict):
    stages = meta.get("stages", [])
    unique, counts = np.unique(labels, return_counts=True)
    mapping = {u: stages[u] if u < len(stages) else str(u) for u in unique}
    summary_lines = []
    for u,c in zip(unique, counts):
        summary_lines.append(f"{u} ({mapping[u]}): {c} ({c/len(labels):.2%})")
    (RESULTS / "label_summary.txt").write_text("\n".join(summary_lines))
    # Bar plot
    plt.figure(figsize=(5,3))
    # Use hue to avoid future deprecation warning
    stage_names = [mapping[u] for u in unique]
    import pandas as pd
    df_counts = pd.DataFrame({"stage": stage_names, "count": counts})
    sns.barplot(data=df_counts, x="stage", y="count", hue="stage", palette="viridis", legend=False)
    plt.title("Stage counts")
    plt.ylabel("Epochs")
    plt.tight_layout()
    plt.savefig(RESULTS / "stage_counts.png", dpi=150)
    plt.close()
    # Transition matrix
    tm = np.zeros((len(stages), len(stages)))
    for a,b in zip(labels[:-1], labels[1:]):
        tm[a,b]+=1
    row_sums = tm.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        tm_prob = np.where(row_sums>0, tm/row_sums, 0)
    plt.figure(figsize=(4,4))
    sns.heatmap(tm_prob, annot=True, fmt=".2f", cmap="magma", xticklabels=stages, yticklabels=stages)
    plt.title("Transition matrix (P(next|current))")
    plt.tight_layout()
    plt.savefig(RESULTS / "transition_matrix.png", dpi=150)
    plt.close()

def epoch_std_distribution(eeg: np.ndarray):
    stds = eeg.std(axis=1)
    unique_vals = np.unique(stds)
    plt.figure(figsize=(5,3))
    if len(unique_vals) <= 1:
        # Degenerate case: all epochs identical amplitude
        val = float(unique_vals[0]) if unique_vals.size else 0.0
        plt.axvline(val, color="steelblue")
        plt.text(val, 0.5, f"std={val:.3g}", rotation=90, va="center")
        plt.xlim(val - 1 if val != 0 else -1, val + 1 if val != 0 else 1)
        plt.ylim(0,1)
    else:
        n_bins = min(30, len(unique_vals))
        # Guard against zero range
        data_range = stds.max() - stds.min()
        if data_range <= 1e-12:
            n_bins = 1
        sns.histplot(stds, bins=n_bins, kde=len(unique_vals) > 5)
    plt.xlabel("Epoch std (a.u.)")
    plt.title("Epoch amplitude distribution")
    plt.tight_layout()
    plt.savefig(RESULTS / "epoch_std_hist.png", dpi=150)
    plt.close()
    np.savetxt(RESULTS / "epoch_std_summary.txt", [stds.mean(), np.median(stds), stds.min(), stds.max()], header="mean,median,min,max")

def _compute_epoch_psd(eeg: np.ndarray, sr: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute simple per-epoch power spectra using FFT (power = |FFT|^2).

    Returns
    -------
    freqs : (F,) array
        Frequency bins (Hz)
    psds : (E, F) array
        Power spectrum per epoch
    """
    n = eeg.shape[1]
    freqs = np.fft.rfftfreq(n, d=1 / sr)
    fft_vals = np.fft.rfft(eeg, axis=1)
    psd = (np.abs(fft_vals) ** 2) / n  # simple normalization
    return freqs, psd


def _spectral_confidence(psds: Iterable[np.ndarray], alpha: float = 0.05) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute mean and (1-alpha) confidence interval across a collection of PSD arrays.

    Parameters
    ----------
    psds : iterable of (F,) arrays OR a 2D array (N,F)
    alpha : float
        Significance level (default 0.05 gives 95% CI)
    """
    psds_arr = np.asarray(psds)
    if psds_arr.ndim == 1:
        psds_arr = psds_arr[None, :]
    mean = psds_arr.mean(axis=0)
    n = psds_arr.shape[0]
    if n < 2:
        return mean, None, None
    # Normal approximation
    std = psds_arr.std(axis=0, ddof=1)
    from math import sqrt
    z = 1.96  # for ~95%
    half_width = z * std / sqrt(n)
    return mean, np.clip(mean - half_width, a_min=0, a_max=None), mean + half_width


def mean_power_spectrum(eeg: np.ndarray, labels: np.ndarray, meta: dict):
    """Create a detailed multi-panel PSD analysis with confidence intervals.

    Panels:
      (0,0) Log-scale PSD 0.5-50 Hz
      (0,1) Linear 0.5-10 Hz detail
      (1,0) Linear 10-50 Hz detail
      (1,1) Normalized PSD 0.5-50 Hz
    """
    sr = meta.get("sampling_rate_hz", meta.get("config_used", {}).get("sampling_rate_hz", 1))
    stages = meta.get("stages", [])
    if eeg.ndim != 2:
        raise ValueError("Expected eeg shape (epochs, samples)")

    freqs, all_psd = _compute_epoch_psd(eeg, sr)

    # Collect per-stage psds
    stage_psds = {}
    for idx, stage in enumerate(stages):
        mask = labels == idx
        if mask.any():
            stage_psds[stage] = all_psd[mask]

    if not stage_psds:
        print("No stage PSD data found; skipping detailed PSD plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Detailed Power Spectral Density by Sleep Stage (with 95% CI)', fontsize=16, fontweight='bold')

    # Consistent ordering based on appearance in meta
    ordered_stages = [s for s in stages if s in stage_psds]
    palette = sns.color_palette("tab10", n_colors=len(ordered_stages))
    colors = {s: c for s, c in zip(ordered_stages, palette)}

    def _mask_range(lo, hi):
        return (freqs >= lo) & (freqs <= hi)

    # Plot 1: log-scale 0.5-50 Hz
    ax = axes[0, 0]
    freq_mask_main = _mask_range(0.5, 50)
    for stage in ordered_stages:
        mean_psd, ci_lo, ci_hi = _spectral_confidence(stage_psds[stage])
        color = colors[stage]
        ax.semilogy(freqs[freq_mask_main], mean_psd[freq_mask_main], label=stage, color=color, linewidth=2)
        if ci_lo is not None:
            ax.fill_between(freqs[freq_mask_main], ci_lo[freq_mask_main], ci_hi[freq_mask_main], color=color, alpha=0.2)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density (log)')
    ax.set_title('Average PSD with 95% CI (0.5-50 Hz)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 2: 0.5-10 Hz linear
    ax = axes[0, 1]
    freq_mask_low = _mask_range(0.5, 10)
    for stage in ordered_stages:
        mean_psd, ci_lo, ci_hi = _spectral_confidence(stage_psds[stage])
        color = colors[stage]
        ax.plot(freqs[freq_mask_low], mean_psd[freq_mask_low], label=stage, color=color, linewidth=2)
        if ci_lo is not None:
            ax.fill_between(freqs[freq_mask_low], ci_lo[freq_mask_low], ci_hi[freq_mask_low], color=color, alpha=0.2)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density')
    ax.set_title('Low Frequency Detail 0.5-10 Hz (95% CI)', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 3: 10-50 Hz linear
    ax = axes[1, 0]
    freq_mask_high = _mask_range(10, 50)
    for stage in ordered_stages:
        mean_psd, ci_lo, ci_hi = _spectral_confidence(stage_psds[stage])
        color = colors[stage]
        ax.plot(freqs[freq_mask_high], mean_psd[freq_mask_high], label=stage, color=color, linewidth=2)
        if ci_lo is not None:
            ax.fill_between(freqs[freq_mask_high], ci_lo[freq_mask_high], ci_hi[freq_mask_high], color=color, alpha=0.2)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density')
    ax.set_title('High Frequency Detail 10-50 Hz (95% CI)', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 4: Normalized PSD 0.5-50 Hz
    ax = axes[1, 1]
    norm_psds = {}
    for stage in ordered_stages:
        # Normalize each epoch within mask range
        epoch_psds = stage_psds[stage][:, freq_mask_main]
        totals = epoch_psds.sum(axis=1, keepdims=True)
        valid = totals[:, 0] > 0
        if not valid.any():
            continue
        norm_psds[stage] = epoch_psds[valid] / totals[valid]
        mean_norm, ci_lo, ci_hi = _spectral_confidence(norm_psds[stage])
        color = colors[stage]
        ax.plot(freqs[freq_mask_main], mean_norm, label=stage, color=color, linewidth=2)
        if ci_lo is not None:
            ax.fill_between(freqs[freq_mask_main], ci_lo, ci_hi, color=color, alpha=0.2)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Normalized Power')
    ax.set_title('Normalized PSD 0.5-50 Hz (95% CI)', fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = RESULTS / 'detailed_mean_power_spectrum.png'
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Detailed PSD figure saved to {out_path}")

def main():
    eeg, labels, meta = load_run()
    summarize_labels(labels, meta)
    epoch_std_distribution(eeg)
    mean_power_spectrum(eeg, labels, meta)
    print("Synthetic exploration complete ->", RESULTS)

if __name__ == "__main__":
    main()
