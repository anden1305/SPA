"""Synthetic data exploration (v2).

Usage:
    python -m scripts.synthetic_data_exploration_2.synthetic_data_exploration_v2 <relative_path_to_synthetic_folder>

Argument:
    <relative_path_to_synthetic_folder>: Folder containing eeg.npy, labels.npy, and optional config_copy.yml or metadata.

Replicates plots from scripts/synthetic_data_exploration/synthetic_data_exploration.py but
instead of relying on a fixed naming convention it loads directly from the provided folder.

Outputs saved to results/synthetic_exploration.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Iterable, Optional, Sequence, Tuple
import json


SYN_SLEEP_STAGE_COLORS = {
    'AWAKE': '#FF6B6B',
    'NREM': '#4ECDC4',
    'REM': '#45B7D1',
}

BASE_RESULTS = Path("results") / "synthetic_exploration"
RESULTS = BASE_RESULTS  # will be refined per dataset in main()

# ---------------------- Core Loading ----------------------

def load_folder(folder: Path):
    eeg_path = folder / "eeg.npy"
    labels_path = folder / "labels.npy"
    if not eeg_path.exists():
        raise FileNotFoundError(f"Missing eeg.npy at {eeg_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels.npy at {labels_path}")
    eeg = np.load(eeg_path)
    labels = np.load(labels_path)
    # Try to detect sampling rate from config_copy.yml
    meta = {}
    cfg_path = folder / "config_copy.yml"
    if cfg_path.exists():
        import yaml
        try:
            cfg = yaml.safe_load(cfg_path.read_text()) or {}
            meta["sampling_rate_hz"] = cfg.get("sampling_rate_hz", cfg.get("sampling_rate", 1))
            # Optional stage names if user added stage_names
            if "stage_names" in cfg:
                meta["stages"] = cfg["stage_names"]
        except Exception:
            pass
    # Reshape EEG if it's 1D continuous to (epochs, samples)
    if eeg.ndim == 1:
        # Derive epochs from labels length
        n_epochs = len(labels)
        if n_epochs == 0:
            raise ValueError("Labels empty; can't reshape EEG")
        # Need sampling rate & epoch length to reshape: attempt from config
        cfg_epoch_len = None
        if cfg_path.exists():
            try:
                import yaml
                cfg = yaml.safe_load(cfg_path.read_text()) or {}
                epoch_len = int(cfg.get("epoch_length_s", 30))
                sr = int(cfg.get("sampling_rate_hz", cfg.get("sampling_rate", 1)))
                samples_per_epoch = sr * epoch_len
                expected = samples_per_epoch * n_epochs
                if expected != eeg.shape[0]:
                    raise ValueError(
                        f"Continuous EEG length {eeg.shape[0]} does not match n_epochs*epoch_len*sr ({expected})"
                    )
                eeg = eeg.reshape(n_epochs, samples_per_epoch)
            except Exception as e:
                raise ValueError("Failed to reshape 1D EEG into epochs. Provide per-epoch array.") from e
    return eeg, labels, meta

# ---------------------- Label Summary ----------------------

def summarize_labels(labels: np.ndarray, meta: dict):
    stages = meta.get("stages", [])
    unique, counts = np.unique(labels, return_counts=True)
    mapping = {u: stages[u] if u < len(stages) else str(u) for u in unique}
    summary_lines = [
        f"{u} ({mapping[u]}): {c} ({c/len(labels):.2%})" for u, c in zip(unique, counts)
    ]
    (RESULTS / "label_summary.txt").write_text("\n".join(summary_lines))
    # Plot
    plt.figure(figsize=(5,3))
    import pandas as pd
    df_counts = pd.DataFrame({"stage": [mapping[u] for u in unique], "count": counts})
    sns.barplot(data=df_counts, x="stage", y="count", hue="stage", palette="viridis", legend=False)
    plt.title("Stage counts")
    plt.ylabel("Epochs")
    plt.tight_layout()
    plt.savefig(RESULTS / "stage_counts.png", dpi=150)
    plt.close()
    # Transition matrix if stage names length known
    n_stages = len(stages) if stages else unique.max() + 1
    tm = np.zeros((n_stages, n_stages))
    for a,b in zip(labels[:-1], labels[1:]):
        tm[a,b] += 1
    row_sums = tm.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        tm_prob = np.where(row_sums>0, tm/row_sums, 0)
    plt.figure(figsize=(4,4))
    tick_labels = stages if stages else [str(i) for i in range(n_stages)]
    sns.heatmap(tm_prob, annot=True, fmt=".2f", cmap="magma", xticklabels=tick_labels, yticklabels=tick_labels)
    plt.title("Transition matrix (P(next|current))")
    plt.tight_layout()
    plt.savefig(RESULTS / "transition_matrix.png", dpi=150)
    plt.close()

# ---------------------- Epoch Std Distribution ----------------------

def epoch_std_distribution(eeg: np.ndarray):
    stds = eeg.std(axis=1)
    unique_vals = np.unique(stds)
    plt.figure(figsize=(5,3))
    if len(unique_vals) <= 1:
        val = float(unique_vals[0]) if unique_vals.size else 0.0
        plt.axvline(val, color="steelblue")
        plt.text(val, 0.5, f"std={val:.3g}", rotation=90, va="center")
        plt.xlim(val - 1 if val != 0 else -1, val + 1 if val != 0 else 1)
        plt.ylim(0,1)
    else:
        n_bins = min(30, len(unique_vals))
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

# ---------------------- Raw EEG Excerpt ----------------------

def raw_eeg_excerpt(eeg: np.ndarray, labels: np.ndarray, meta: dict, seconds: int = 40, show_labels: bool = True):
    """Exact-style raw EEG excerpt (port of original plot_raw_eeg logic).

    Accepts either (epochs, samples) EEG or 1-D continuous. Determines epoch samples
    from total length / n_labels. Colors spans by stage with legend.
    """
    fs = int(meta.get('sampling_rate_hz', 128))
    if fs <= 0: fs = 1
    # Flatten if 2D
    if eeg.ndim == 2:
        flat = eeg.reshape(-1)
    else:
        flat = eeg
    if len(labels) == 0:
        print('No labels; skipping raw excerpt')
        return
    # Derive ep_samples robustly
    ep_samples = len(flat) // len(labels)
    if ep_samples == 0:
        print('Epoch sample calculation failed; skipping raw excerpt')
        return
    # Use provided stage names or numeric indices
    stages = meta.get('stages') or [str(i) for i in range(int(labels.max())+1)]
    samples_to_show = min(int(seconds * fs), len(flat))
    actual_seconds = samples_to_show / fs
    sig = flat[:samples_to_show]
    time = np.arange(samples_to_show) / fs
    fig, ax = plt.subplots(figsize=(14,4))
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Amplitude (µV)')
    ax.set_title(f'Raw Synthetic EEG Excerpt ({actual_seconds:.1f}s)')
    epoch_len_s = ep_samples / fs
    # Plot each epoch segment in its stage color (and optional translucent background span)
    # Build a color map: if provided stage names absent from predefined, assign palette colors
    defined_colors = SYN_SLEEP_STAGE_COLORS
    # If any stage not in predefined map, create new palette mapping all stages
    if any((s not in defined_colors) for s in stages):
        palette = sns.color_palette('tab10', n_colors=len(stages))
        stage_color_map = {s: palette[i] for i, s in enumerate(stages)}
    else:
        stage_color_map = {s: defined_colors[s] for s in stages if s in defined_colors}

    for i, st in enumerate(labels):
        start_sample = i * ep_samples
        start_t = start_sample / fs
        if start_t >= actual_seconds:
            break
        end_sample = start_sample + ep_samples
        end_t = min(end_sample / fs, actual_seconds)
        stage_name = stages[st] if st < len(stages) else str(st)
        color = stage_color_map.get(stage_name, '#666666')
        # Background span
        ax.axvspan(start_t, end_t, color=color, alpha=0.10, lw=0)
        # Line segment (clip to available samples)
        seg_end_sample = min(end_sample, samples_to_show)
        seg = flat[start_sample:seg_end_sample]
        seg_time = np.arange(start_sample, start_sample + len(seg)) / fs
        ax.plot(seg_time, seg, color=color, linewidth=0.9)
        if show_labels and epoch_len_s:
            if (end_t - start_t) >= max(0.6, 0.35 * epoch_len_s):
                ax.text((start_t + end_t)/2, 0.95, stage_name, ha='center', va='top', color='black', fontsize=9,
                        transform=ax.get_xaxis_transform())
    shown_epochs = int(np.ceil(actual_seconds / epoch_len_s)) if epoch_len_s else 0
    present_stage_indices = np.unique(labels[:shown_epochs]) if shown_epochs else []
    handles = []
    for idx in present_stage_indices:
        sn = stages[idx] if idx < len(stages) else str(idx)
        handles.append(plt.Line2D([0],[0], color=stage_color_map.get(sn, '#ccc'), lw=6, label=sn))
    if handles:
        ax.legend(handles=handles, loc='upper right', framealpha=0.85, title='Stage')
    ax.set_xlim(0, actual_seconds)
    ax.grid(alpha=0.25, axis='y')
    fig.tight_layout()
    fig.savefig(RESULTS / 'raw_eeg_excerpt.png', dpi=300)
    plt.close(fig)

# ---------------------- PSD Helpers ----------------------

def _compute_epoch_psd(eeg: np.ndarray, sr: float) -> Tuple[np.ndarray, np.ndarray]:
    n = eeg.shape[1]
    freqs = np.fft.rfftfreq(n, d=1 / sr)
    fft_vals = np.fft.rfft(eeg, axis=1)
    psd = (np.abs(fft_vals) ** 2) / n
    return freqs, psd


def _spectral_confidence(psds: Iterable[np.ndarray], alpha: float = 0.05) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    psds_arr = np.asarray(psds)
    if psds_arr.ndim == 1:
        psds_arr = psds_arr[None, :]
    mean = psds_arr.mean(axis=0)
    n = psds_arr.shape[0]
    if n < 2:
        return mean, None, None
    std = psds_arr.std(axis=0, ddof=1)
    from math import sqrt
    z = 1.96
    half_width = z * std / sqrt(n)
    return mean, np.clip(mean - half_width, a_min=0, a_max=None), mean + half_width


def mean_power_spectrum(eeg: np.ndarray, labels: np.ndarray, meta: dict):
    sr = meta.get("sampling_rate_hz", meta.get("config_used", {}).get("sampling_rate_hz", 1))
    stages = meta.get("stages", [])
    if eeg.ndim != 2:
        raise ValueError("Expected eeg shape (epochs, samples)")
    freqs, all_psd = _compute_epoch_psd(eeg, sr)
    stage_psds = {}
    if stages:
        for idx, stage in enumerate(stages):
            mask = labels == idx
            if mask.any():
                stage_psds[stage] = all_psd[mask]
    else:
        # Fallback: derive stage indices only
        unique = np.unique(labels)
        for u in unique:
            mask = labels == u
            stage_psds[str(u)] = all_psd[mask]
        stages = list(stage_psds.keys())
    if not stage_psds:
        print("No stage PSD data found; skipping detailed PSD plot")
        return
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Detailed Power Spectral Density by Sleep Stage (with 95% CI)', fontsize=16, fontweight='bold')
    ordered_stages = [s for s in stages if s in stage_psds]
    palette = sns.color_palette("tab10", n_colors=len(ordered_stages))
    colors = {s: c for s, c in zip(ordered_stages, palette)}
    def _mask_range(lo, hi):
        return (freqs >= lo) & (freqs <= hi)
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
    ax = axes[1, 1]
    norm_psds = {}
    for stage in ordered_stages:
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

# ---------------------- Main ----------------------

def main(argv: Sequence[str] | None = None):
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) != 1:
        print("Usage: python -m scripts.synthetic_data_exploration_2.synthetic_data_exploration_v2 <relative_synthetic_folder>")
        return 1
    folder = Path(argv[0])
    if not folder.exists():
        print(f"Provided folder does not exist: {folder}")
        return 1
    eeg, labels, meta = load_folder(folder)
    # Derive run name from folder (e.g., data/synthetic_data/demo_basic -> demo_basic)
    run_name = folder.name
    global RESULTS
    RESULTS = BASE_RESULTS / run_name
    RESULTS.mkdir(parents=True, exist_ok=True)
    # Save a small meta summary
    summary_path = RESULTS / "run_info.txt"
    summary_lines = [
        f"source_folder: {folder}",
        f"run_name: {run_name}",
        f"sampling_rate_hz: {meta.get('sampling_rate_hz','?')}",
        f"n_epochs: {len(labels)}",
        f"epoch_length_s: {meta.get('epoch_length_s', meta.get('config_used',{}).get('epoch_length_s','?'))}",
    ]
    summary_path.write_text("\n".join(summary_lines))
    summarize_labels(labels, meta)
    epoch_std_distribution(eeg)
    raw_eeg_excerpt(eeg, labels, meta, seconds=40)
    mean_power_spectrum(eeg, labels, meta)
    print("Synthetic exploration v2 complete ->", RESULTS)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
