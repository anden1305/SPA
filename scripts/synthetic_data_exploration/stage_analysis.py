"""Synthetic sleep data stage analysis plots.
Generates selected plots:
- EEG band power analysis by sleep stage (absolute + relative)
- Detailed PSD with (optional) simple CI across epochs
- Sleep stage transition matrices
- Median sleep stage episode length
- Raw EEG excerpt plot (microVolt y-axis)

Usage (example):
python -m scripts.synthetic_data_exploration.stage_analysis --epochs 2000 --seed 42
"""
from __future__ import annotations
from pathlib import Path
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

from helpers import (ensure_output_directory, SYN_SLEEP_STAGE_COLORS, STAGE_ORDER, compute_epoch_psd, band_power)

def load_synthetic_dataset(prefix: str = "synthetic_default", data_dir: str | Path = Path("data") / "synthetic"):
    data_dir = Path(data_dir)
    eeg_path = data_dir / f"{prefix}_eeg.npy"
    labels_path = data_dir / f"{prefix}_labels.npy"
    meta_path = data_dir / f"{prefix}_metadata.json"
    if not eeg_path.exists():
        raise FileNotFoundError(f"Missing EEG file: {eeg_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels file: {labels_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {meta_path}")
    eeg = np.load(eeg_path)
    labels = np.load(labels_path)
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
    # Derive sampling info
    cfg = metadata.get('config_used', {})
    fs = int(cfg.get('sampling_rate_hz', 128))
    epoch_length_s = int(cfg.get('epoch_length_s', 4))
    stages = metadata.get('stages') or cfg.get('stages') or ["AWAKE", "NREM", "REM"]
    return {
        'eeg': eeg,
        'labels': labels,
        'metadata': metadata,
        'sampling_rate': fs,
        'epoch_length_s': epoch_length_s,
        'stages': stages,
    }


sns.set_palette('husl')

BANDS = [
    (0.5, 4.0, 'Delta'),
    (4.0, 8.0, 'Theta'),
    (8.0, 12.0, 'Alpha'),
    (12.0, 30.0, 'Beta'),
    (30.0, 50.0, 'Gamma'),
]

@dataclass
class EpochSpectralStats:
    stage: str
    band_powers: dict
    total_power: float
    freqs: np.ndarray
    psd: np.ndarray


def _compute_epoch_stats(eeg_epoch: np.ndarray, stage: str, fs: int) -> EpochSpectralStats:
    freqs, psd = compute_epoch_psd(eeg_epoch, fs)
    # restrict to 0.5-50
    mask = (freqs >= 0.5) & (freqs <= 50)
    freqs_m = freqs[mask]; psd_m = psd[mask]
    band_p = {}
    for lo, hi, name in BANDS:
        band_p[name] = band_power(freqs_m, psd_m, lo, hi)
    total = psd_m.sum()
    return EpochSpectralStats(stage, band_p, total, freqs_m, psd_m)


def _aggregate_by_stage(stats_list: list[EpochSpectralStats]):
    by_stage = {}
    for st in stats_list:
        by_stage.setdefault(st.stage, []).append(st)
    agg = {}
    for stage, lst in by_stage.items():
        bp_names = list(lst[0].band_powers.keys())
        band_matrix = np.array([[s.band_powers[n] for n in bp_names] for s in lst])
        totals = np.array([s.total_power for s in lst])
        rel_matrix = band_matrix / totals[:, None]
        agg[stage] = {
            'abs_mean': band_matrix.mean(axis=0),
            'abs_std': band_matrix.std(axis=0),
            'rel_mean': rel_matrix.mean(axis=0),
            'rel_std': rel_matrix.std(axis=0),
            'band_names': bp_names,
            'n_epochs': len(lst),
            'psds': np.stack([s.psd for s in lst]),
            'freqs': lst[0].freqs,
            'episode_lengths': _episode_lengths([st.stage for st in lst]),
        }
    return agg


def _episode_lengths(stage_sequence: list[str]):
    lengths = []
    if not stage_sequence:
        return lengths
    curr = stage_sequence[0]; count = 1
    for s in stage_sequence[1:]:
        if s == curr:
            count += 1
        else:
            lengths.append((curr, count))
            curr = s; count = 1
    lengths.append((curr, count))
    return lengths


def plot_band_powers(agg, out_dir: Path):
    stages = [s for s in STAGE_ORDER if s in agg]
    band_names = agg[stages[0]]['band_names']
    # Absolute
    fig, ax = plt.subplots(figsize=(10,6))
    width = 0.12
    x = np.arange(len(band_names))
    for i, stage in enumerate(stages):
        m = agg[stage]['abs_mean']; std = agg[stage]['abs_std']
        ax.bar(x + i*width, m, width, yerr=std, label=stage, color=SYN_SLEEP_STAGE_COLORS[stage], alpha=0.8)
    ax.set_xticks(x + width*(len(stages)-1)/2)
    ax.set_xticklabels(band_names)
    ax.set_ylabel('Power (a.u.)')
    ax.set_title('Absolute Band Power by Stage')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout(); fig.savefig(out_dir / 'band_power_absolute.png', dpi=300)

    # Relative
    fig, ax = plt.subplots(figsize=(10,6))
    for i, stage in enumerate(stages):
        m = agg[stage]['rel_mean']; std = agg[stage]['rel_std']
        ax.bar(x + i*width, m*100, width, yerr=std*100, label=stage, color=SYN_SLEEP_STAGE_COLORS[stage], alpha=0.8)
    ax.set_xticks(x + width*(len(stages)-1)/2)
    ax.set_xticklabels(band_names)
    ax.set_ylabel('Relative Power (%)')
    ax.set_title('Relative Band Power by Stage')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout(); fig.savefig(out_dir / 'band_power_relative.png', dpi=300)


def plot_detailed_psd(agg, out_dir: Path, overlay: bool = True):
    """Plot detailed PSD per stage.

    If overlay=True, all stage mean±1SD PSD curves are drawn on a single set of axes
    (requested behavior). Otherwise falls back to separate subplots (not used now).
    """
    stages = [s for s in STAGE_ORDER if s in agg]
    if overlay:
        fig, ax = plt.subplots(figsize=(12, 6))
        for stage in stages:
            freqs = agg[stage]['freqs']; psds = agg[stage]['psds']
            mean_psd = psds.mean(axis=0)
            std_psd = psds.std(axis=0)
            color = SYN_SLEEP_STAGE_COLORS[stage]
            ax.plot(freqs, mean_psd, color=color, label=stage, linewidth=1.4)
            ax.fill_between(freqs, np.maximum(mean_psd-std_psd, 0), mean_psd+std_psd,
                            color=color, alpha=0.15, linewidth=0)
        ax.set_xlim(0.5, 50)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD (a.u.)')
        ax.set_title('Detailed Power Spectral Density by Stage (0.5–50 Hz)')
        ax.grid(alpha=0.3)
        ax.legend(framealpha=0.9, title='Stage')
        fig.tight_layout()
        fig.savefig(out_dir / 'detailed_psd.png', dpi=300)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()
        for idx, stage in enumerate(stages):
            freqs = agg[stage]['freqs']; psds = agg[stage]['psds']
            mean_psd = psds.mean(axis=0)
            std_psd = psds.std(axis=0)
            ax = axes[idx]
            ax.plot(freqs, mean_psd, color=SYN_SLEEP_STAGE_COLORS[stage], label=stage)
            ax.fill_between(freqs, np.maximum(mean_psd-std_psd,0), mean_psd+std_psd, color=SYN_SLEEP_STAGE_COLORS[stage], alpha=0.2)
            ax.set_title(f'{stage} PSD (mean±1SD)')
            ax.set_xlabel('Hz'); ax.set_ylabel('PSD (a.u.)')
            ax.grid(alpha=0.3)
        fig.suptitle('Detailed Power Spectral Density by Stage (0.5-50 Hz)')
        fig.tight_layout(rect=[0,0,1,0.97])
        fig.savefig(out_dir / 'detailed_psd.png', dpi=300)


def plot_transition_matrix(labels: np.ndarray, stages: list[str], out_dir: Path):
    # labels already numeric sequence aligned with given stages order
    stage_to_idx = {s:i for i,s in enumerate(stages)}
    n = len(stages)
    counts = np.zeros((n,n), dtype=int)
    for a,b in zip(labels[:-1], labels[1:]):
        counts[a,b] += 1
    probs = counts / counts.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(probs, cmap='Blues', vmin=0, vmax=probs.max())
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(stages); ax.set_yticklabels(stages)
    ax.set_xlabel('To'); ax.set_ylabel('From')
    ax.set_title('Sleep Stage Transition Matrix')
    for i in range(n):
        for j in range(n):
            ax.text(j,i,f'{probs[i,j]:.2f}', ha='center', va='center', color='white' if probs[i,j]>0.5 else 'black', fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.75, label='Probability')
    fig.tight_layout(); fig.savefig(out_dir / 'transition_matrix.png', dpi=300)


def plot_episode_lengths(labels: np.ndarray, stages: list[str], out_dir: Path, epoch_len_s: int):
    # compute consecutive runs
    lengths = {s: [] for s in stages}
    current = labels[0]; run = 1
    for l in labels[1:]:
        if l == current:
            run += 1
        else:
            lengths[stages[current]].append(run)
            current = l; run = 1
    lengths[stages[current]].append(run)
    medians = {s: np.median(lengths[s])*epoch_len_s for s in stages if lengths[s]}
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(list(medians.keys()), list(medians.values()), color=[SYN_SLEEP_STAGE_COLORS[s] for s in medians.keys()])
    ax.set_ylabel('Median Episode Length (s)')
    ax.set_title('Median Sleep Stage Episode Length')
    for x,y in medians.items():
        ax.text(x,y+0.5,f'{y:.1f}s', ha='center', va='bottom')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / 'median_episode_length.png', dpi=300)


def plot_raw_eeg(eeg: np.ndarray, fs: int, stages: list[str], labels: np.ndarray, out_dir: Path,
                 seconds: int = 40, show_labels: bool = True):
    """Plot a raw EEG excerpt with color-coded sleep stage backgrounds.

    Parameters
    ----------
    eeg : np.ndarray
        Flattened 1-D EEG signal (concatenated epochs).
    fs : int
        Sampling frequency in Hz.
    stages : list[str]
        Mapping from numeric label to stage name.
    labels : np.ndarray
        1-D array of integer labels (one per epoch).
    out_dir : Path
        Directory to write the figure.
    seconds : int, default 40
        Duration in seconds from start to display.
    show_labels : bool, default True
        Whether to print stage names on the colored spans.
    """
    samples = seconds * fs
    sig = eeg[:samples]
    time = np.arange(samples) / fs
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(time, sig, linewidth=0.8, color='black')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (µV)')
    ax.set_title(f'Raw Synthetic EEG Excerpt ({seconds}s)')

    ep_samples = len(eeg) // len(labels) if len(labels) else samples
    epoch_len_s = ep_samples / fs if ep_samples else 0

    for i, st in enumerate(labels):
        start_sample = i * ep_samples
        start_t = start_sample / fs
        if start_t >= seconds:
            break
        end_t = min((start_sample + ep_samples) / fs, seconds)
        stage_name = stages[st]
        color = SYN_SLEEP_STAGE_COLORS.get(stage_name, '#cccccc')
        ax.axvspan(start_t, end_t, color=color, alpha=0.18, lw=0)
        if show_labels and epoch_len_s:
            if (end_t - start_t) >= max(0.6, 0.35 * epoch_len_s):
                ax.text((start_t + end_t) / 2, 0.95, stage_name,
                        ha='center', va='top', color='black', fontsize=9,
                        transform=ax.get_xaxis_transform())

    shown_epochs = int(np.ceil(seconds / epoch_len_s)) if epoch_len_s else 0
    present_stage_indices = np.unique(labels[:shown_epochs]) if shown_epochs else []
    handles = []
    for idx in present_stage_indices:
        sn = stages[idx]
        handles.append(plt.Line2D([0], [0], color=SYN_SLEEP_STAGE_COLORS.get(sn, '#ccc'), lw=6, label=sn))
    if handles:
        ax.legend(handles=handles, loc='upper right', framealpha=0.85, title='Stage')

    ax.set_xlim(0, seconds)
    ax.grid(alpha=0.25, axis='y')
    fig.tight_layout()
    fig.savefig(out_dir / 'raw_eeg_excerpt.png', dpi=300)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str, default='synthetic_default', help='Prefix of pre-generated synthetic dataset in data/synthetic')
    parser.add_argument('--data-dir', type=str, default='data/synthetic', help='Directory containing synthetic dataset files')
    parser.add_argument('--output', type=str, default='results/synthetic_exploration')
    args = parser.parse_args()

    out_dir = ensure_output_directory(args.output)

    ds = load_synthetic_dataset(prefix=args.prefix, data_dir=args.data_dir)
    raw_eeg = ds['eeg']
    labels = ds['labels']
    fs = ds['sampling_rate']
    stages = ds['stages']
    epoch_length_s = ds['epoch_length_s']

    # Ensure EEG is shaped (n_epochs, samples_per_epoch)
    ep_samples = epoch_length_s * fs
    if raw_eeg.ndim == 1:
        total_epochs_possible = len(raw_eeg) // ep_samples
        if total_epochs_possible == 0:
            raise ValueError("Continuous EEG length shorter than one epoch.")
        # Align with provided labels length (use min to avoid mismatch)
        n_epochs = min(len(labels), total_epochs_possible)
        eeg_epochs = raw_eeg[: n_epochs * ep_samples].reshape(n_epochs, ep_samples)
        if n_epochs < len(labels):
            labels = labels[:n_epochs]
        elif n_epochs > len(labels):
            # Trim extra epochs if labels shorter
            eeg_epochs = eeg_epochs[:len(labels)]
            n_epochs = len(labels)
    elif raw_eeg.ndim == 2:
        eeg_epochs = raw_eeg
        # Adjust epochs to match labels if mismatch
        if eeg_epochs.shape[0] != len(labels):
            n_epochs = min(eeg_epochs.shape[0], len(labels))
            eeg_epochs = eeg_epochs[:n_epochs]
            labels = labels[:n_epochs]
    else:
        raise ValueError(f"Unsupported EEG array shape {raw_eeg.shape}; expected 1-D or 2-D")

    # Compute spectral stats per epoch over ALL epochs available
    stats = []
    for ep_sig, lab in zip(eeg_epochs, labels):
        stage = stages[lab]
        stats.append(_compute_epoch_stats(ep_sig, stage, fs))
    agg = _aggregate_by_stage(stats)

    # Plots
    plot_band_powers(agg, out_dir)
    plot_detailed_psd(agg, out_dir)
    plot_transition_matrix(labels, stages, out_dir)
    plot_episode_lengths(labels, stages, out_dir, epoch_length_s)
    flat = eeg_epochs.reshape(-1)
    plot_raw_eeg(flat, fs, stages, labels, out_dir)

    # Save summary JSON
    summary = {}
    for st, val in agg.items():
        summary[st] = {
            'abs_band_power_mean': {bn: float(m) for bn, m in zip(val['band_names'], val['abs_mean'])},
            'rel_band_power_mean': {bn: float(m) for bn, m in zip(val['band_names'], val['rel_mean'])},
            'n_epochs': val['n_epochs']
        }
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'Synthetic stage analysis complete (loaded prefix "{args.prefix}"). Results in {out_dir}')

if __name__ == '__main__':
    main()
