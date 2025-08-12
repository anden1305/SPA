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

from scr.synthetic.generator import SyntheticSleepGenerator, DEFAULT_CONFIG
from .helpers import (
    ensure_output_directory, SYN_SLEEP_STAGE_COLORS, STAGE_ORDER, compute_epoch_psd, band_power
)

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


def plot_detailed_psd(agg, out_dir: Path):
    stages = [s for s in STAGE_ORDER if s in agg]
    fig, axes = plt.subplots(2,2, figsize=(14,10))
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


def plot_raw_eeg(eeg: np.ndarray, fs: int, stages: list[str], labels: np.ndarray, out_dir: Path, seconds: int = 40):
    samples = seconds * fs
    sig = eeg[:samples]
    time = np.arange(samples)/fs
    fig, ax = plt.subplots(figsize=(14,4))
    ax.plot(time, sig, linewidth=0.8, color='black')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Amplitude (µV)')
    ax.set_title(f'Raw Synthetic EEG Excerpt ({seconds}s)')
    # overlay stage boundaries
    ep_samples = len(eeg)//len(labels)
    for i, st in enumerate(labels):
        start = i*ep_samples
        if start/ fs > seconds: break
        ax.axvspan(start/fs, min((start+ep_samples)/fs, seconds), color=SYN_SLEEP_STAGE_COLORS[stages[st]], alpha=0.05)
    fig.tight_layout(); fig.savefig(out_dir / 'raw_eeg_excerpt.png', dpi=300)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='results/synthetic_exploration')
    parser.add_argument('--save-data', action='store_true')
    args = parser.parse_args()

    out_dir = ensure_output_directory(args.output)

    gen = SyntheticSleepGenerator(DEFAULT_CONFIG, seed=args.seed)
    data = gen.generate(epochs=args.epochs, save=args.save_data, prefix='synthetic_analysis')
    eeg = data['eeg']  # shape (epochs, samples)
    labels = data['labels']

    fs = gen.sampling_rate
    # Flatten into epoch list stats
    stats = []
    for ep_sig, lab in zip(eeg, labels):
        stage = gen.stages[lab]
        stats.append(_compute_epoch_stats(ep_sig, stage, fs))
    agg = _aggregate_by_stage(stats)

    # Plots
    plot_band_powers(agg, out_dir)
    plot_detailed_psd(agg, out_dir)
    plot_transition_matrix(labels, gen.stages, out_dir)
    plot_episode_lengths(labels, gen.stages, out_dir, gen.epoch_length_s)
    # Raw EEG (use first epoch_length*some epochs) flatten
    flat = eeg.reshape(-1)
    plot_raw_eeg(flat, fs, gen.stages, labels, out_dir)

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

    print(f'Synthetic stage analysis complete. Results in {out_dir}')

if __name__ == '__main__':
    main()
