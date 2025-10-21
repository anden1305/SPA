"""
Produce a small set of data exploration figures and save them to results/data_exploration_report.

Figures produced:
1) Example raw EEG and EMG over 12 epochs with multiple labels.
2) Detailed PSD of EEG by sleep stage.
3) Band power (delta/theta/alpha/beta/gamma) by sleep stage.
4) Absolute EMG power by sleep stage.

This script reuses helpers from scripts/data_exploration/helpers.py
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from scipy import signal

# Ensure scripts package imports
import sys
THIS_DIR = Path(__file__).parent
if str(THIS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(THIS_DIR.parent))

from data_exploration.helpers import (
    load_participant_data,
    SLEEP_STAGE_MAPPING,
    SLEEP_STAGE_COLORS,
    STAGE_ORDER,
    calculate_spectral_confidence_intervals,
)

# Output folder for saved figures
OUTPUT_DIR = Path('results/data_exploration_report')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# local numpy/pandas aliases used in this script
import pandas as pd
_pd = pd
import numpy as _np
 
def figure_raw_signals_example(participant_id, run_id, save_path: Path, save_only_idx=None):
    """Plot example raw EEG and EMG snippets with epoch stage overlays.

    Produces one or more candidate figures (appends _1/_2/_3 to save_path) so the
    user can pick the clearest example. This function expects the helper
    `load_participant_data` to provide 'labels' and EEG/EMG channels.
    """
    data = load_participant_data(participant_id, run_id)
    labels = data['labels']
    fs = 128

    # defaults for example window search and plotting
    epoch_seconds = 30
    epochs_to_plot = 12
    samples_to_plot = epochs_to_plot * epoch_seconds * fs

    # helper: find candidate windows (defined here so it can access epoch_seconds/fs)
    def _find_best_windows(labels_array, epochs_to_plot, top_k=3):
        """Return up to top_k window start indices (in samples) ranked by unique stage count.

        labels_array is expected at sample resolution; we downsample to epoch indices.
        """
        epoch_samples = epoch_seconds * fs
        if len(labels_array) >= epoch_samples:
            n_epochs_labels = len(labels_array) // epoch_samples
            labels_epoch = labels_array[:n_epochs_labels * epoch_samples].reshape(n_epochs_labels, epoch_samples)[:, 0]
        else:
            labels_epoch = labels_array

        L = max(1, len(labels_epoch) - epochs_to_plot + 1)
        scores = []
        # prefer windows that contain Awake, NREM, REM and exclude Artifact
        desired_stage_names = {'Awake', 'NREM', 'REM'}
        artifact_name = 'Artifact'
        for i in range(L):
            segment = labels_epoch[i:i+epochs_to_plot]
            # map numeric labels to stage names
            stage_names = [SLEEP_STAGE_MAPPING.get(int(x), 'Unknown') for x in segment]
            counts = {}
            for sname in stage_names:
                counts[sname] = counts.get(sname, 0) + 1
            dom_frac = max(counts.values()) / sum(counts.values()) if sum(counts.values())>0 else 1.0

            contains_desired = desired_stage_names.issubset(set(stage_names))
            contains_artifact = (artifact_name in set(stage_names))

            # scoring tuple: (contains desired (1/0), not contains artifact (1/0), unique count, diversity)
            score = (1 if contains_desired else 0, 0 if contains_artifact else 1, len(set(stage_names)), 1.0 - dom_frac)
            scores.append((score, i))

        # sort by score descending (lexicographic gives priority to desired & artifact filter)
        scores_sorted = sorted(scores, key=lambda x: x[0], reverse=True)
        starts = [i for (_, i) in scores_sorted[:top_k]]
        # convert epoch index to sample index
        return [s * epoch_samples for s in starts]

    # find up to 3 candidate windows (start samples)
    epoch_samples = epoch_seconds * fs
    candidate_starts = _find_best_windows(labels, epochs_to_plot, top_k=3)
    # if fewer candidates were found, ensure at least one
    if len(candidate_starts) == 0:
        candidate_starts = [0]

    # Choose EEG channel early so we can clamp to the available signal length
    eeg_channel = None
    for ch in ['EEG1', 'EEG2', 'EEG3', 'EEG4']:
        if ch in data:
            eeg_channel = ch
            break
    if eeg_channel is None:
        raise FileNotFoundError("No EEG channel found for example plot")
    # Display name: collapse EEG1/EEG2/... to 'EEG' for titles/legends
    display_eeg_name = 'EEG' if eeg_channel and eeg_channel.upper().startswith('EEG') else eeg_channel

    # For convenience, this function now writes multiple example files: append an index to save_path
    # and produce up to len(candidate_starts) figures so the user can pick the clearest one.
    for idx, start_sample in enumerate(candidate_starts, start=1):
        # Ensure we don't request samples beyond available data
        available_len = min(len(labels), len(data[eeg_channel]), len(data['EMG'])) if 'EMG' in data else min(len(labels), len(data[eeg_channel]))
        desired_end = start_sample + samples_to_plot
        if desired_end > available_len:
            # reduce number of epochs to what is available
            max_epochs_available = max(1, (available_len - start_sample) // epoch_samples)
            samples_to_plot_adj = max_epochs_available * epoch_samples
        else:
            samples_to_plot_adj = samples_to_plot
        end_sample = start_sample + samples_to_plot_adj

        fig, axes = plt.subplots(3, 1, figsize=(15, 8), sharex=True)

        eeg = data[eeg_channel][start_sample:end_sample]
        emg = data['EMG'][start_sample:end_sample] if 'EMG' in data else np.zeros_like(eeg)
        labels_seg = labels[start_sample:end_sample]

        times = np.arange(len(eeg)) / fs / 60.0  # minutes

        # Use user-requested signal colors: EEG = #4C72B0, EMG = #DD8452
        axes[0].plot(times, eeg, color='#4C72B0', linewidth=0.6)
        axes[0].set_ylabel(f'{display_eeg_name} (uV)')
        # Use the requested figure title
        fig.suptitle('Raw EEG and EMG Signals', fontsize=14, fontweight='bold', x=0.5, y=0.995)

        axes[1].plot(times, emg, color='#DD8452', linewidth=0.6)
        axes[1].set_ylabel('EMG (uV)')

        # label heatmap
        # convert labels to names and map to colors
        label_names = [SLEEP_STAGE_MAPPING.get(int(l), 'Unknown') for l in labels_seg]
        unique_labels = sorted(list(dict.fromkeys(label_names)), key=lambda x: STAGE_ORDER.index(x) if x in STAGE_ORDER else 999)

        # plot colored patches per epoch and overlay on EEG/EMG axes
        epoch_boundaries = np.arange(start_sample, end_sample+1, epoch_seconds * fs)
        # first overlay shading and annotations
        for i in range(len(epoch_boundaries)-1):
            s = epoch_boundaries[i]
            e = epoch_boundaries[i+1]
            name = SLEEP_STAGE_MAPPING.get(int(labels[s]), 'Unknown')
            color = SLEEP_STAGE_COLORS.get(name, '#cccccc')
            x0 = (s-start_sample)/fs/60.0
            x1 = (e-start_sample)/fs/60.0
            # overlay on EEG and EMG
            axes[0].axvspan(x0, x1, color=color, alpha=0.12)
            axes[1].axvspan(x0, x1, color=color, alpha=0.12)
            # annotate stage name in the top subplot at the center of epoch (use a stable y location)
            try:
                y_top = axes[0].get_ylim()[1]
            except Exception:
                y_top = None
            if y_top is not None:
                axes[0].text((x0+x1)/2, y_top*0.9, name, ha='center', va='top', fontsize=8, color='black')

        # then draw vertical separators between epochs; make transitions (different labels) more prominent
        boundary_times = (epoch_boundaries - start_sample) / fs / 60.0
        for j in range(1, len(boundary_times)-1):
            # boundary between epoch j-1 and j
            t = boundary_times[j]
            lbl_prev = SLEEP_STAGE_MAPPING.get(int(labels[epoch_boundaries[j-1]]), None)
            lbl_next = SLEEP_STAGE_MAPPING.get(int(labels[epoch_boundaries[j]]), None)
            if lbl_prev != lbl_next:
                lw = 1.6
                alpha = 0.9
                color = 'k'
            else:
                lw = 0.6
                alpha = 0.25
                color = 'k'
            axes[0].axvline(t, color=color, linewidth=lw, alpha=alpha)
            axes[1].axvline(t, color=color, linewidth=lw, alpha=alpha)

        # remove bottom stage axis (we use overlay + legend)
        axes[2].set_visible(False)

        # set exact x-limits so there's no padding before/after the signal
        if len(times) > 0:
            axes[0].set_xlim(times[0], times[-1])
            axes[1].set_xlim(times[0], times[-1])
            # remove any automatic x-margins
            axes[0].margins(x=0)
            axes[1].margins(x=0)

        # create horizontal legend centered under the suptitle and above the axes
        legend_labels = [s for s in STAGE_ORDER if s in unique_labels]
        handles = [Patch(facecolor=SLEEP_STAGE_COLORS.get(name, '#777777'), label=name) for name in legend_labels]
        if len(handles) > 0:
            # Subplots have their own legends; skip figure-level legend to avoid redundancy
            plt.tight_layout()
        outpath = save_path.with_name(save_path.stem + f'_{idx}' + save_path.suffix)
        # If save_only_idx is provided, only save the requested candidate (and skip others)
        if save_only_idx is None or idx == save_only_idx:
            plt.savefig(outpath, dpi=300, bbox_inches='tight')
        plt.close()


def figure_psd_by_stage(participant_id, run_id, save_path: Path):
    """Detailed PSD of EEG signal by sleep stage. We'll compute PSD segments per-stage and plot mean +- CI."""
    data = load_participant_data(participant_id, run_id)
    labels = data['labels']
    fs = 128

    eeg_channel = None
    for ch in ['EEG1', 'EEG2', 'EEG3', 'EEG4']:
        if ch in data:
            eeg_channel = ch
            break
    if eeg_channel is None:
        raise FileNotFoundError("No EEG channel found for PSD plot")

    eeg = data[eeg_channel]

    # For each stage, break signal into non-overlapping 30s segments and compute Welch PSD
    stage_psds = {name: [] for name in STAGE_ORDER}
    stage_freqs = None

    epoch_len = 30 * fs
    n_epochs = len(eeg) // epoch_len
    for e in range(n_epochs):
        s = e * epoch_len
        eidx = s + epoch_len
        seg = eeg[s:eidx]
        stage_label = SLEEP_STAGE_MAPPING.get(int(labels[s]), 'Unknown')
        if stage_label not in stage_psds:
            continue
        f, pxx = signal.welch(seg, fs=fs, nperseg=fs*4)
        stage_psds[stage_label].append(pxx)
        stage_freqs = f

    # Plot
    fig = plt.figure(figsize=(10, 6))
    for stage in STAGE_ORDER:
        psds = stage_psds.get(stage, [])
        if len(psds) == 0:
            continue
        mean_psd, ci_low, ci_up = calculate_spectral_confidence_intervals(psds)
        plt.semilogy(stage_freqs, mean_psd, label=stage, color=SLEEP_STAGE_COLORS.get(stage))
        plt.fill_between(stage_freqs, ci_low, ci_up, color=SLEEP_STAGE_COLORS.get(stage), alpha=0.2)

    # Restrict PSD display to 0-60 Hz as requested
    plt.xlim(0.0, 60)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (power)')
    plt.title('EEG Power Spectral Density by Sleep Stage')
    plt.legend()
    # Use constrained layout and then tight_layout for best spacing
    try:
        fig.set_constrained_layout_pads(hspace=0.02, wspace=0.02)
    except Exception:
        pass
    plt.tight_layout()
    # Save with bbox_inches='tight' so the legend placed to the right is included
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def figure_band_power_by_stage(participant_id, run_id, save_path: Path):
    """Compute band powers per epoch and plot grouped bar chart by stage."""
    data = load_participant_data(participant_id, run_id)
    labels = data['labels']
    fs = 128

    eeg_channel = None
    for ch in ['EEG1', 'EEG2', 'EEG3', 'EEG4']:
        if ch in data:
            eeg_channel = ch
            break
    if eeg_channel is None:
        raise FileNotFoundError("No EEG channel found for band power plot")

    eeg = data[eeg_channel]
    epoch_len = 30 * fs
    n_epochs = len(eeg) // epoch_len

    bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 12),
        'Beta': (12, 30),
        'Gamma': (30, 100),
    }

    import pandas as pd

    # accumulate per-stage band power
    import pandas as pd
    records = []
    for e in range(n_epochs):
        s = e * epoch_len
        eidx = s + epoch_len
        seg = eeg[s:eidx]
        label = SLEEP_STAGE_MAPPING.get(int(labels[s]), 'Unknown')
        f, pxx = signal.welch(seg, fs=fs, nperseg=fs*4)
        total = np.sum(pxx[(f>=0.5)&(f<=100)])
        rec = {'stage': label}
        for bname, (lo, hi) in bands.items():
            power = np.sum(pxx[(f>=lo)&(f<hi)])
            rec[bname] = power if total==0 else power / total
        records.append(rec)

    df = pd.DataFrame.from_records(records)

    # plot
    plt.figure(figsize=(10, 6))
    df_m = df.melt(id_vars='stage', value_vars=list(bands.keys()), var_name='band', value_name='relative_power')
    order = STAGE_ORDER
    sns.barplot(data=df_m, x='band', y='relative_power', hue='stage', hue_order=order, palette=[SLEEP_STAGE_COLORS.get(s) for s in order])
    plt.title('Relative Band Power by Sleep Stage')
    plt.ylabel('Relative Power')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def figure_emg_power_by_stage(participant_id, run_id, save_path: Path):
    """Compute absolute EMG power per epoch and plot by stage."""
    data = load_participant_data(participant_id, run_id)
    labels = data['labels']
    fs = 128

    if 'EMG' not in data:
        raise FileNotFoundError("No EMG channel for EMG power plot")

    emg = data['EMG']
    epoch_len = 30 * fs
    n_epochs = len(emg) // epoch_len

    import pandas as pd
    records = []
    for e in range(n_epochs):
        s = e * epoch_len
        eidx = s + epoch_len
        seg = emg[s:eidx]
        label = SLEEP_STAGE_MAPPING.get(int(labels[s]), 'Unknown')
        # absolute power as mean squared amplitude
        power = np.mean(seg.astype(float)**2)
        records.append({'stage': label, 'emg_power': power})

    df = pd.DataFrame.from_records(records)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x='stage', y='emg_power', order=STAGE_ORDER, palette=[SLEEP_STAGE_COLORS.get(s) for s in STAGE_ORDER])
    plt.title('Absolute EMG Power by Sleep Stage')
    plt.ylabel('EMG Power (mean squared amplitude)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def figure_eeg_power_by_stage(participant_id, run_id, save_path: Path):
    """Compute absolute EEG power per epoch and plot by stage (mean squared amplitude), similar to EMG."""
    data = load_participant_data(participant_id, run_id)
    labels = data['labels']
    fs = 128

    # choose first available EEG channel
    eeg_channel = None
    for ch in ['EEG1', 'EEG2', 'EEG3', 'EEG4']:
        if ch in data:
            eeg_channel = ch
            break
    if eeg_channel is None:
        raise FileNotFoundError("No EEG channel for EEG power plot")

    eeg = data[eeg_channel]
    epoch_len = 30 * fs
    n_epochs = len(eeg) // epoch_len

    import pandas as pd
    records = []
    for e in range(n_epochs):
        s = e * epoch_len
        eidx = s + epoch_len
        seg = eeg[s:eidx]
        label = SLEEP_STAGE_MAPPING.get(int(labels[s]), 'Unknown')
        # absolute power as mean squared amplitude
        power = np.mean(seg.astype(float)**2)
        records.append({'stage': label, 'eeg_power': power})

    df = pd.DataFrame.from_records(records)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x='stage', y='eeg_power', order=STAGE_ORDER, palette=[SLEEP_STAGE_COLORS.get(s) for s in STAGE_ORDER])
    plt.title(f'Absolute {eeg_channel} Power by Sleep Stage')
    plt.ylabel('EEG Power (mean squared amplitude)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def figure_eeg_band_power_by_stage(participant_id, run_id, save_path: Path):
    """Compute absolute band powers (delta/theta/alpha/beta/gamma) per 30s epoch for EEG and plot by sleep stage."""
    data = load_participant_data(participant_id, run_id)
    labels = data['labels']
    fs = 128

    # choose first available EEG channel
    eeg_channel = None
    for ch in ['EEG1', 'EEG2', 'EEG3', 'EEG4']:
        if ch in data:
            eeg_channel = ch
            break
    if eeg_channel is None:
        raise FileNotFoundError("No EEG channel for EEG band power plot")

    eeg = data[eeg_channel]
    epoch_len = 30 * fs
    n_epochs = len(eeg) // epoch_len

    bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 12),
        'Beta': (12, 30),
        'Gamma': (30, 100),
    }

    import pandas as pd
    records = []
    for e in range(n_epochs):
        s = e * epoch_len
        eidx = s + epoch_len
        seg = eeg[s:eidx]
        label = SLEEP_STAGE_MAPPING.get(int(labels[s]), 'Unknown')
        f, pxx = signal.welch(seg, fs=fs, nperseg=fs*4)
        rec = {'stage': label}
        for bname, (lo, hi) in bands.items():
            # absolute band power (sum of PSD in band)
            power = np.sum(pxx[(f>=lo)&(f<hi)])
            rec[bname] = power
        records.append(rec)

    df = pd.DataFrame.from_records(records)

    # plot absolute band powers grouped by stage (show mean + CI via seaborn barplot default error bars)
    plt.figure(figsize=(10, 6))
    df_m = df.melt(id_vars='stage', value_vars=list(bands.keys()), var_name='band', value_name='power')
    order = STAGE_ORDER
    sns.barplot(data=df_m, x='band', y='power', hue='stage', hue_order=order, palette=[SLEEP_STAGE_COLORS.get(s) for s in order])
    plt.title(f'EEG Absolute Band Power by Sleep Stage ({eeg_channel})')
    plt.ylabel('Power (absolute)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def figure_eeg_band_power_ratios_by_stage(participant_id, run_id, save_path: Path):
    """Compute EEG band power ratios per epoch and plot them by sleep stage.

    Ratios computed: Delta/Theta, Delta/Alpha, Theta/Alpha, Beta/Alpha
    (these are simple examples; users can extend as needed).
    """
    data = load_participant_data(participant_id, run_id)
    labels = data['labels']
    fs = 128

    # choose first available EEG channel
    eeg_channel = None
    for ch in ['EEG1', 'EEG2', 'EEG3', 'EEG4']:
        if ch in data:
            eeg_channel = ch
            break
    if eeg_channel is None:
        raise FileNotFoundError("No EEG channel for EEG band-power-ratio plot")

    eeg = data[eeg_channel]
    epoch_len = 30 * fs
    n_epochs = len(eeg) // epoch_len

    # define sigma candidates and a small helper localized to this function
    base_bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 12),
        'Beta': (12, 30),
        'Gamma': (30, 100),
    }
    sigma_candidates = [(10, 15), (11, 16), (12, 15), (9, 18)]

    def compute_df_for_sigma_local(sigma_range):
        bands_local = dict(base_bands)
        bands_local['Sigma'] = sigma_range
        records_local = []
        for e in range(n_epochs):
            s = e * epoch_len
            eidx = s + epoch_len
            seg = eeg[s:eidx]
            label = SLEEP_STAGE_MAPPING.get(int(labels[s]), 'Unknown')
            f, pxx = signal.welch(seg, fs=fs, nperseg=fs*4)
            band_powers = {}
            for bname, (lo, hi) in bands_local.items():
                power = np.sum(pxx[(f>=lo)&(f<hi)])
                band_powers[bname] = power
            eps = 1e-12
            rec = {'stage': label}
            rec['Theta/Delta'] = band_powers.get('Theta', 0.0) / (band_powers.get('Delta', 0.0) + eps)
            rec['Alpha/Theta'] = band_powers.get('Alpha', 0.0) / (band_powers.get('Theta', 0.0) + eps)
            rec['Sigma/Delta'] = band_powers.get('Sigma', 0.0) / (band_powers.get('Delta', 0.0) + eps)
            rec['Beta/Delta'] = band_powers.get('Beta', 0.0) / (band_powers.get('Delta', 0.0) + eps)
            records_local.append(rec)
        import pandas as pd
        return pd.DataFrame.from_records(records_local)

    df = compute_df_for_sigma_local(sigma_candidates[0])
    thresh = 1e-6
    if df['Sigma/Delta'].abs().sum() < thresh:
        found = False
        for alt in sigma_candidates[1:]:
            df_alt = compute_df_for_sigma_local(alt)
            if df_alt['Sigma/Delta'].abs().sum() >= thresh:
                print(f"Sigma band {sigma_candidates[0]} appeared empty — using fallback Sigma={alt}")
                df = df_alt
                found = True
                break
        if not found:
            alt = (9, 18)
            df_alt = compute_df_for_sigma_local(alt)
            if df_alt['Sigma/Delta'].abs().sum() >= thresh:
                print(f"Using broader Sigma={alt} as fallback")
                df = df_alt
            else:
                print("Warning: Sigma band power appears to be essentially zero for all tested ranges. Sigma/Delta will be NaN.")
                df['Sigma/Delta'] = np.nan

    # Only include the two requested ratios: Theta/Delta and Beta/Delta
    df_m = df.melt(id_vars='stage', value_vars=['Theta/Delta', 'Beta/Delta'], var_name='ratio', value_name='value')

    plt.figure(figsize=(10, 6))
    order = STAGE_ORDER
    sns.barplot(data=df_m, x='ratio', y='value', hue='stage', hue_order=order, palette=[SLEEP_STAGE_COLORS.get(s) for s in order])
    plt.title(f'EEG Band Power Ratios by Sleep Stage ({eeg_channel})')
    plt.ylabel('Ratio (unitless)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def figure_lab_comparison_2x2(metadata_csv: str, save_path: Path, *, max_subjects_per_lab: int | None = None, max_epochs_psd: int = 8, nperseg_factor: int = 2):
    """Create a 2x2 comparison figure across labs.

    Panels:
      TL: Stage composition per lab (stacked bar, subject-averaged fractions)
      TR: NREM PSD overlay per lab (0-60 Hz) with 95% CI
      BL: Relative band power (NREM) grouped by lab (bands as x, lab hue)
      BR: Transition matrix difference between top two labs (labA - labB)
    """
    import pandas as pd
    from collections import defaultdict

    meta = pd.read_csv(metadata_csv)
    # Map raw lab keys to nicer display names like 'Lab 1' (local mapping)
    raw_labs = sorted(meta['lab'].unique())
    lab_display_map = {}
    import re
    for lab in raw_labs:
        if isinstance(lab, str) and lab.lower().startswith('lab_'):
            try:
                n = lab.split('_')[-1]
                lab_display_map[lab] = f'Lab {int(n)}'
                continue
            except Exception:
                pass
        m = re.search(r'(\d+)', str(lab))
        if m:
            lab_display_map[lab] = f'Lab {int(m.group(1))}'
        else:
            lab_display_map[lab] = str(lab).replace('_', ' ').title()

    lab_colors = {
        'Lab 1': '#9467BD',
        'Lab 2': '#17BECF',
        'Lab 3': '#D62728',
        'Lab 4': '#8C564B',
        'Lab 5': '#7F7F7F',
    }
    # Map raw lab keys to nicer display names like 'Lab 1' (local mapping) early so
    # we can use it when aggregating/plotting.
    raw_labs = sorted(meta['lab'].unique())
    lab_display_map = {}
    import re
    for lab in raw_labs:
        if isinstance(lab, str) and lab.lower().startswith('lab_'):
            try:
                n = lab.split('_')[-1]
                lab_display_map[lab] = f'Lab {int(n)}'
                continue
            except Exception:
                pass
        m = re.search(r'(\d+)', str(lab))
        if m:
            lab_display_map[lab] = f'Lab {int(m.group(1))}'
        else:
            lab_display_map[lab] = str(lab).replace('_', ' ').title()

    lab_colors = {
        'Lab 1': '#9467BD',
        'Lab 2': '#17BECF',
        'Lab 3': '#D62728',
        'Lab 4': '#8C564B',
        'Lab 5': '#7F7F7F',
    }

    # recreate lab display mapping and colors locally
    raw_labs = sorted(meta['lab'].unique())
    lab_display_map = {}
    for lab in raw_labs:
        if isinstance(lab, str) and lab.lower().startswith('lab_'):
            try:
                n = lab.split('_')[-1]
                lab_display_map[lab] = f'Lab {int(n)}'
                continue
            except Exception:
                pass
        import re
        m = re.search(r'(\d+)', str(lab))
        if m:
            lab_display_map[lab] = f'Lab {int(m.group(1))}'
        else:
            lab_display_map[lab] = str(lab).replace('_', ' ').title()

    lab_colors = {
        'Lab 1': '#9467BD',
        'Lab 2': '#17BECF',
        'Lab 3': '#D62728',
        'Lab 4': '#8C564B',
        'Lab 5': '#7F7F7F',
    }

    # recreate lab_display_map and lab_colors used elsewhere
    raw_labs = sorted(meta['lab'].unique())
    lab_display_map = {}
    for lab in raw_labs:
        if isinstance(lab, str) and lab.lower().startswith('lab_'):
            try:
                n = lab.split('_')[-1]
                lab_display_map[lab] = f'Lab {int(n)}'
                continue
            except Exception:
                pass
        import re
        m = re.search(r'(\d+)', str(lab))
        if m:
            lab_display_map[lab] = f'Lab {int(m.group(1))}'
        else:
            lab_display_map[lab] = str(lab).replace('_', ' ').title()

    lab_colors = {
        'Lab 1': '#9467BD',
        'Lab 2': '#17BECF',
        'Lab 3': '#D62728',
        'Lab 4': '#8C564B',
        'Lab 5': '#7F7F7F',
    }

    # collect per-subject summaries
    subj_stage_frac = []  # rows: subject, lab, stage, fraction
    subj_band_rel = []    # rows: subject, lab, stage, band, rel_power
    subj_psd = []         # rows: subject, lab, f, psd (only NREM)
    subj_trans = []       # rows: subject, lab, from, to, p

    # limit bands to <=30 Hz per request
    bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 12),
        'Beta': (12, 30),
    }

    # track how many subjects we've processed per lab (to allow quick debugging runs)
    seen_per_lab = defaultdict(int)
    for _, row in meta.iterrows():
        pid = row['participant_id']
        run = int(row['run'])
        lab = row.get('lab', 'unknown')
        if max_subjects_per_lab is not None and seen_per_lab[lab] >= max_subjects_per_lab:
            continue
        try:
            data = load_participant_data(pid, run)
        except Exception as e:
            print(f"Skipping {pid} run {run}: {e}")
            continue

        labels = data['labels']
        fs = 128
        epoch_len = 30 * fs
        # determine number of epochs
        n_epochs = len(labels) // epoch_len
        if n_epochs == 0:
            continue

        # per-epoch stage names
        epoch_stages = []
        for e in range(n_epochs):
            s = e * epoch_len
            stage_name = SLEEP_STAGE_MAPPING.get(int(labels[s]), 'Unknown')
            epoch_stages.append(stage_name)

        # stage fractions (subject-level)
        unique, counts = np.unique(epoch_stages, return_counts=True)
        frac = {u: c / float(len(epoch_stages)) for u, c in zip(unique, counts)}
        for st in STAGE_ORDER:
            subj_stage_frac.append({'subject': pid, 'lab': lab, 'stage': st, 'fraction': frac.get(st, 0.0)})

    # band relative power per epoch (compute per-epoch then average across epochs of stage NREM)
        # choose EEG channel
        eeg_channel = None
        for ch in ['EEG1','EEG2','EEG3','EEG4']:
            if ch in data:
                eeg_channel = ch
                break
        if eeg_channel is None:
            continue
        # Apply bandpass filter removing frequencies above 30 Hz (and below 0.5 Hz)
        eeg = np.asarray(data[eeg_channel], dtype=float)
        try:
            nyq = 128.0 / 2.0
            low = 0.5 / nyq
            high = 30.0 / nyq
            b, a = signal.butter(4, [low, high], btype='band')
            # filtfilt may fail on very short signals; guard with try/except
            eeg = signal.filtfilt(b, a, eeg)
        except Exception:
            # if filtering fails, fall back to original signal
            eeg = np.asarray(data[eeg_channel], dtype=float)

        epoch_band_rel = []
        epoch_psds = defaultdict(list)
        # collect indices of NREM epochs for optional sampling
        nrem_epoch_indices = []
        for e in range(n_epochs):
            s = e * epoch_len
            eidx = s + epoch_len
            seg = eeg[s:eidx]
            # compute PSD with a shorter window when needed for speed and for short segments
            nperseg = min(fs * nperseg_factor, len(seg))
            try:
                f, pxx = signal.welch(seg, fs=fs, nperseg=nperseg)
            except Exception:
                # fallback to a safe small nperseg
                f, pxx = signal.welch(seg, fs=fs, nperseg=min(256, len(seg)))
            # total power 0.5-60
            total = np.sum(pxx[(f>=0.5)&(f<=60)])
            bandvals = {}
            for bname, (lo, hi) in bands.items():
                p = np.sum(pxx[(f>=lo)&(f<hi)])
                bandvals[bname] = p if total==0 else p / total
            stage_name = epoch_stages[e]
            # store per-epoch band rel
            for b, v in bandvals.items():
                subj_band_rel.append({'subject': pid, 'lab': lab, 'stage': stage_name, 'band': b, 'rel_power': v})

            # collect PSD for NREM stage only (but limit how many epochs used per subject)
            if stage_name == 'NREM':
                nrem_epoch_indices.append((s, eidx, f, pxx))

        # sample up to max_epochs_psd NREM epochs for PSD to avoid long runtimes
        if len(nrem_epoch_indices) > 0:
            import random
            sample_idx = list(range(len(nrem_epoch_indices)))
            if max_epochs_psd is not None and len(sample_idx) > max_epochs_psd:
                sample_idx = random.sample(sample_idx, max_epochs_psd)
            for ii in sample_idx:
                s, eidx, f, pxx = nrem_epoch_indices[ii]
                mask = f <= 60
                subj_psd.append({'subject': pid, 'lab': lab, 'f': f[mask], 'psd': pxx[mask]})

        # transition matrix per subject
        # build name->index map
        name_to_idx = {name: i for i, name in enumerate(STAGE_ORDER)}
        counts = np.zeros((len(STAGE_ORDER), len(STAGE_ORDER)), dtype=float)
        for a, b in zip(epoch_stages[:-1], epoch_stages[1:]):
            if a not in name_to_idx or b not in name_to_idx:
                continue
            if a == b:
                continue
            i = name_to_idx[a]; j = name_to_idx[b]
            counts[i, j] += 1.0
        probs = np.zeros_like(counts)
        for i in range(counts.shape[0]):
            s = counts[i].sum()
            probs[i] = counts[i] / s if s > 0 else 0.0
        # append flattened
        for i, a in enumerate(STAGE_ORDER):
            for j, bname in enumerate(STAGE_ORDER):
                subj_trans.append({'subject': pid, 'lab': lab, 'from': a, 'to': bname, 'p': probs[i, j]})

    df_stage = pd.DataFrame(subj_stage_frac)
    df_bands = pd.DataFrame(subj_band_rel)
    # PSD needs special treatment: stack per-frequency by subject
    # convert subj_psd list of dicts to DataFrame by interpolating to common freq grid
    if len(subj_psd) == 0:
        print('No PSD data collected (no NREM epochs found)')
        return
    # build common freq grid (limit to 30 Hz)
    fgrid = subj_psd[0]['f']
    fgrid = fgrid[fgrid <= 30]
    psd_rows = []
    for rec in subj_psd:
        # interpolate onto fgrid (clip to <=30 Hz)
        psd_interp = np.interp(fgrid, rec['f'], rec['psd'])
        psd_rows.append({'subject': rec['subject'], 'lab': rec['lab'], 'f': fgrid, 'psd': psd_interp})
    # expand into dataframe with columns f and psd per row
    rows = []
    for r in psd_rows:
        for fi, pi in zip(r['f'], r['psd']):
            rows.append({'subject': r['subject'], 'lab': r['lab'], 'f': fi, 'psd': pi})
    df_psd = pd.DataFrame(rows)

    df_trans = pd.DataFrame(subj_trans)

    # Map raw lab keys to nicer display names like 'Lab 1'
    raw_labs = sorted(meta['lab'].unique())
    lab_display_map = {}
    for lab in raw_labs:
        # try pattern 'lab_#'
        if isinstance(lab, str) and lab.lower().startswith('lab_'):
            try:
                n = lab.split('_')[-1]
                lab_display_map[lab] = f'Lab {int(n)}'
                continue
            except Exception:
                pass
        # fallback: try to extract trailing digit
        import re
        m = re.search(r'(\d+)', str(lab))
        if m:
            lab_display_map[lab] = f'Lab {int(m.group(1))}'
        else:
            # title-case fallback
            lab_display_map[lab] = str(lab).replace('_', ' ').title()

    # add display columns to dataframes
    df_stage['lab_display'] = df_stage['lab'].map(lab_display_map)
    df_bands['lab_display'] = df_bands['lab'].map(lab_display_map)
    df_psd['lab_display'] = df_psd['lab'].map(lab_display_map)
    df_trans['lab_display'] = df_trans['lab'].map(lab_display_map)

    # --- Plot ---
    fig, axs = plt.subplots(2,2, figsize=(14,10))
    ax0 = axs[0,0]; ax1 = axs[0,1]; ax2 = axs[1,0]; ax3 = axs[1,1]

    # Top-left: stage composition per lab (subject-level mean)
    order = STAGE_ORDER
    # use displayed lab names for x-axis (e.g., 'Lab 1')
    labs_disp = [d for d in sorted(df_stage['lab_display'].unique())]
    # keep raw lab keys for filtering PSD/band data
    labs_raw = [d for d in sorted(df_stage['lab'].unique())]
    # pivot to lab_display x stage mean
    pivot = df_stage.groupby(['lab_display','stage'])['fraction'].mean().unstack(fill_value=0).reindex(index=labs_disp, columns=order)
    pivot.plot(kind='bar', stacked=True, color=[SLEEP_STAGE_COLORS[s] for s in order], ax=ax0)
    ax0.set_title('Stage composition per lab (mean across subjects)')
    ax0.set_ylabel('Fraction of epochs')
    ax0.legend(title='Stage')

    # Lab colors (user-specified)
    # Colors keyed by display names
    lab_colors = {
        'Lab 1': '#9467BD',
        'Lab 2': '#17BECF',
        'Lab 3': '#D62728',
        'Lab 4': '#8C564B',
        'Lab 5': '#7F7F7F',
    }

    # Top-right: Transition matrix difference (Lab 3 - Lab 5), moved to top-right
    unique_disp_labs = sorted(df_trans['lab_display'].unique())
    labA_disp = 'Lab 3'
    labB_disp = 'Lab 5'
    if labA_disp in unique_disp_labs and labB_disp in unique_disp_labs:
        # Exclude the 'Artifact' stage from the transition matrices
        order_no_art = [s for s in order if s != 'Artifact']
        mA = df_trans[df_trans['lab_display']==labA_disp].query("`from` != 'Artifact' and `to` != 'Artifact'").groupby(['from','to'])['p'].mean().unstack(fill_value=0).reindex(index=order_no_art, columns=order_no_art)
        mB = df_trans[df_trans['lab_display']==labB_disp].query("`from` != 'Artifact' and `to` != 'Artifact'").groupby(['from','to'])['p'].mean().unstack(fill_value=0).reindex(index=order_no_art, columns=order_no_art)
        diff = mA - mB
        sns.heatmap(diff, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax1, xticklabels=order_no_art, yticklabels=order_no_art)
        ax1.set_title(f'Transition matrix difference: {labA_disp} - {labB_disp} (excluding Artifact)')
    else:
        ax1.text(0.5, 0.5, 'Lab 3 and/or Lab 5 not present', ha='center', va='center')

    # Bottom-left: relative band power by stage (NREM only) showing 95% CI
    df_bands_nrem = df_bands[df_bands['stage']=='NREM']
    if not df_bands_nrem.empty:
        df_bands_nrem['lab_display'] = df_bands_nrem['lab_display'].astype(str)
        lab_order = sorted(df_bands_nrem['lab_display'].unique())
        lab_palette = {disp: lab_colors.get(disp, '#777777') for disp in lab_order}
        sns.barplot(data=df_bands_nrem, x='band', y='rel_power', hue='lab_display', ax=ax2, palette=lab_palette, ci=95, capsize=0.08)
        ax2.set_title('Relative band power (NREM) by lab (95% CI)')
        ax2.set_ylabel('Relative power')

    # Bottom-right: Relative band power (REM) by lab showing 95% CI
    df_bands_rem = df_bands[df_bands['stage']=='REM']
    if not df_bands_rem.empty:
        df_bands_rem['lab_display'] = df_bands_rem['lab_display'].astype(str)
        lab_order_rem = sorted(df_bands_rem['lab_display'].unique())
        lab_palette_rem = {disp: lab_colors.get(disp, '#777777') for disp in lab_order_rem}
        sns.barplot(data=df_bands_rem, x='band', y='rel_power', hue='lab_display', ax=ax3, palette=lab_palette_rem, ci=95, capsize=0.08)
        ax3.set_title('Relative band power (REM) by lab (95% CI)')
        ax3.set_ylabel('Relative power')
    else:
        ax3.text(0.5, 0.5, 'No REM epochs found', ha='center', va='center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def figure_inter_individual_variability(metadata_csv: str, save_path: Path, *, epoch_seconds: int = 4, fs: int = 128, bands: dict | None = None, max_subjects_per_lab: int | None = None, bootstrap_rounds: int = 200, lab_include: str | None = None):
    """Produce a 2x2 inter-individual variability figure as requested by the user.

    Panels implemented:
      TL: %NREM vs %REM per mouse (point size = %Artifact), color = lab
      TR: Median bout length per stage (boxplots + per-mouse points)
      BL: REM theta/delta ratio per mouse (z-scored within lab) with bootstrap 95% CI
      BR: Transition probabilities P(NREM->REM) vs P(REM->NREM) per mouse

    Returns dict with subject-level summary DataFrames for inspection.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import signal
    from collections import defaultdict

    # Use a clean white style for publication-ready figures and adjust fonts
    sns.set(style='white')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
    })

    # frequency bands for spectral ratio
    theta_band = (6.0, 9.0)
    delta_band = (0.5, 4.0)
    total_band = (0.5, 45.0)

    meta = pd.read_csv(metadata_csv)
    # Optional: filter to only a single raw lab key (e.g., 'lab_3') when requested
    if lab_include is not None:
        if 'lab' in meta.columns:
            meta = meta[meta['lab'] == lab_include].copy()
        else:
            # nothing to filter
            pass

    # helper to map raw lab -> display
    import re
    raw_labs_local = sorted(meta['lab'].dropna().unique()) if 'lab' in meta.columns else []
    lab_display_map = {}
    for lab in raw_labs_local:
        if isinstance(lab, str) and lab.lower().startswith('lab_'):
            try:
                n = lab.split('_')[-1]
                lab_display_map[lab] = f'Lab {int(n)}'
                continue
            except Exception:
                pass
        m = re.search(r'(\d+)', str(lab))
        if m:
            lab_display_map[lab] = f'Lab {int(m.group(1))}'
        else:
            lab_display_map[lab] = str(lab).replace('_', ' ').title()

    # aggregate per-subject records
    subj_records = {}
    seen_per_lab = defaultdict(int)

    for _, row in meta.iterrows():
        pid = row['participant_id']
        run = int(row['run'])
        lab = row.get('lab', 'unknown')
        if max_subjects_per_lab is not None and seen_per_lab[lab] >= max_subjects_per_lab:
            continue
        try:
            data = load_participant_data(pid, run)
        except Exception as e:
            # skip missing or unreadable runs
            continue

        # pick first EEG channel
        eeg_ch = None
        for ch in ['EEG1','EEG2','EEG3','EEG4']:
            if ch in data:
                eeg_ch = ch
                break
        if eeg_ch is None:
            continue

        # normalize per-run
        eeg = np.asarray(data[eeg_ch], dtype=float)
        eeg = (eeg - np.nanmean(eeg)) / (np.nanstd(eeg) + 1e-12)

        labels = np.asarray(data.get('labels', []), dtype=int)

        # epochize
        nper = int(epoch_seconds * fs)
        n_epochs = len(eeg) // nper
        if n_epochs == 0:
            continue
        segs = eeg[:n_epochs * nper].reshape(n_epochs, nper)
        epoch_labels = labels[:n_epochs * nper].reshape(n_epochs, nper)[:,0] if labels.size >= n_epochs * nper else np.array(['Unknown']*n_epochs)
        # map numeric labels to stage names
        stage_names = [SLEEP_STAGE_MAPPING.get(int(x), 'Unknown') if (not (isinstance(x, str) and x=='Unknown')) else 'Unknown' for x in epoch_labels]

        # compute per-epoch PSD for spectral measures (REM ratios)
        f, pxx = signal.welch(segs, fs=fs, nperseg=min(fs*4, nper), axis=1)
        # mask frequencies
        tot_mask = (f >= total_band[0]) & (f <= total_band[1])
        th_mask = (f >= theta_band[0]) & (f < theta_band[1])
        dl_mask = (f >= delta_band[0]) & (f < delta_band[1])

        # per-epoch band totals
        tot_power = pxx[:, tot_mask].sum(axis=1) + 1e-12
        theta_power = pxx[:, th_mask].sum(axis=1)
        delta_power = pxx[:, dl_mask].sum(axis=1)

        # store or append to subject records keyed by pid
        if pid not in subj_records:
            subj_records[pid] = {
                'lab': lab,
                'runs': [],
                'stages': [],
                'theta_power': [],
                'delta_power': [],
                'tot_power': [],
            }
        subj = subj_records[pid]
        subj['runs'].append(run)
        subj['stages'].append(stage_names)
        subj['theta_power'].append(theta_power)
        subj['delta_power'].append(delta_power)
        subj['tot_power'].append(tot_power)
        seen_per_lab[lab] += 1

    # Build per-subject summary DataFrame
    rows = []
    for pid, rec in subj_records.items():
        lab = rec['lab']
        # concatenate runs
        stages = np.concatenate(rec['stages']) if len(rec['stages'])>0 else np.array([])
        theta = np.concatenate(rec['theta_power']) if len(rec['theta_power'])>0 else np.array([])
        delta = np.concatenate(rec['delta_power']) if len(rec['delta_power'])>0 else np.array([])
        tot = np.concatenate(rec['tot_power']) if len(rec['tot_power'])>0 else np.array([])

        # artifact-free mask
        mask_good = np.array([s not in ('Artifact','Unknown') for s in stages])
        stages_good = np.array(stages)[mask_good]
        theta_good = theta[mask_good]
        delta_good = delta[mask_good]
        tot_good = tot[mask_good]

        # per-stage proportions (excluding Artifact by default)
        counts = {st: np.sum(stages_good==st) for st in STAGE_ORDER if st!='Artifact'}
        total_nonartifact = sum(counts.values())
        pct = {st: (counts.get(st,0)/total_nonartifact if total_nonartifact>0 else np.nan) for st in ['Awake','NREM','REM']}
        pct_artifact = 1.0 - (total_nonartifact / max(1, len(stages)))

        # average bout lengths per stage (in seconds) - compute mean bout length per subject
        def mean_bout_length(st_name):
            bouts = []
            seq = stages
            if len(seq) == 0:
                return np.nan
            cur = None
            cur_len = 0
            for s in seq:
                if s == 'Artifact' or s == 'Unknown':
                    if cur == st_name and cur_len > 0:
                        bouts.append(cur_len)
                    cur = None
                    cur_len = 0
                    continue
                if s == cur:
                    cur_len += 1
                else:
                    if cur == st_name and cur_len > 0:
                        bouts.append(cur_len)
                    cur = s
                    cur_len = 1
            if cur == st_name and cur_len > 0:
                bouts.append(cur_len)
            if len(bouts) == 0:
                return np.nan
            return np.mean(bouts) * epoch_seconds

        med_awake = mean_bout_length('Awake')
        med_nrem = mean_bout_length('NREM')
        med_rem = mean_bout_length('REM')

        # compute relative theta per epoch using only theta and delta bands (ignore other bands)
        # relative_theta = theta / (theta + delta)
        rem_mask = np.array(stages) == 'REM'
        eps = 1e-12
        if np.sum(rem_mask) > 0:
            theta_p = theta[rem_mask]
            delta_p = delta[rem_mask]
            rel_theta_epochs = theta_p / (theta_p + delta_p + eps)
            rem_rel_theta = np.nanmean(rel_theta_epochs)
        else:
            rem_rel_theta = np.nan

        # compute relative theta for non-REM epochs (Awake + NREM)
        nonrem_mask = np.array([s in ('Awake','NREM') for s in stages])
        if np.sum(nonrem_mask) > 0:
            theta_p_non = theta[nonrem_mask]
            delta_p_non = delta[nonrem_mask]
            rel_theta_non_epochs = theta_p_non / (theta_p_non + delta_p_non + eps)
            other_rel_theta = np.nanmean(rel_theta_non_epochs)
        else:
            other_rel_theta = np.nan

        # compute relative theta for NREM epochs only (for stacked bar)
        nrem_mask = np.array(stages) == 'NREM'
        if np.sum(nrem_mask) > 0:
            theta_p_nrem = theta[nrem_mask]
            delta_p_nrem = delta[nrem_mask]
            rel_theta_nrem_epochs = theta_p_nrem / (theta_p_nrem + delta_p_nrem + eps)
            nrem_rel_theta = np.nanmean(rel_theta_nrem_epochs)
        else:
            nrem_rel_theta = np.nan

        # compute theta/delta ratio for REM and for NREM (for stacked theta/delta bars)
        if np.sum(rem_mask) > 0:
            theta_p = theta[rem_mask]
            delta_p = delta[rem_mask]
            rem_theta_delta = np.nanmean(theta_p / (delta_p + eps))
        else:
            rem_theta_delta = np.nan

        if np.sum(nrem_mask) > 0:
            theta_p_nrem = theta[nrem_mask]
            delta_p_nrem = delta[nrem_mask]
            nrem_theta_delta = np.nanmean(theta_p_nrem / (delta_p_nrem + eps))
        else:
            nrem_theta_delta = np.nan

        # transitions (first-order) excluding self-transitions
        seq_all = stages
        tr_counts = defaultdict(int)
        tr_from_counts = defaultdict(int)
        for i in range(len(seq_all)-1):
            a = seq_all[i]
            b = seq_all[i+1]
            if a in ('Artifact','Unknown') or b in ('Artifact','Unknown'):
                continue
            if a == b:
                continue
            tr_counts[(a,b)] += 1
            tr_from_counts[a] += 1
        p_nrem_to_rem = tr_counts.get(('NREM','REM'), 0) / tr_from_counts.get('NREM', 1) if tr_from_counts.get('NREM',0)>0 else np.nan
        p_rem_to_nrem = tr_counts.get(('REM','NREM'), 0) / tr_from_counts.get('REM', 1) if tr_from_counts.get('REM',0)>0 else np.nan
        p_rem_to_awake = tr_counts.get(('REM','Awake'), 0) / tr_from_counts.get('REM', 1) if tr_from_counts.get('REM',0)>0 else np.nan

        rows.append({
            'subject': pid,
            'lab': lab,
            'lab_display': lab_display_map.get(lab, str(lab)),
            'pct_NREM': pct['NREM'],
            'pct_REM': pct['REM'],
            'pct_Awake': pct['Awake'],
            'pct_Artifact': pct_artifact,
            'avg_bout_awake_s': med_awake,
            'avg_bout_nrem_s': med_nrem,
            'avg_bout_rem_s': med_rem,
            # per-subject REM relative theta (theta / (theta + delta)) ignoring other bands
            'rem_rel_theta': rem_rel_theta,
            'other_rel_theta': other_rel_theta,
            'nrem_rel_theta': nrem_rel_theta,
            'rem_theta_delta': rem_theta_delta,
            'nrem_theta_delta': nrem_theta_delta,
            'rem_minus_other': (rem_rel_theta - other_rel_theta) if (not np.isnan(rem_rel_theta) and not np.isnan(other_rel_theta)) else np.nan,
            'p_nrem_to_rem': p_nrem_to_rem,
            'p_rem_to_nrem': p_rem_to_nrem,
            'p_rem_to_awake': p_rem_to_awake,
        })

    df_subj = pd.DataFrame.from_records(rows)
    if df_subj.empty:
        print('No subject-level data to plot')
        return {}

    # Recompute rem/non-REM relative theta per subject explicitly from subj_records (robust)
    def compute_rem_nonrem(pid):
        rec = subj_records.get(pid)
        if rec is None:
            return np.nan, np.nan, np.nan, np.nan, np.nan
        stages_arr = np.concatenate(rec['stages']) if len(rec['stages'])>0 else np.array([])
        theta_arr = np.concatenate(rec['theta_power']) if len(rec['theta_power'])>0 else np.array([])
        delta_arr = np.concatenate(rec['delta_power']) if len(rec['delta_power'])>0 else np.array([])
        n = min(len(stages_arr), len(theta_arr), len(delta_arr))
        if n == 0:
            return np.nan, np.nan, np.nan, np.nan, np.nan
        stages_a = np.array(stages_arr[:n])
        theta_a = theta_arr[:n]
        delta_a = delta_arr[:n]
        eps = 1e-12
        rem_mask = stages_a == 'REM'
        nonrem_mask = np.isin(stages_a, ['Awake','NREM'])
        rem_rel = np.nan
        nonrem_rel = np.nan
        nrem_rel = np.nan
        rem_ratio = np.nan
        nrem_ratio = np.nan
        if np.sum(rem_mask) > 0:
            theta_rel_rem = theta_a[rem_mask]
            delta_rel_rem = delta_a[rem_mask]
            rem_rel_epochs = theta_rel_rem / (theta_rel_rem + delta_rel_rem + eps)
            rem_rel = np.nanmean(rem_rel_epochs)
            # theta/delta ratio for REM
            rem_ratio = np.nanmean(theta_rel_rem / (delta_rel_rem + eps))
        if np.sum(nonrem_mask) > 0:
            theta_rel_non = theta_a[nonrem_mask]
            delta_rel_non = delta_a[nonrem_mask]
            nonrem_rel_epochs = theta_rel_non / (theta_rel_non + delta_rel_non + eps)
            nonrem_rel = np.nanmean(nonrem_rel_epochs)
        # NREM-only
        nrem_mask = stages_a == 'NREM'
        if np.sum(nrem_mask) > 0:
            theta_rel_nrem = theta_a[nrem_mask]
            delta_rel_nrem = delta_a[nrem_mask]
            nrem_rel_epochs = theta_rel_nrem / (theta_rel_nrem + delta_rel_nrem + eps)
            nrem_rel = np.nanmean(nrem_rel_epochs)
            # theta/delta ratio for NREM
            nrem_ratio = np.nanmean(theta_rel_nrem / (delta_rel_nrem + eps))
        diff = rem_rel - nonrem_rel if (not np.isnan(rem_rel) and not np.isnan(nonrem_rel)) else np.nan
        return rem_rel, nonrem_rel, diff, nrem_rel, rem_ratio, nrem_ratio

    rem_list = []
    nonrem_list = []
    diff_list = []
    nrem_list = []
    rem_ratio_list = []
    nrem_ratio_list = []
    for pid in df_subj['subject'].tolist():
        r, nr, d, nrem, rratio, nratio = compute_rem_nonrem(pid)
        rem_list.append(r)
        nonrem_list.append(nr)
        diff_list.append(d)
        nrem_list.append(nrem)
        rem_ratio_list.append(rratio)
        nrem_ratio_list.append(nratio)
    # store recomputed relative-theta values and theta/delta ratios
    df_subj['rem_rel_theta_recomputed'] = rem_list
    df_subj['other_rel_theta_recomputed'] = nonrem_list
    df_subj['nrem_rel_theta_recomputed'] = nrem_list
    df_subj['rem_minus_other'] = diff_list
    df_subj['rem_theta_delta_recomputed'] = rem_ratio_list
    df_subj['nrem_theta_delta_recomputed'] = nrem_ratio_list
    # z-score relative theta within lab for optional display later
    df_subj['rem_rel_theta_z'] = df_subj.groupby('lab')['rem_rel_theta_recomputed'].transform(lambda x: (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12))

    # plotting - create panels per user's updated spec
    fig, axs = plt.subplots(2,2, figsize=(14,10))
    ax1 = axs[0,0]  # TL: REM/(REM+NREM) bar
    ax2 = axs[0,1]  # TR: boxplot avg bout length per stage
    ax3 = axs[1,0]  # BL: (REM θ/δ) - (other θ/δ) bar
    ax4 = axs[1,1]  # BR: transition probs scatter

    # lab color scheme (reuse lab comparison palette)
    lab_colors = {
        'Lab 1': '#9467BD',
        'Lab 2': '#17BECF',
        'Lab 3': '#D62728',
        'Lab 4': '#8C564B',
        'Lab 5': '#7F7F7F',
    }
    labs = sorted(df_subj['lab_display'].unique())
    # build a palette dict only for labs present
    palette = {ld: lab_colors.get(ld, '#777777') for ld in labs}

    # 1) REM/(REM+NREM) per subject as bar, sorted
    df_subj['rem_over_remplusnrem'] = df_subj.apply(lambda r: (r['pct_REM'] / (r['pct_REM'] + r['pct_NREM'])) if ((r['pct_REM'] + r['pct_NREM'])>0) else np.nan, axis=1)
    df_p1 = df_subj[['subject','lab_display','rem_over_remplusnrem']].dropna().copy()
    df_p1 = df_p1.sort_values('rem_over_remplusnrem', ascending=False).reset_index(drop=True)
    colors_p1 = [palette.get(x, '#777777') for x in df_p1['lab_display']]
    ax1.bar(range(len(df_p1)), df_p1['rem_over_remplusnrem'], color=colors_p1, edgecolor='none')
    ax1.set_xticks(range(len(df_p1)))
    ax1.set_xticklabels(df_p1['subject'], rotation=90, fontsize=7, ha='center')
    ax1.set_ylabel('REM / (REM + NREM)')
    if not df_p1.empty:
        vmin = max(0.0, float(df_p1['rem_over_remplusnrem'].min()) - 0.03)
        vmax = min(1.0, float(df_p1['rem_over_remplusnrem'].max()) + 0.03)
        ax1.set_ylim(vmin, vmax)
    ax1.set_title('REM proportion of sleep')
    ax1.grid(False)
    sns.despine(ax=ax1)

    # 2) Boxplot over average time stayed in each stage (before transitioning) per individual
    # For each subject we computed median bout lengths; we now plot distributions of these per-subject averages
    df_bouts = df_subj[['subject','lab_display','avg_bout_awake_s','avg_bout_nrem_s','avg_bout_rem_s']].copy()
    df_bouts_long = df_bouts.melt(id_vars=['subject','lab_display'], value_vars=['avg_bout_awake_s','avg_bout_nrem_s','avg_bout_rem_s'], var_name='metric', value_name='med_bout_s')
    df_bouts_long['stage'] = df_bouts_long['metric'].map({'avg_bout_awake_s':'Awake','avg_bout_nrem_s':'NREM','avg_bout_rem_s':'REM'})
    sns.boxplot(data=df_bouts_long, x='stage', y='med_bout_s', ax=ax2, color='lightgray', showcaps=True, boxprops={'zorder':0})
    sns.stripplot(data=df_bouts_long, x='stage', y='med_bout_s', hue='lab_display', ax=ax2, dodge=True, palette=palette, size=6, jitter=True, alpha=0.9, edgecolor='w', linewidth=0.5)
    ax2.set_ylabel('Average bout length (s)')
    ax2.set_xlabel('Stage')
    ax2.set_title('Average time spent in each stage (per subject)')
    if ax2.get_legend() is not None:
        ax2.get_legend().remove()
    ax2.grid(False)
    sns.despine(ax=ax2)

    # 3) Stacked bar: REM relative theta (bottom) + NREM relative theta (top)
    # Prefer the recomputed columns (fall back to original computed ones)
    # use theta/delta ratios (rem_theta_delta_recomputed / nrem_theta_delta_recomputed) for stacked theta/delta bars
    rem_col = 'rem_theta_delta_recomputed' if 'rem_theta_delta_recomputed' in df_subj.columns else 'rem_theta_delta'
    nrem_col = 'nrem_theta_delta_recomputed' if 'nrem_theta_delta_recomputed' in df_subj.columns else 'nrem_theta_delta'
    df_p3 = df_subj[['subject', 'lab_display', rem_col, nrem_col]].rename(columns={rem_col: 'rem_theta_delta', nrem_col: 'nrem_theta_delta'})
    # keep subjects that have at least one of the two measures (theta/delta ratios)
    df_p3 = df_p3.dropna(subset=['rem_theta_delta','nrem_theta_delta'], how='all').copy()
    # Sort by combined height (rem + nrem) for a meaningful ordering
    df_p3['sum_height'] = df_p3[['rem_theta_delta','nrem_theta_delta']].fillna(0).sum(axis=1)
    df_p3 = df_p3.sort_values('sum_height', ascending=False).reset_index(drop=True)

    # helper to lighten hex color
    def _lighten_hex(hex_color, amount=0.5):
        # mix with white by amount (0-1)
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        r2 = r + (1.0 - r) * amount
        g2 = g + (1.0 - g) * amount
        b2 = b + (1.0 - b) * amount
        return '#%02x%02x%02x' % (int(r2*255), int(g2*255), int(b2*255))

    x = np.arange(len(df_p3))
    rem_vals = df_p3['rem_theta_delta'].fillna(0).astype(float).to_numpy()
    nrem_vals = df_p3['nrem_theta_delta'].fillna(0).astype(float).to_numpy()
    lab_vals = df_p3['lab_display'].tolist()
    colors_rem = [palette.get(ld, '#777777') for ld in lab_vals]
    colors_nrem = [_lighten_hex(palette.get(ld, '#777777'), amount=0.6) for ld in lab_vals]

    # draw bottom (REM) and top (NREM) bars stacked
    ax3.bar(x, rem_vals, color=colors_rem, edgecolor='none')
    ax3.bar(x, nrem_vals, bottom=rem_vals, color=colors_nrem, edgecolor='none')
    ax3.set_xticks(x)
    ax3.set_xticklabels(df_p3['subject'], rotation=90, fontsize=7, ha='center')
    ax3.set_ylabel('Theta / Delta (REM bottom, NREM top)')
    # autoscale y with small padding based on sum of rem+nrem theta/delta
    if len(df_p3) > 0:
        max_sum = float((df_p3[['rem_theta_delta','nrem_theta_delta']].fillna(0).sum(axis=1)).max())
    pad = max(0.02, 0.05 * max_sum)
    ax3.set_ylim(0.0, max_sum + pad)
    ax3.set_title('REM (bottom) and NREM (top) θ/δ per subject')
    ax3.grid(False)
    sns.despine(ax=ax3)

    # small legend explaining stack colors (segment meaning)
    from matplotlib.patches import Patch as MPatch
    seg_handles = [MPatch(facecolor='#444444', label='REM θ/δ (bottom)'), MPatch(facecolor='#bbbbbb', label='NREM θ/δ (top)')]
    # place this legend inside the subplot in a compact location
    ax3.legend(handles=seg_handles, loc='upper right', fontsize=8)

    # 4) Transition probabilities scatter with fixed axes 0-1 (x = P(REM -> Awake), y = P(REM -> NREM))
    for ld in labs:
        sel = df_subj['lab_display'] == ld
        ax4.scatter(df_subj.loc[sel,'p_rem_to_awake'], df_subj.loc[sel,'p_rem_to_nrem'], label=ld, color=palette.get(ld, '#777777'), s=70, edgecolor='k', linewidth=0.4, alpha=0.95)
    ax4.set_xlim(0,1)
    ax4.set_ylim(0,1)
    ax4.set_xlabel('P(REM → Awake)')
    ax4.set_ylabel('P(REM → NREM)')
    ax4.set_title('Transition probabilities (first-order)')
    ax4.grid(False)
    sns.despine(ax=ax4)

    # Create a single legend for labs for the whole figure
    from matplotlib.patches import Patch as MPatch
    legend_handles = [MPatch(facecolor=palette.get(ld, '#777777'), label=ld) for ld in labs]
    fig.legend(handles=legend_handles, loc='center right', title='Lab', bbox_to_anchor=(0.98, 0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'df_subjects': df_subj,
        'df_bouts_long': df_bouts_long,
        'df_rem_vs_other': df_p3,
    }
    df_v = df_epochs[['subject','lab','Delta']].rename(columns={'Delta':'value'})
    df_v['lab_display'] = df_v['lab'].map(lambda x: lab_display_map.get(x, str(x)))
    # draw violin plot only (no individual dots)
    _sns.violinplot(data=df_v, x='lab_display', y='value', inner=None, ax=ax1, palette=[lab_colors.get(d, '#777777') for d in sorted(df_v['lab_display'].unique())])
    ax1.set_title('Per-subject epoch distributions (Relative Delta)')

    # BL: heatmap of subject means z-scored
    _sns.heatmap(subj_means_z, cmap='RdBu_r', center=0, ax=ax2, cbar_kws={'label':'z-score'})
    ax2.set_title('Subject mean across bands (Relative Delta)')

    # BR: ICC with CI
    ax3.errorbar(df_icc['band'], df_icc['icc'], yerr=[df_icc['icc']-df_icc['icc_lo'], df_icc['icc_hi']-df_icc['icc']], fmt='o', capsize=5)
    ax3.axhline(0.5, color='gray', linestyle='--', alpha=0.6)
    ax3.set_ylim(-0.05, 1.05)
    ax3.set_title('ICC(1) by band (95% CI)')

    _plt.tight_layout()
    _plt.savefig(save_path, dpi=300, bbox_inches='tight')
    _plt.close()

    return {'df_epochs': df_epochs, 'df_var': df_var, 'df_icc': df_icc, 'subj_means_z': subj_means_z}


def main():
    metadata_csv = Path('data/ds006366_processed/metadata.csv')
    
    # Create the inter-individual variability figure (conservative caps)
    try:
        figure_inter_individual_variability(str(metadata_csv), OUTPUT_DIR / 'inter_individual_variability.png', epoch_seconds=4, max_subjects_per_lab=20, bootstrap_rounds=200)
        print('Saved inter_individual_variability.png')
    except Exception as e:
        print(f'Failed to create inter-individual variability figure: {e}')

    # Also create the lab_3-only version
    try:
        figure_inter_individual_variability(str(metadata_csv), OUTPUT_DIR / 'inter_individual_variability_2.png', epoch_seconds=4, max_subjects_per_lab=None, bootstrap_rounds=200, lab_include='lab_3')
        print('Saved inter_individual_variability_2.png (lab_3 only)')
    except Exception as e:
        print(f'Failed to create lab_3 inter-individual variability figure: {e}')


    # print('Selecting example recording...')
    # participant, run_id = _choose_example_recording()
    # print(f'Using {participant} run {run_id} for figures')
    # # Only create the 3rd example raw-signals candidate and the all-labs pooled combined plot
    # # Produce only example_raw_signals_3.png
    # figure_raw_signals_example(participant, run_id, OUTPUT_DIR / 'example_raw_signals.png', save_only_idx=3)
    # print('Saved example_raw_signals_3.png')
    # # Create the lab comparison 2x2 figure

    # try:
    #     # use tighter caps to keep runtime small for interactive runs
    #     figure_lab_comparison_2x2(str(metadata_csv), OUTPUT_DIR / 'lab_comparison_2x2.png', max_subjects_per_lab=3, max_epochs_psd=1, nperseg_factor=1)
    #     print('Saved lab_comparison_2x2.png')
    # except Exception as e:
    #     print(f'Failed to create lab comparison figure: {e}')


    # Combined 2x2 figure containing: Relative Band Power by Sleep Stage,
    # EEG Band Power Ratios by Sleep Stage, EEG Power by Sleep Stage, EMG Power by Sleep Stage
    def figure_combined_stage_subplots(participant_id, run_id, save_path: Path):
        """Create a 2x2 subplot with the four requested panels.

        Layout:
        [ Relative Band Power | EEG Ratios ]
        [ EEG Power           | EMG Power  ]
        """
        # Recompute the data used in each individual figure so we can place them in subplots
        # We'll reuse logic from existing functions but keep local copies to avoid refactoring many return values.
        data = load_participant_data(participant_id, run_id)
        labels = data['labels']
        fs = 128

        # --- Band power (relative) per epoch ---
        eeg_channel = None
        for ch in ['EEG1', 'EEG2', 'EEG3', 'EEG4']:
            if ch in data:
                eeg_channel = ch
                break
        if eeg_channel is None:
            raise FileNotFoundError("No EEG channel for combined plot")

        eeg = data[eeg_channel]
        epoch_len = 30 * fs
        n_epochs = len(eeg) // epoch_len

        bands = {
            'Delta': (0.5, 4),
            'Theta': (4, 8),
            'Alpha': (8, 12),
            'Beta': (12, 30),
            'Gamma': (30, 100),
        }

        import pandas as pd
        records_band = []
        for e in range(n_epochs):
            s = e * epoch_len
            eidx = s + epoch_len
            seg = eeg[s:eidx]
            label = SLEEP_STAGE_MAPPING.get(int(labels[s]), 'Unknown')
            f, pxx = signal.welch(seg, fs=fs, nperseg=fs*4)
            total = np.sum(pxx[(f>=0.5)&(f<=100)])
            rec = {'stage': label}
            for bname, (lo, hi) in bands.items():
                power = np.sum(pxx[(f>=lo)&(f<hi)])
                rec[bname] = power if total==0 else power / total
            records_band.append(rec)
        df_band = pd.DataFrame.from_records(records_band)

        # --- EEG absolute power per epoch ---
        records_eeg = []
        for e in range(n_epochs):
            s = e * epoch_len
            eidx = s + epoch_len
            seg = eeg[s:eidx]
            label = SLEEP_STAGE_MAPPING.get(int(labels[s]), 'Unknown')
            power = np.mean(seg.astype(float)**2)
            records_eeg.append({'stage': label, 'eeg_power': power})
        df_eeg = pd.DataFrame.from_records(records_eeg)

        # --- EMG absolute power per epoch ---
        if 'EMG' in data:
            emg = data['EMG']
            records_emg = []
            for e in range(n_epochs):
                s = e * epoch_len
                eidx = s + epoch_len
                seg = emg[s:eidx]
                label = SLEEP_STAGE_MAPPING.get(int(labels[s]), 'Unknown')
                power = np.mean(seg.astype(float)**2)
                records_emg.append({'stage': label, 'emg_power': power})
            df_emg = pd.DataFrame.from_records(records_emg)
        else:
            df_emg = pd.DataFrame(columns=['stage','emg_power'])

    # --- EEG band power ratios (Theta/Delta, Beta/Delta) ---
        # We'll mimic the previous ratio computation but only keep the specified ratios
        records_ratio = []
        base_bands = {
            'Delta': (0.5, 4),
            'Theta': (4, 8),
            'Alpha': (8, 12),
            'Beta': (12, 30),
            'Gamma': (30, 100),
        }
        for e in range(n_epochs):
            s = e * epoch_len
            eidx = s + epoch_len
            seg = eeg[s:eidx]
            label = SLEEP_STAGE_MAPPING.get(int(labels[s]), 'Unknown')
            f, pxx = signal.welch(seg, fs=fs, nperseg=fs*4)
            band_powers = {}
            for bname, (lo, hi) in base_bands.items():
                power = np.sum(pxx[(f>=lo)&(f<hi)])
                band_powers[bname] = power
            eps = 1e-12
            rec = {'stage': label}
            rec['Theta/Delta'] = band_powers.get('Theta', 0.0) / (band_powers.get('Delta', 0.0) + eps)
            rec['Beta/Delta'] = band_powers.get('Beta', 0.0) / (band_powers.get('Delta', 0.0) + eps)
            records_ratio.append(rec)
        df_ratio = pd.DataFrame.from_records(records_ratio)

        # --- Build 2x2 subplot figure ---
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))

        # Top-left: Relative Band Power by Sleep Stage
        ax = axs[0,0]
        df_m = df_band.melt(id_vars='stage', value_vars=list(bands.keys()), var_name='band', value_name='relative_power')
        sns.barplot(data=df_m, x='band', y='relative_power', hue='stage', hue_order=STAGE_ORDER, palette=[SLEEP_STAGE_COLORS.get(s) for s in STAGE_ORDER], ax=ax)
        ax.set_title('Relative Band Power by Sleep Stage')
        ax.set_ylabel('Relative Power')

        # Top-right: EEG Band Power Ratios by Sleep Stage
        ax = axs[0,1]
        df_mr = df_ratio.melt(id_vars='stage', value_vars=['Theta/Delta','Beta/Delta'], var_name='ratio', value_name='value')
        sns.barplot(data=df_mr, x='ratio', y='value', hue='stage', hue_order=STAGE_ORDER, palette=[SLEEP_STAGE_COLORS.get(s) for s in STAGE_ORDER], ax=ax)
        ax.set_title('EEG Band Power Ratios by Sleep Stage')
        ax.set_ylabel('Ratio')

        # Bottom-left: Combined EEG and EMG Power by Sleep Stage (grouped bars)
        ax = axs[1, 0]
        # Combine EEG and EMG into a long-form dataframe for grouped plotting
        import pandas as _pd
        # use display name for signal label (show 'EEG' instead of 'EEG1')
        display_signal_name = 'EEG' if eeg_channel and str(eeg_channel).upper().startswith('EEG') else eeg_channel
        df_eeg_long = _pd.DataFrame({'stage': df_eeg['stage'], 'value': df_eeg['eeg_power'], 'signal': display_signal_name})
        if not df_emg.empty:
            df_emg_long = _pd.DataFrame({'stage': df_emg['stage'], 'value': df_emg['emg_power'], 'signal': 'EMG'})
            df_combined = _pd.concat([df_eeg_long, df_emg_long], ignore_index=True)
        else:
            df_combined = df_eeg_long.copy()

        # Use user-requested signal colors: EEG = #4C72B0, EMG = #DD8452
        sig_palette = {'EEG': '#4C72B0', 'EMG': '#DD8452'}
        # Determine the ordering of signal levels present
        signals_present = sorted(df_combined['signal'].unique())
        palette_list = [sig_palette.get(s, '#777777') for s in signals_present]
        sns.barplot(data=df_combined, x='stage', y='value', hue='signal', order=STAGE_ORDER, palette=palette_list, ax=ax)
        ax.set_title(f'Absolute {display_signal_name} and EMG Power by Sleep Stage')
        ax.set_ylabel('Power (mean squared amplitude)')
        ax.legend(title='Signal')

        # Bottom-right: Transition matrix (P(next|current)) excluding self-transitions
        ax = axs[1, 1]
        # Build transition counts between different stages only
        stage_names = STAGE_ORDER
        name_to_idx = {name: i for i, name in enumerate(stage_names)}
        # Build sequence of stage names per epoch (use the first sample of each epoch)
        seq = []
        for e in range(n_epochs):
            s = e * epoch_len
            seq.append(SLEEP_STAGE_MAPPING.get(int(labels[s]), 'Unknown'))

        import numpy as _np
        counts = _np.zeros((len(stage_names), len(stage_names)), dtype=float)
        for a, b in zip(seq[:-1], seq[1:]):
            if a not in name_to_idx or b not in name_to_idx:
                continue
            if a == b:
                # skip self-transitions
                continue
            i = name_to_idx[a]
            j = name_to_idx[b]
            counts[i, j] += 1.0

        # Convert counts to conditional probabilities P(next=b | current=a, next!=a)
        probs = _np.zeros_like(counts)
        for i in range(counts.shape[0]):
            row_sum = counts[i].sum()
            if row_sum > 0:
                probs[i] = counts[i] / row_sum
            else:
                probs[i] = 0.0

        import seaborn as _sns
        df_tm = _pd.DataFrame(probs, index=stage_names, columns=stage_names)
        # Mask diagonal since self-transitions were excluded and probabilities there are zero
        mask = _np.eye(len(stage_names), dtype=bool)
        # Create a two-tone same-hue colormap from light -> dark of #6C9BD9
        from matplotlib.colors import LinearSegmentedColormap
        light = '#DCE9FB'  # lighter tint of #6C9BD9
        dark = '#6C9BD9'
        cmap_same_hue = LinearSegmentedColormap.from_list('same_hue', [light, dark])
        _sns.heatmap(df_tm, annot=True, fmt='.2f', cmap=cmap_same_hue, cbar=True, ax=ax, mask=mask, linewidths=0.5, linecolor='white')
        ax.set_title('Transition Matrix (Excluding Self-transitions)')
        ax.set_xlabel('Next stage')
        ax.set_ylabel('Current stage')

        # Adjust layout: rely on per-subplot legends and avoid a redundant figure-level legend
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    # skip single-recording and lab-pooled combined figures; only produce all-labs pooled combined plot below

    # --- New: pooled across ALL labs ---
    try:
        # Reuse the lab pooling function by passing no lab filter: implement a thin wrapper
        def figure_combined_stage_subplots_all(metadata_csv: str, save_path: Path):
            import pandas as pd
            meta = pd.read_csv(metadata_csv)
            # pass each unique (participant_id, run) row just like the lab function did
            # create a fake lab_name filter by selecting all rows
            # We'll call the existing function's internal logic by reconstructing rows DataFrame
            rows = meta.copy()

            # We'll duplicate the core of the lab function here to avoid exposing internals
            pooled_band = []
            pooled_eeg = []
            pooled_emg = []
            pooled_ratio = []

            for _, r in rows.iterrows():
                participant_id = r['participant_id']
                run_id = int(r['run'])
                try:
                    data = load_participant_data(participant_id, run_id)
                except Exception as e:
                    print(f"Skipping {participant_id} run {run_id}: failed to load ({e})")
                    continue

                labels = data['labels']
                fs = 128
                # choose first EEG
                eeg_channel = None
                for ch in ['EEG1', 'EEG2', 'EEG3', 'EEG4']:
                    if ch in data:
                        eeg_channel = ch
                        break
                if eeg_channel is None:
                    print(f"Skipping {participant_id} run {run_id}: no EEG channel")
                    continue
                eeg = data[eeg_channel]
                epoch_len = 30 * fs
                n_epochs = len(eeg) // epoch_len

                bands = {
                    'Delta': (0.5, 4),
                    'Theta': (4, 8),
                    'Alpha': (8, 12),
                    'Beta': (12, 30),
                    'Gamma': (30, 100),
                }

                for e in range(n_epochs):
                    s = e * epoch_len
                    eidx = s + epoch_len
                    seg = eeg[s:eidx]
                    label = SLEEP_STAGE_MAPPING.get(int(labels[s]), 'Unknown')
                    f, pxx = signal.welch(seg, fs=fs, nperseg=fs*4)
                    total = np.sum(pxx[(f>=0.5)&(f<=100)])
                    rec_band = {'stage': label}
                    for bname, (lo, hi) in bands.items():
                        power = np.sum(pxx[(f>=lo)&(f<hi)])
                        rec_band[bname] = power if total==0 else power / total
                    pooled_band.append(rec_band)

                    rec_eeg = {'stage': label, 'eeg_power': np.mean(seg.astype(float)**2)}
                    pooled_eeg.append(rec_eeg)

                    band_powers = {b: np.sum(pxx[(f>=lo)&(f<hi)]) for b, (lo, hi) in bands.items()}
                    eps = 1e-12
                    rec_ratio = {'stage': label}
                    rec_ratio['Theta/Delta'] = band_powers.get('Theta', 0.0) / (band_powers.get('Delta', 0.0) + eps)
                    rec_ratio['Beta/Delta'] = band_powers.get('Beta', 0.0) / (band_powers.get('Delta', 0.0) + eps)
                    pooled_ratio.append(rec_ratio)

                    if 'EMG' in data:
                        emg = data['EMG']
                        seg_emg = emg[s:eidx]
                        rec_emg = {'stage': label, 'emg_power': np.mean(seg_emg.astype(float)**2)}
                        pooled_emg.append(rec_emg)

            import pandas as pd
            df_band = pd.DataFrame.from_records(pooled_band)
            df_eeg = pd.DataFrame.from_records(pooled_eeg)
            df_emg = pd.DataFrame.from_records(pooled_emg) if len(pooled_emg) > 0 else pd.DataFrame(columns=['stage','emg_power'])
            df_ratio = pd.DataFrame.from_records(pooled_ratio)

            # plot same 2x2 layout (reuse code from lab-level but mark as pooled-all)
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))
            ax = axs[0,0]
            df_m = df_band.melt(id_vars='stage', value_vars=list(bands.keys()), var_name='band', value_name='relative_power')
            sns.barplot(data=df_m, x='band', y='relative_power', hue='stage', hue_order=STAGE_ORDER, palette=[SLEEP_STAGE_COLORS.get(s) for s in STAGE_ORDER], ax=ax)
            ax.set_title('Relative Band Power by Sleep Stage')
            ax.set_ylabel('Relative Power')

            ax = axs[0,1]
            df_mr = df_ratio.melt(id_vars='stage', value_vars=['Theta/Delta','Beta/Delta'], var_name='ratio', value_name='value')
            sns.barplot(data=df_mr, x='ratio', y='value', hue='stage', hue_order=STAGE_ORDER, palette=[SLEEP_STAGE_COLORS.get(s) for s in STAGE_ORDER], ax=ax)
            ax.set_title('EEG Band Power Ratios by Sleep Stage')
            ax.set_ylabel('Ratio')

            ax = axs[1,0]
            display_signal_name = 'EEG'
            df_eeg_long = pd.DataFrame({'stage': df_eeg['stage'], 'value': df_eeg['eeg_power'], 'signal': display_signal_name})
            if not df_emg.empty:
                df_emg_long = pd.DataFrame({'stage': df_emg['stage'], 'value': df_emg['emg_power'], 'signal': 'EMG'})
                df_combined = pd.concat([df_eeg_long, df_emg_long], ignore_index=True)
            else:
                df_combined = df_eeg_long.copy()
            sig_palette = {'EEG': '#4C72B0', 'EMG': '#DD8452'}
            signals_present = sorted(df_combined['signal'].unique())
            palette_list = [sig_palette.get(s, '#777777') for s in signals_present]
            sns.barplot(data=df_combined, x='stage', y='value', hue='signal', order=STAGE_ORDER, palette=palette_list, ax=ax)
            ax.set_title(f'Absolute {display_signal_name} and EMG Power by Sleep Stage')
            ax.set_ylabel('Power (mean squared amplitude)')

            ax = axs[1,1]
            stage_names = STAGE_ORDER
            name_to_idx = {name: i for i, name in enumerate(stage_names)}
            seq = list(df_band['stage'])
            import numpy as _np
            counts = _np.zeros((len(stage_names), len(stage_names)), dtype=float)
            for a, b in zip(seq[:-1], seq[1:]):
                if a not in name_to_idx or b not in name_to_idx:
                    continue
                if a == b:
                    continue
                i = name_to_idx[a]
                j = name_to_idx[b]
                counts[i, j] += 1.0
            probs = _np.zeros_like(counts)
            for i in range(counts.shape[0]):
                row_sum = counts[i].sum()
                if row_sum > 0:
                    probs[i] = counts[i] / row_sum
                else:
                    probs[i] = 0.0

            df_tm = pd.DataFrame(probs, index=stage_names, columns=stage_names)
            mask = _np.eye(len(stage_names), dtype=bool)
            from matplotlib.colors import LinearSegmentedColormap
            light = '#DCE9FB'
            dark = '#6C9BD9'
            cmap_same_hue = LinearSegmentedColormap.from_list('same_hue', [light, dark])
            sns.heatmap(df_tm, annot=True, fmt='.2f', cmap=cmap_same_hue, cbar=True, ax=ax, mask=mask, linewidths=0.5, linecolor='white')
            ax.set_title('Transition Matrix (Excluding Self-transitions)')
            ax.set_xlabel('Next stage')
            ax.set_ylabel('Current stage')

            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

        metadata_csv = Path('data/ds006366_processed/metadata.csv')
        figure_combined_stage_subplots_all(str(metadata_csv), OUTPUT_DIR / 'combined_stage_subplots_all_labs.png')
        print('Saved combined_stage_subplots_all_labs.png')
    except Exception as e:
        print(f'Failed to create pooled ALL-labs figure: {e}')


if __name__ == '__main__':
    main()
