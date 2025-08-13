"""Artifact stage characteristics analysis.

Compares Artifact vs non-Artifact sleep stages using signal statistics and spectral metrics.
Generates 3-4 core plots:
 1. Relative band power comparison (Artifact vs Non-Artifact)
 2. Signal variance / standard deviation distribution (Artifact vs Non-Artifact)
 3. Mean PSD (0.5–60 Hz) comparison with 95% CI shading
 4. 50 Hz line noise relative power distribution (Artifact vs Non-Artifact)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from helpers import (
    load_metadata,
    load_participant_data,
    ensure_output_directory,
    SLEEP_STAGE_MAPPING,
    STAGE_ORDER,
    SLEEP_STAGE_COLORS,
    calculate_signal_statistics,
)

plt.style.use('seaborn-v0_8')

BANDS_REL = ['delta_power_rel','theta_power_rel','alpha_power_rel','beta_power_rel','gamma_power_rel']


def analyze_artifact_characteristics(output_dir="results/data_exploration", exclude_lab1=True, sampling_rate=128):
    print("="*80)
    print("ANALYZING ARTIFACT STAGE CHARACTERISTICS")
    print("="*80)
    output_dir = ensure_output_directory(output_dir)
    out_dir = Path(output_dir) / "artifact_characteristics"
    out_dir.mkdir(exist_ok=True)

    metadata = load_metadata()
    if exclude_lab1:
        before = len(metadata)
        metadata = metadata[metadata['lab'] != 'lab_1']
        print(f"Excluded lab_1: {before - len(metadata)} records removed")

    rows = []
    artifact_psds = []
    non_artifact_psds = []
    freq_grid = np.arange(0.5, 60.5, 0.5)
    eps = 1e-12

    for (pid, run_id), group in metadata.groupby(['participant_id','run']):
        try:
            data = load_participant_data(pid, run_id)
        except Exception:
            continue
        labels = data.get('labels')
        if labels is None or len(labels) < 10:
            continue
    # Pick first available EEG channel
        eeg_channel = None
        for ch in ['EEG1','EEG2','EEG3','EEG4']:
            if ch in data:
                eeg_channel = ch
                break
        if eeg_channel is None:
            continue
        signal = data[eeg_channel]
        try:
            stats = calculate_signal_statistics(signal, labels, sampling_rate=sampling_rate)
        except Exception:
            continue

        if 'Artifact' not in stats:
            # No artifact stage present
            continue

        # Weighted aggregation across non-artifact stages
        total_non_artifact_samples = sum(stats[s]['samples'] for s in stats if s != 'Artifact')
        if total_non_artifact_samples == 0:
            continue

        def weighted(metric):
            num = 0.0
            for s, sd in stats.items():
                if s == 'Artifact':
                    continue
                if metric in sd and total_non_artifact_samples > 0:
                    num += sd[metric] * sd['samples']
            return num / total_non_artifact_samples if total_non_artifact_samples>0 else np.nan

        row = {
            'participant_id': pid,
            'run': run_id,
            'lab': group.iloc[0]['lab']
        }
        # Store band powers (relative), variance, std, power_50hz / total_power
        for m in BANDS_REL:
            row[f'artifact_{m}'] = stats['Artifact'].get(m, np.nan)
            row[f'non_artifact_{m}'] = weighted(m)
        # Raw stats
        row['artifact_variance'] = stats['Artifact'].get('variance', np.nan)
        row['non_artifact_variance'] = weighted('variance')
        row['artifact_std'] = stats['Artifact'].get('std', np.nan)
        row['non_artifact_std'] = weighted('std')
        # Line noise relative contribution
        artifact_total = stats['Artifact'].get('total_power', np.nan)
        non_artifact_total = weighted('total_power')
        artifact_50 = stats['Artifact'].get('power_50hz', np.nan)
        non_artifact_50 = weighted('power_50hz')
        row['artifact_50hz_rel'] = artifact_50 / artifact_total if artifact_total and artifact_total>0 else np.nan
        row['non_artifact_50hz_rel'] = non_artifact_50 / non_artifact_total if non_artifact_total and non_artifact_total>0 else np.nan
        # ----------------------------------------------------
        # EMG amplitude characteristics (if EMG present)
        # ----------------------------------------------------
        if 'EMG' in data:
            emg = data['EMG']
            # Compute per-stage amplitude metrics directly (avoid heavy PSD)
            artifact_mask = (labels == [k for k,v in SLEEP_STAGE_MAPPING.items() if v=='Artifact'][0])
            # Map numeric codes to Artifact and non-artifact masks
            # Numeric codes: we infer by reverse mapping
            code_to_stage = SLEEP_STAGE_MAPPING
            # Precompute masks per stage if needed
            # Artifact metrics
            if artifact_mask.any():
                art_emg = emg[artifact_mask]
                artifact_emg_std = float(np.std(art_emg))
                artifact_emg_rms = float(np.sqrt(np.mean(art_emg**2)))
                artifact_emg_mad = float(np.median(np.abs(art_emg - np.median(art_emg))))
            else:
                artifact_emg_std = np.nan
                artifact_emg_rms = np.nan
                artifact_emg_mad = np.nan
            # Non-artifact weighted metrics
            non_mask = ~artifact_mask
            if non_mask.any():
                non_emg = emg[non_mask]
                non_emg_std = float(np.std(non_emg))
                non_emg_rms = float(np.sqrt(np.mean(non_emg**2)))
                non_emg_mad = float(np.median(np.abs(non_emg - np.median(non_emg))))
            else:
                non_emg_std = np.nan
                non_emg_rms = np.nan
                non_emg_mad = np.nan
            row.update({
                'artifact_emg_std': artifact_emg_std,
                'non_artifact_emg_std': non_emg_std,
                'artifact_emg_rms': artifact_emg_rms,
                'non_artifact_emg_rms': non_emg_rms,
                'artifact_emg_mad': artifact_emg_mad,
                'non_artifact_emg_mad': non_emg_mad,
            })
        else:
            row.update({
                'artifact_emg_std': np.nan,
                'non_artifact_emg_std': np.nan,
                'artifact_emg_rms': np.nan,
                'non_artifact_emg_rms': np.nan,
                'artifact_emg_mad': np.nan,
                'non_artifact_emg_mad': np.nan,
            })

        rows.append(row)

        # PSD handling (interpolate onto common grid)
        for stage_name, sd in stats.items():
            if 'psd' not in sd or 'frequencies' not in sd:
                continue
        artifact_psd = stats['Artifact'].get('psd')
        artifact_freqs = stats['Artifact'].get('frequencies')
        if artifact_psd is not None and artifact_freqs is not None:
            # restrict
            mask = (artifact_freqs >= 0.5) & (artifact_freqs <= 60)
            art_interp = np.interp(freq_grid, artifact_freqs[mask], artifact_psd[mask], left=np.nan, right=np.nan)
        else:
            art_interp = np.full_like(freq_grid, np.nan, dtype=float)
        # Non-artifact weighted PSD
        weighted_psd = np.zeros_like(freq_grid, dtype=float)
        weight_sum = 0.0
        for s, sd in stats.items():
            if s == 'Artifact' or 'psd' not in sd:
                continue
            freqs = sd.get('frequencies')
            psd = sd.get('psd')
            if freqs is None or psd is None:
                continue
            mask = (freqs >= 0.5) & (freqs <= 60)
            interp = np.interp(freq_grid, freqs[mask], psd[mask], left=np.nan, right=np.nan)
            w = sd['samples']
            if np.isnan(interp).all():
                continue
            # Replace NaNs with 0 for weighting
            interp = np.nan_to_num(interp, nan=0.0)
            weighted_psd += interp * w
            weight_sum += w
        if weight_sum > 0:
            weighted_psd /= weight_sum
        else:
            weighted_psd[:] = np.nan

        artifact_psds.append(art_interp)
        non_artifact_psds.append(weighted_psd)

    if not rows:
        print("No recordings with Artifact stage found for analysis.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'artifact_vs_non_artifact_metrics.csv', index=False)

    # ---- Plot 1: Relative band power comparison ----
    band_records = []
    for b in BANDS_REL:
        if df[f'artifact_{b}'].notna().sum()==0:
            continue
        band_records.append({
            'band': b.replace('_power_rel','').capitalize(),
            'Artifact_mean': df[f'artifact_{b}'].mean(),
            'Artifact_std': df[f'artifact_{b}'].std(),
            'NonArtifact_mean': df[f'non_artifact_{b}'].mean(),
            'NonArtifact_std': df[f'non_artifact_{b}'].std()
        })
    band_df = pd.DataFrame(band_records)
    band_df.to_csv(out_dir / 'band_relative_power_summary.csv', index=False)

    if not band_df.empty:
        x = np.arange(len(band_df))
        w = 0.38
        plt.figure(figsize=(8,5))
        plt.bar(x - w/2, band_df['Artifact_mean'], yerr=band_df['Artifact_std'], capsize=4, width=w, label='Artifact', color=SLEEP_STAGE_COLORS['Artifact'])
        plt.bar(x + w/2, band_df['NonArtifact_mean'], yerr=band_df['NonArtifact_std'], capsize=4, width=w, label='Non-Artifact', color='#555555')
        plt.xticks(x, band_df['band'])
        plt.ylabel('Relative Power (fraction of 0.5–100 Hz)')
        plt.title('Relative Band Power: Artifact vs Non-Artifact')
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / 'relative_band_power_comparison.png', dpi=160)
        plt.close()

    # ---- Plot 2: Variance / STD distribution ----
    var_long = df.melt(id_vars=['participant_id','run','lab'], value_vars=['artifact_variance','non_artifact_variance'], var_name='type', value_name='variance')
    var_long['group'] = var_long['type'].map({'artifact_variance':'Artifact','non_artifact_variance':'Non-Artifact'})
    plt.figure(figsize=(6,5))
    sns.boxplot(data=var_long, x='group', y='variance', palette=[SLEEP_STAGE_COLORS['Artifact'],'#555555'], showfliers=False)
    sns.stripplot(data=var_long, x='group', y='variance', color='black', size=2, alpha=0.3)
    plt.ylabel('Signal Variance')
    plt.title('Signal Variance Distribution')
    plt.tight_layout()
    plt.savefig(out_dir / 'variance_distribution.png', dpi=160)
    plt.close()

    # ---- Plot 3: Mean PSD comparison ----
    artifact_psds_arr = np.array(artifact_psds)
    non_artifact_psds_arr = np.array(non_artifact_psds)
    # Remove rows with NaNs entirely
    artifact_mask = ~np.isnan(artifact_psds_arr).all(axis=1)
    non_artifact_mask = ~np.isnan(non_artifact_psds_arr).all(axis=1)
    artifact_psds_arr = artifact_psds_arr[artifact_mask]
    non_artifact_psds_arr = non_artifact_psds_arr[non_artifact_mask]

    mean_art = np.nanmean(artifact_psds_arr, axis=0)
    mean_non = np.nanmean(non_artifact_psds_arr, axis=0)

    def ci_bounds(arr):
        if arr.shape[0] < 2:
            return mean_art, mean_art
        import scipy.stats as st
        n = arr.shape[0]
        sem = np.nanstd(arr, axis=0, ddof=1)/np.sqrt(n)
        t = 1.96  # approximate
        return mean_art - t*sem, mean_art + t*sem

    art_ci_low, art_ci_high = ci_bounds(artifact_psds_arr)
    non_ci_low, non_ci_high = ci_bounds(non_artifact_psds_arr)

    plt.figure(figsize=(8,5))
    plt.plot(freq_grid, 10*np.log10(mean_art+eps), label='Artifact', color=SLEEP_STAGE_COLORS['Artifact'])
    plt.fill_between(freq_grid, 10*np.log10(art_ci_low+eps), 10*np.log10(art_ci_high+eps), color=SLEEP_STAGE_COLORS['Artifact'], alpha=0.25)
    plt.plot(freq_grid, 10*np.log10(mean_non+eps), label='Non-Artifact', color='#555555')
    plt.fill_between(freq_grid, 10*np.log10(non_ci_low+eps), 10*np.log10(non_ci_high+eps), color='#555555', alpha=0.25)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (dB)')
    plt.title('Mean EEG PSD (0.5–60 Hz): Artifact vs Non-Artifact')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'mean_psd_artifact_vs_non.png', dpi=160)
    plt.close()

    # ---- Plot 4: 50 Hz relative line noise distribution (with outlier removal) ----
    ln_long = df[['artifact_50hz_rel','non_artifact_50hz_rel']].melt(value_vars=['artifact_50hz_rel','non_artifact_50hz_rel'], var_name='group', value_name='rel_50hz')
    ln_long['group'] = ln_long['group'].map({'artifact_50hz_rel':'Artifact','non_artifact_50hz_rel':'Non-Artifact'})
    # Remove NaNs
    ln_long = ln_long.dropna(subset=['rel_50hz'])
    # Compute group-wise IQR fences
    fences = []
    filtered_parts = []
    for g, sub in ln_long.groupby('group'):
        if len(sub) < 4:
            # Too few points to define IQR meaningfully
            sub = sub.copy()
            sub['is_outlier'] = False
            filtered_parts.append(sub)
            fences.append({'group': g, 'q1': np.nan, 'q3': np.nan, 'iqr': np.nan, 'lower_fence': np.nan, 'upper_fence': np.nan, 'n_outliers': 0, 'n_total': len(sub)})
            continue
        q1 = sub['rel_50hz'].quantile(0.25)
        q3 = sub['rel_50hz'].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (sub['rel_50hz'] >= lower) & (sub['rel_50hz'] <= upper)
        sub_f = sub.copy()
        sub_f['is_outlier'] = ~mask
        filtered_parts.append(sub_f[mask])
        fences.append({'group': g, 'q1': q1, 'q3': q3, 'iqr': iqr, 'lower_fence': lower, 'upper_fence': upper, 'n_outliers': (~mask).sum(), 'n_total': len(sub)})
    ln_filtered = pd.concat(filtered_parts, ignore_index=True) if filtered_parts else ln_long
    fences_df = pd.DataFrame(fences)
    fences_df.to_csv(out_dir / 'relative_50hz_outlier_summary.csv', index=False)

    plt.figure(figsize=(6,5))
    sns.boxplot(data=ln_filtered, x='group', y='rel_50hz', palette=[SLEEP_STAGE_COLORS['Artifact'],'#555555'], showfliers=False)
    sns.stripplot(data=ln_filtered, x='group', y='rel_50hz', color='black', size=2, alpha=0.35)
    plt.ylabel('Relative 50 Hz Power (Fraction of Total)')
    plt.title('50 Hz Line Noise Contribution (Outliers Removed)')
    plt.tight_layout()
    plt.savefig(out_dir / 'relative_50hz_noise_contribution.png', dpi=160)
    plt.close()

    # ---- EMG Amplitude Plot (RMS) with outlier removal ----
    if df['artifact_emg_rms'].notna().any() or df['non_artifact_emg_rms'].notna().any():
        emg_long = df[['artifact_emg_rms','non_artifact_emg_rms']].melt(value_vars=['artifact_emg_rms','non_artifact_emg_rms'], var_name='group', value_name='emg_rms')
        emg_long['group'] = emg_long['group'].map({'artifact_emg_rms':'Artifact','non_artifact_emg_rms':'Non-Artifact'})
        emg_long = emg_long.dropna(subset=['emg_rms'])
        # IQR filtering
        emg_filtered_parts = []
        emg_fences = []
        for g, sub in emg_long.groupby('group'):
            if len(sub) < 4:
                sub = sub.copy()
                sub['is_outlier'] = False
                emg_filtered_parts.append(sub)
                emg_fences.append({'group': g, 'q1': np.nan, 'q3': np.nan, 'iqr': np.nan, 'lower_fence': np.nan, 'upper_fence': np.nan, 'n_outliers': 0, 'n_total': len(sub)})
                continue
            q1 = sub['emg_rms'].quantile(0.25)
            q3 = sub['emg_rms'].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5*iqr
            upper = q3 + 1.5*iqr
            mask = (sub['emg_rms'] >= lower) & (sub['emg_rms'] <= upper)
            sub_f = sub.copy()
            sub_f['is_outlier'] = ~mask
            emg_filtered_parts.append(sub_f[mask])
            emg_fences.append({'group': g, 'q1': q1, 'q3': q3, 'iqr': iqr, 'lower_fence': lower, 'upper_fence': upper, 'n_outliers': (~mask).sum(), 'n_total': len(sub)})
        emg_filtered = pd.concat(emg_filtered_parts, ignore_index=True) if emg_filtered_parts else emg_long
        pd.DataFrame(emg_fences).to_csv(out_dir / 'emg_rms_outlier_summary.csv', index=False)
        plt.figure(figsize=(6,5))
        sns.boxplot(data=emg_filtered, x='group', y='emg_rms', palette=[SLEEP_STAGE_COLORS['Artifact'],'#555555'], showfliers=False)
        sns.stripplot(data=emg_filtered, x='group', y='emg_rms', color='black', size=2, alpha=0.35)
        plt.ylabel('EMG RMS Amplitude')
        plt.title('EMG RMS Amplitude (Outliers Removed)')
        plt.tight_layout()
        plt.savefig(out_dir / 'emg_rms_amplitude_comparison.png', dpi=160)
        plt.close()

    print(f"Artifact characteristics analysis complete. Records analyzed: {len(df)}")
    print(f"Results saved to: {out_dir}")

if __name__ == '__main__':
    analyze_artifact_characteristics()
