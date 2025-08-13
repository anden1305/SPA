"""Analysis of 50Hz interference / line noise outliers across recordings.
Generates statistics and visualizations for the 49.5–50.5 Hz band (default) compared to local baseline.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from helpers import load_metadata, load_participant_data, ensure_output_directory, SLEEP_STAGE_MAPPING


def analyze_50hz_interference(output_dir="results/data_exploration", exclude_lab1=False,
                              band_center=50.0, band_half_width=0.5,
                              baseline_margin=3.0, max_baseline_offset=5.0):
    print("="*60)
    print("ANALYZING 50Hz INTERFERENCE / LINE NOISE (band {:.1f}–{:.1f} Hz)".format(band_center-band_half_width, band_center+band_half_width))
    print("="*60)
    output_dir = ensure_output_directory(output_dir)
    out_dir = Path(output_dir) / "interference_50hz"
    out_dir.mkdir(exist_ok=True)

    metadata = load_metadata()
    if exclude_lab1:
        metadata = metadata[metadata['lab'] != 'lab_1']
        print("Excluding lab_1 for 50Hz analysis")

    records = []  # per-recording/channel metrics (one row per participant/run/channel)
    psd_snippets = []  # for overlay plots (40-60 Hz)
    full_psds = []  # store full PSDs (freqs 0..Nyquist) for later global plotting

    # Iterate all participant runs
    for (pid, run_id), run_rows in metadata.groupby(['participant_id', 'run']):
        lab = run_rows.iloc[0]['lab']
        try:
            data = load_participant_data(pid, run_id)
        except Exception as e:
            print(f"  Skipping {pid} run {run_id}: load error {e}")
            continue
        labels = data.get('labels')
        # Analyze EEG channels only
        for channel, signal in data.items():
            if channel == 'labels' or 'EMG' in channel:
                continue
            # We need stage-based stats for PSD from helpers: recompute quickly
            from helpers import calculate_signal_statistics
            try:
                stats_per_stage = calculate_signal_statistics(signal, labels)
            except Exception as e:
                print(f"    Failed stats for {pid} {run_id} {channel}: {e}")
                continue
            # Combine PSD across stages (average)
            psds = []
            freqs_ref = None
            stage_durations = []
            for stage_name, st in stats_per_stage.items():
                if 'psd' in st and 'frequencies' in st:
                    freqs = st['frequencies']
                    psd_vals = st['psd']
                    if freqs_ref is None:
                        freqs_ref = freqs
                        psds.append(psd_vals)
                    else:
                        # If length mismatch, interpolate to freqs_ref
                        if len(freqs) != len(freqs_ref) or not np.allclose(freqs, freqs_ref):
                            try:
                                psd_vals = np.interp(freqs_ref, freqs, psd_vals)
                            except Exception:
                                continue
                        psds.append(psd_vals)
                    stage_durations.append(st.get('duration_minutes', 0))
            if not psds or freqs_ref is None:
                continue
            try:
                psd_matrix = np.vstack(psds)
            except ValueError:
                # Fallback: skip this channel if still inconsistent
                continue
            weights = np.array(stage_durations) + 1e-6
            weights /= weights.sum()
            weighted_psd = (psd_matrix.T @ weights).T if psd_matrix.shape[0] == len(weights) else psd_matrix.mean(axis=0)
            # Focus on required frequency bands
            # Interference band (e.g., 48-52 Hz)
            band_low = band_center - band_half_width
            band_high = band_center + band_half_width
            band_mask = (freqs_ref >= band_low) & (freqs_ref <= band_high)
            if not band_mask.any():
                continue
            band_segment = weighted_psd[band_mask]
            band_power = band_segment.mean()
            band_peak_power = band_segment.max()
            # Integrated (sum) powers for fraction (0..1)
            band_sum = band_segment.sum()
            total_mask_range = (freqs_ref >= 0.5) & (freqs_ref <= 60)
            total_sum = weighted_psd[total_mask_range].sum()
            # Local baseline just outside band: (band_low - baseline_margin) to band_low, and band_high to (band_high + baseline_margin)
            base_low_start = max(0, band_low - baseline_margin)
            base_low_end = band_low
            base_high_start = band_high
            base_high_end = band_high + baseline_margin
            # Cap high baseline to band_center + max_baseline_offset if provided
            base_high_end = min(base_high_end, band_center + max_baseline_offset)
            base_mask_low = (freqs_ref >= base_low_start) & (freqs_ref < base_low_end)
            base_mask_high = (freqs_ref > base_high_start) & (freqs_ref <= base_high_end)
            baseline_vals = np.concatenate([weighted_psd[base_mask_low], weighted_psd[base_mask_high]])
            if baseline_vals.size == 0:
                baseline_power = np.nan
                baseline_median = np.nan
            else:
                baseline_power = baseline_vals.mean()
                baseline_median = np.median(baseline_vals)
            ratio = band_power / (baseline_power + 1e-12) if not np.isnan(baseline_power) else np.nan
            ratio_peak = band_peak_power / (baseline_median + 1e-12) if not np.isnan(baseline_median) else np.nan
            db_ratio = 10 * np.log10(ratio + 1e-12) if not np.isnan(ratio) else np.nan
            db_ratio_peak = 10 * np.log10(ratio_peak + 1e-12) if not np.isnan(ratio_peak) else np.nan
            records.append({
                'participant_id': pid,
                'run': run_id,
                'lab': lab,
                'channel': channel,
                'band_low': band_low,
                'band_high': band_high,
                'interference_band_power': band_power,
                'interference_band_peak_power': band_peak_power,
                'baseline_power': baseline_power,
                'baseline_median': baseline_median,
                'interference_ratio': ratio,
                'interference_ratio_peak': ratio_peak,
                'interference_db_ratio': db_ratio,
                'interference_db_ratio_peak': db_ratio_peak,
                'interference_fraction_total': band_sum / (total_sum + 1e-12),
                'n_freq_points_band': band_mask.sum(),
                'n_freq_points_baseline': base_mask_low.sum() + base_mask_high.sum()
            })
            # Store full PSD up to Nyquist
            full_psds.append({
                'participant_id': pid,
                'run': run_id,
                'lab': lab,
                'channel': channel,
                'freqs': freqs_ref,
                'psd': weighted_psd,
                'ratio': ratio
            })
            # Store snippet 40-60 Hz for overlay
            snippet_mask = (freqs_ref >= 40) & (freqs_ref <= 60)
            if snippet_mask.any():
                psd_snippets.append({
                    'participant_id': pid,
                    'lab': lab,
                    'channel': channel,
                    'freqs': freqs_ref[snippet_mask],
                    'psd': weighted_psd[snippet_mask],
                    'ratio': ratio
                })
    if not records:
        print("No 50Hz interference data collected.")
        return

    df = pd.DataFrame(records)
    df.to_csv(out_dir / 'interference_50hz_raw.csv', index=False)

    # Flag outliers using IQR on interference_ratio
    ratio_series = df['interference_ratio'].replace([np.inf, -np.inf], np.nan).dropna()
    q1, q3 = ratio_series.quantile([0.25, 0.75])
    iqr = q3 - q1
    upper_fence = q3 + 1.5 * iqr
    df['is_outlier'] = df['interference_ratio'] > upper_fence
    df.to_csv(out_dir / 'interference_50hz_with_outliers.csv', index=False)

    summary = df.groupby('lab').agg(
        mean_ratio=('interference_ratio', 'mean'),
        median_ratio=('interference_ratio', 'median'),
        max_ratio=('interference_ratio', 'max'),
        outlier_count=('is_outlier', 'sum'),
        recordings=('interference_ratio', 'count')
    ).reset_index()
    summary.to_csv(out_dir / 'interference_50hz_summary_by_lab.csv', index=False)

    # Plot 1: Distribution of interference ratios
    plt.figure(figsize=(10,6))
    sns.histplot(df, x='interference_ratio', hue='lab', element='step', stat='density', common_norm=False, bins=40)
    plt.axvline(upper_fence, color='red', linestyle='--', label='Outlier threshold')
    plt.title(f'Distribution of {band_low:.1f}-{band_high:.1f} Hz Interference Ratios (local baseline)')
    plt.xlabel(f'Interference Ratio ({band_low:.1f}-{band_high:.1f} Hz / baseline)')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'interference_ratio_distribution.png', dpi=160)
    plt.close()

    # Plot 2: Boxplot per lab
    plt.figure(figsize=(8,6))
    sns.boxplot(data=df, x='lab', y='interference_ratio')
    sns.stripplot(data=df, x='lab', y='interference_ratio', color='black', size=3, alpha=0.5)
    plt.axhline(upper_fence, color='red', linestyle='--', label='Outlier threshold')
    plt.title(f'{band_low:.1f}-{band_high:.1f} Hz Interference Ratio by Lab')
    plt.ylabel('Interference Ratio')
    plt.tight_layout()
    plt.savefig(out_dir / 'interference_ratio_by_lab.png', dpi=160)
    plt.close()

    # Compute participant-level 0..1 normalization based on interference_fraction_total or ratio
    # Primary heatmap now uses fraction of total 0.5–60 Hz power (always 0..1)
    pivot_fraction = df.pivot_table(index='participant_id', columns='channel', values='interference_fraction_total', aggfunc='mean')
    plt.figure(figsize=(10, max(4, 0.25*len(pivot_fraction))))
    sns.heatmap(pivot_fraction, cmap='viridis', vmin=0, vmax=1, annot=False)
    plt.title(f'Fraction of Total Power in {band_low:.1f}-{band_high:.1f} Hz Band (0.5–60 Hz total)')
    plt.xlabel('Channel')
    plt.ylabel('Participant')
    plt.tight_layout()
    plt.savefig(out_dir / 'interference_fraction_heatmap.png', dpi=160)
    plt.close()

    # Retain previous ratio heatmap (clipped) for reference
    pivot = df.pivot_table(index='participant_id', columns='channel', values='interference_ratio', aggfunc='max')
    ratio_clip_high = np.nanpercentile(pivot.values.flatten(), 99)
    pivot_clipped = pivot.clip(upper=ratio_clip_high)
    plt.figure(figsize=(10, max(4, 0.25*len(pivot))))
    sns.heatmap(pivot_clipped, cmap='magma', annot=False)
    plt.title(f'Max Ratio (Band / Local Baseline) Clipped @99th pct')
    plt.xlabel('Channel')
    plt.ylabel('Participant')
    plt.tight_layout()
    plt.savefig(out_dir / 'interference_ratio_heatmap.png', dpi=160)
    plt.close()

    # Plot 3b: Log10 ratio heatmap (diverging around 0 -> ratio=1)
    log_pivot = np.log10(pivot.replace(0, np.nan))
    vmin = np.nanpercentile(log_pivot.values, 5)
    vmax = np.nanpercentile(log_pivot.values, 95)
    plt.figure(figsize=(10, max(4, 0.25*len(pivot))))
    sns.heatmap(log_pivot.clip(lower=vmin, upper=vmax), cmap='coolwarm', center=0, annot=False)
    plt.title(f'Log10 Interference Ratio Heatmap (center=0; 0=>ratio=1)')
    plt.xlabel('Channel')
    plt.ylabel('Participant')
    plt.tight_layout()
    plt.savefig(out_dir / 'interference_ratio_heatmap_log.png', dpi=160)
    plt.close()

    # Plot 3c: Peak ratio heatmap
    pivot_peak = df.pivot_table(index='participant_id', columns='channel', values='interference_ratio_peak', aggfunc='max')
    peak_clip = np.nanpercentile(pivot_peak.values.flatten(), 99)
    plt.figure(figsize=(10, max(4, 0.25*len(pivot_peak))))
    sns.heatmap(pivot_peak.clip(upper=peak_clip), cmap='viridis', annot=False)
    plt.title(f'Peak Ratio (Peak / Baseline Median) {band_low:.1f}-{band_high:.1f} Hz')
    plt.xlabel('Channel')
    plt.ylabel('Participant')
    plt.tight_layout()
    plt.savefig(out_dir / 'interference_peak_ratio_heatmap.png', dpi=160)
    plt.close()

    # Plot 3d: Absolute band peak power heatmap (log10)
    pivot_peak_power = df.pivot_table(index='participant_id', columns='channel', values='interference_band_peak_power', aggfunc='max')
    log_peak_power = np.log10(pivot_peak_power + 1e-12)
    vmax_lp = np.nanpercentile(log_peak_power.values, 99)
    vmin_lp = np.nanpercentile(log_peak_power.values, 5)
    plt.figure(figsize=(10, max(4, 0.25*len(pivot_peak_power))))
    sns.heatmap(log_peak_power.clip(lower=vmin_lp, upper=vmax_lp), cmap='magma', annot=False)
    plt.title('Log10 Peak Band Power (Absolute)')
    plt.xlabel('Channel')
    plt.ylabel('Participant')
    plt.tight_layout()
    plt.savefig(out_dir / 'interference_peak_power_heatmap.png', dpi=160)
    plt.close()

    # Plot 4: Overlay of PSD snippets (40-60 Hz) highlighting interference band
    if psd_snippets:
        plt.figure(figsize=(10,6))
        for snip in psd_snippets[:300]:  # cap for clarity
            freqs = snip['freqs']
            psd = snip['psd']
            color = 'red' if snip['ratio'] > upper_fence else 'gray'
            alpha = 0.15 if color=='gray' else 0.7
            plt.plot(freqs, psd, color=color, alpha=alpha, linewidth=1 if color=='gray' else 1.5)
        plt.axvspan(band_low, band_high, color='yellow', alpha=0.2, label=f'{band_low:.1f}-{band_high:.1f} Hz band')
        plt.title(f'PSD Overlays (40-60 Hz) Highlighting {band_low:.1f}-{band_high:.1f} Hz Interference')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectral Density (a.u.)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / 'psd_overlays_40_60Hz.png', dpi=170)
        plt.close()

    # Plot 5: Outlier ratio bar chart
    outlier_counts = df.groupby('lab')['is_outlier'].sum().reindex(summary['lab'])
    plt.figure(figsize=(8,5))
    sns.barplot(x=summary['lab'], y=outlier_counts.values)
    plt.title('Count of 50 Hz Interference Outlier Recordings per Lab')
    plt.ylabel('Outlier Recordings')
    plt.xlabel('Lab')
    plt.tight_layout()
    plt.savefig(out_dir / 'interference_outlier_counts_by_lab.png', dpi=160)
    plt.close()

    # Additional PSD plots (log-domain, 0.5-60 Hz)
    if full_psds:
        # Choose reference frequency grid
        ref_freqs = full_psds[0]['freqs']
        psd_arrays = []
        labs_list = []
        ratios_list = []
        for rec in full_psds:
            freqs_cur = rec['freqs']
            psd_cur = rec['psd']
            if len(freqs_cur) != len(ref_freqs) or not np.allclose(freqs_cur, ref_freqs):
                try:
                    psd_cur = np.interp(ref_freqs, freqs_cur, psd_cur)
                except Exception:
                    continue
            psd_arrays.append(psd_cur)
            labs_list.append(rec['lab'])
            ratios_list.append(rec['ratio'])
        if psd_arrays:
            psd_stack = np.vstack(psd_arrays)
            mean_psd = psd_stack.mean(axis=0)
            # Prepare log-domain conversion
            eps = 1e-12
            freq_mask = (ref_freqs >= 0.5) & (ref_freqs <= 60)
            f_plot = ref_freqs[freq_mask]
            mean_psd_db = 10 * np.log10(mean_psd[freq_mask] + eps)

            # Overall mean PSD 0.5-60 Hz (dB)
            plt.figure(figsize=(10,6))
            plt.plot(f_plot, mean_psd_db, label='Mean PSD (all)')
            plt.axvspan(max(band_low, 0.5), min(band_high, 60), color='orange', alpha=0.2, label='Interference band')
            if 50 >= 0.5 and 50 <= 60:
                plt.axvline(50, color='red', linestyle='--', alpha=0.7, label='50 Hz')
            plt.title('Mean PSD (All Channels) 0.5–60 Hz (dB)')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('PSD (dB)')
            plt.xlim(0.5, 60)
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / 'mean_psd_full.png', dpi=170)
            plt.close()

            # Per-lab mean PSD (dB)
            plt.figure(figsize=(10,6))
            for lab_name in sorted(set(labs_list)):
                lab_mask = [l == lab_name for l in labs_list]
                if any(lab_mask):
                    lab_mean = psd_stack[lab_mask].mean(axis=0)
                    lab_mean_db = 10 * np.log10(lab_mean[freq_mask] + eps)
                    plt.plot(f_plot, lab_mean_db, label=lab_name, linewidth=1.2)
            plt.axvspan(max(band_low, 0.5), min(band_high, 60), color='orange', alpha=0.15)
            if 50 >= 0.5 and 50 <= 60:
                plt.axvline(50, color='red', linestyle='--', alpha=0.5)
            plt.title('Mean PSD by Lab 0.5–60 Hz (dB)')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('PSD (dB)')
            plt.xlim(0.5, 60)
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(out_dir / 'mean_psd_by_lab.png', dpi=170)
            plt.close()

            # Outlier vs non-outlier mean PSD (if both present)
            if df['is_outlier'].any() and (~df['is_outlier']).any():
                # Map rows to PSD entries (same order as records added) using index alignment
                outlier_flags = []
                for rec in full_psds:
                    flag_row = df[(df.participant_id == rec['participant_id']) &
                                  (df.channel == rec['channel']) & (df.run == rec['run'])]
                    outlier_flags.append(bool(flag_row.iloc[0]['is_outlier'])) if not flag_row.empty else outlier_flags.append(False)
                outlier_mask = np.array(outlier_flags)
                if outlier_mask.any() and (~outlier_mask).any():
                    mean_out = psd_stack[outlier_mask].mean(axis=0)
                    mean_in = psd_stack[~outlier_mask].mean(axis=0)
                    mean_out_db = 10 * np.log10(mean_out[freq_mask] + eps)
                    mean_in_db = 10 * np.log10(mean_in[freq_mask] + eps)
                    plt.figure(figsize=(10,6))
                    plt.plot(f_plot, mean_in_db, label='Non-outlier mean', color='steelblue')
                    plt.plot(f_plot, mean_out_db, label='Outlier mean', color='crimson')
                    plt.axvspan(max(band_low, 0.5), min(band_high, 60), color='orange', alpha=0.15)
                    if 50 >= 0.5 and 50 <= 60:
                        plt.axvline(50, color='red', linestyle='--', alpha=0.5)
                    plt.title('Mean PSD: Outlier vs Non-Outlier 0.5–60 Hz (dB)')
                    plt.xlabel('Frequency (Hz)')
                    plt.ylabel('PSD (dB)')
                    plt.xlim(0.5, 60)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(out_dir / 'mean_psd_outlier_vs_non.png', dpi=170)
                    plt.close()

    print(f"50Hz interference analysis completed. Results saved to: {out_dir}")

if __name__ == "__main__":
    analyze_50hz_interference()
