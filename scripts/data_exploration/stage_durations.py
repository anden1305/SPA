"""Sleep stage duration analysis.
Computes per-recording absolute durations and proportions of each sleep stage
and produces three plots:
 1. Distribution of absolute stage durations (minutes) per recording
 2. Distribution of stage proportions (excluding Artifact) per recording
 3. ECDF (cumulative distribution) of stage proportions (excluding Artifact)
Outputs CSV summaries with per-recording and aggregated statistics.
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
    STAGE_ORDER,
    SLEEP_STAGE_MAPPING,
    SLEEP_STAGE_COLORS,
)

plt.style.use('seaborn-v0_8')


def analyze_stage_durations(output_dir="results/data_exploration", exclude_lab1=True, sampling_rate=128):
    print("="*60)
    print("ANALYZING SLEEP STAGE DURATIONS")
    print("="*60)

    output_dir = ensure_output_directory(output_dir)
    out_dir = Path(output_dir) / "stage_durations"
    out_dir.mkdir(exist_ok=True)

    metadata = load_metadata()
    if exclude_lab1:
        before = len(metadata)
        metadata = metadata[metadata['lab'] != 'lab_1']
        print(f"Excluded lab_1: {before - len(metadata)} records removed")

    records = []

    # Exclude Artifact stage entirely (global rule: only artifact_characteristics keeps it)
    stage_order = [s for s in STAGE_ORDER if s != 'Artifact']
    stage_numeric_map = {v: k for k, v in SLEEP_STAGE_MAPPING.items()}  # name->code reverse

    for (pid, run_id), group in metadata.groupby(['participant_id', 'run']):
        try:
            data = load_participant_data(pid, run_id)
        except Exception as e:
            print(f"  Skipping {pid} run {run_id}: load error {e}")
            continue
        labels = data.get('labels')
        if labels is None or len(labels) == 0:
            continue

        # Count samples by numeric stage code
        unique, counts = np.unique(labels, return_counts=True)
        sample_counts = dict(zip(unique, counts))

        # Build row
        row = {
            'participant_id': pid,
            'run': run_id,
            'lab': group.iloc[0]['lab'],
            'total_samples': len(labels),
            'total_minutes': len(labels)/(sampling_rate*60)
        }

        # Absolute durations & minutes (excluding Artifact)
        for stage_name in stage_order:
            code = stage_numeric_map.get(stage_name)
            scount = sample_counts.get(code, 0)
            row[f"{stage_name}_samples"] = scount
            row[f"{stage_name}_minutes"] = scount/(sampling_rate*60)
        records.append(row)

    if not records:
        print("No recordings processed for stage durations.")
        return

    df = pd.DataFrame(records)

    # Compute proportions (Artifact already excluded)
    stage_minute_cols = [f"{s}_minutes" for s in stage_order]
    df['total_stage_minutes'] = df[stage_minute_cols].sum(axis=1)
    for s in stage_order:
        df[f"{s}_prop"] = df[f"{s}_minutes"] / df['total_stage_minutes']

    # Summary stats per stage
    summary_rows = []
    for s in stage_order:
        minutes_col = f"{s}_minutes"
        prop_col = f"{s}_prop"
        vals_minutes = df[minutes_col]
        vals_prop = df[prop_col].dropna()
        summary_rows.append({
            'stage': s,
            'mean_minutes': vals_minutes.mean(),
            'median_minutes': vals_minutes.median(),
            'std_minutes': vals_minutes.std(),
            'mean_prop': vals_prop.mean() if len(vals_prop)>0 else np.nan,
            'median_prop': vals_prop.median() if len(vals_prop)>0 else np.nan,
            'std_prop': vals_prop.std() if len(vals_prop)>0 else np.nan,
            'n_records': len(df)
        })
    summary_df = pd.DataFrame(summary_rows)

    # Save CSVs
    df.to_csv(out_dir / 'stage_durations_per_recording.csv', index=False)
    summary_df.to_csv(out_dir / 'stage_durations_summary.csv', index=False)

    # Long format for plotting absolute durations
    abs_long = df.melt(id_vars=['participant_id','run','lab'],
                       value_vars=[f"{s}_minutes" for s in stage_order],
                       var_name='stage', value_name='minutes')
    abs_long['stage'] = abs_long['stage'].str.replace('_minutes','')

    # Plot 1: Absolute duration distribution
    plt.figure(figsize=(8,5))
    sns.boxplot(data=abs_long, x='stage', y='minutes', palette=[SLEEP_STAGE_COLORS[s] for s in stage_order], showfliers=False)
    sns.stripplot(data=abs_long, x='stage', y='minutes', color='black', size=2, alpha=0.4)
    plt.title('Sleep Stage Durations (Minutes) per Recording')
    plt.ylabel('Minutes')
    plt.xlabel('Stage')
    plt.tight_layout()
    plt.savefig(out_dir / 'stage_duration_distribution.png', dpi=160)
    plt.close()

    # Long format for proportions (excluding artifact)
    prop_cols = [f"{s}_prop" for s in stage_order]
    prop_long = df.melt(id_vars=['participant_id','run','lab'], value_vars=prop_cols, var_name='stage', value_name='proportion')
    prop_long['stage'] = prop_long['stage'].str.replace('_prop','')

    # Plot 2: Proportion distribution (excluding Artifact)
    plt.figure(figsize=(8,5))
    sns.violinplot(data=prop_long, x='stage', y='proportion', inner='box', palette=[SLEEP_STAGE_COLORS[s] for s in stage_order])
    plt.title('Distribution of Stage Proportions (Excl. Artifact)')
    plt.ylabel('Proportion of Recording')
    plt.xlabel('Stage')
    plt.ylim(0,1)
    plt.tight_layout()
    plt.savefig(out_dir / 'stage_proportion_distribution.png', dpi=160)
    plt.close()

    # Plot 3: ECDF of stage proportions
    plt.figure(figsize=(7,5))
    for s in stage_order:
        vals = df[f"{s}_prop"].dropna().sort_values()
        if len(vals)==0:
            continue
        y = np.linspace(0,1,len(vals))
        plt.step(vals, y, where='post', label=s, color=SLEEP_STAGE_COLORS[s])
    plt.title('ECDF of Stage Proportions Across Recordings')
    plt.xlabel('Proportion of Recording')
    plt.ylabel('Cumulative Fraction of Recordings')
    plt.legend(title='Stage')
    plt.tight_layout()
    plt.savefig(out_dir / 'stage_proportion_ecdf.png', dpi=160)
    plt.close()

    # ============================================================
    # Episode (Bout) Length Analysis
    # ============================================================
    print("Calculating episode (bout) length distributions ...")
    episode_rows = []
    per_record_episode_stats = []
    code_to_name = SLEEP_STAGE_MAPPING  # numeric -> name

    # Re-iterate metadata to load labels once more (keeps memory usage modest)
    for (pid, run_id), group in metadata.groupby(['participant_id', 'run']):
        try:
            data = load_participant_data(pid, run_id)
        except Exception:
            continue
        labels = data.get('labels')
        if labels is None or len(labels) == 0:
            continue

        # Traverse contiguous segments
        current_label = labels[0]
        start_idx = 0
        episode_index = 0
        per_stage_durations = {stage: [] for stage in stage_order}
        for i in range(1, len(labels)):
            if labels[i] != current_label:
                length_samples = i - start_idx
                duration_sec = length_samples / sampling_rate
                duration_min = duration_sec / 60.0
                stage_name = code_to_name.get(current_label, f"Unknown_{current_label}")
                episode_rows.append({
                    'participant_id': pid,
                    'run': run_id,
                    'lab': group.iloc[0]['lab'],
                    'stage': stage_name,
                    'episode_index': episode_index,
                    'samples': length_samples,
                    'duration_seconds': duration_sec,
                    'duration_minutes': duration_min
                })
                per_stage_durations.setdefault(stage_name, []).append(duration_min)
                episode_index += 1
                current_label = labels[i]
                start_idx = i
        # Handle final episode
        length_samples = len(labels) - start_idx
        duration_sec = length_samples / sampling_rate
        duration_min = duration_sec / 60.0
        stage_name = code_to_name.get(current_label, f"Unknown_{current_label}")
        episode_rows.append({
            'participant_id': pid,
            'run': run_id,
            'lab': group.iloc[0]['lab'],
            'stage': stage_name,
            'episode_index': episode_index,
            'samples': length_samples,
            'duration_seconds': duration_sec,
            'duration_minutes': duration_min
        })
        per_stage_durations.setdefault(stage_name, []).append(duration_min)

        # Per-record summary of episode means
        for stage in stage_order:
            durations_list = per_stage_durations.get(stage, [])
            if durations_list:
                per_record_episode_stats.append({
                    'participant_id': pid,
                    'run': run_id,
                    'lab': group.iloc[0]['lab'],
                    'stage': stage,
                    'mean_episode_minutes': float(np.mean(durations_list)),
                    'median_episode_minutes': float(np.median(durations_list)),
                    'std_episode_minutes': float(np.std(durations_list)),
                    'n_episodes': len(durations_list)
                })

    if episode_rows:
        episodes_df = pd.DataFrame(episode_rows)
        episodes_df.to_csv(out_dir / 'episode_lengths_all.csv', index=False)
        per_record_episode_df = pd.DataFrame(per_record_episode_stats)
        per_record_episode_df.to_csv(out_dir / 'episode_lengths_per_recording_summary.csv', index=False)

        # Base aggregated stats with additional quantiles
        quantiles = episodes_df.groupby('stage')['duration_minutes'].quantile([0.10,0.25,0.50,0.75,0.90,0.95,0.99]).unstack()
        stage_episode_stats = episodes_df.groupby('stage')['duration_minutes'].agg(['mean','median','std','count','min','max']).join(quantiles)
        stage_episode_stats = stage_episode_stats.reindex(stage_order)
        stage_episode_stats.rename(columns={0.10:'p10',0.25:'p25',0.50:'p50',0.75:'p75',0.90:'p90',0.95:'p95',0.99:'p99'}, inplace=True)
        stage_episode_stats.to_csv(out_dir / 'episode_length_stats_extended.csv')

        # Plot A (raw mean ± SD)
        plt.figure(figsize=(7,5))
        bars = plt.bar(stage_episode_stats.index, stage_episode_stats['mean'], yerr=stage_episode_stats['std'], capsize=4,
                        color=[SLEEP_STAGE_COLORS[s] for s in stage_episode_stats.index], edgecolor='black', alpha=0.85)
        plt.ylabel('Mean Episode Length (minutes)')
        plt.title('Mean Sleep Stage Episode Length ±1 SD (Raw, Skewed)')
        for b, m in zip(bars, stage_episode_stats['mean']):
            plt.text(b.get_x()+b.get_width()/2, b.get_height(), f"{m:.2f}", ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        plt.savefig(out_dir / 'mean_episode_length_bar_raw.png', dpi=160)
        plt.close()

        # Plot A2 (median with IQR)
        med = stage_episode_stats['p50']
        q1 = stage_episode_stats['p25']
        q3 = stage_episode_stats['p75']
        lower_err = (med - q1).values
        upper_err = (q3 - med).values
        asym_err = np.vstack([lower_err, upper_err])
        plt.figure(figsize=(7,5))
        bars = plt.bar(stage_episode_stats.index, med, yerr=asym_err, capsize=5,
                        color=[SLEEP_STAGE_COLORS[s] for s in stage_episode_stats.index], edgecolor='black', alpha=0.9)
        plt.ylabel('Median Episode Length (minutes)')
        plt.title('Median Sleep Stage Episode Length with IQR')
        for b, m in zip(bars, med):
            plt.text(b.get_x()+b.get_width()/2, m, f"{m:.2f}", ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        plt.savefig(out_dir / 'median_episode_length_IQR_bar.png', dpi=160)
        plt.close()

        # Plot B: Distribution (violin) of episode lengths (log-scale)
        plot_eps = episodes_df.copy()
        plot_eps['duration_minutes_clipped'] = plot_eps['duration_minutes'].clip(upper=180)
        plt.figure(figsize=(8,5))
        sns.violinplot(data=plot_eps, x='stage', y='duration_minutes_clipped', palette=[SLEEP_STAGE_COLORS[s] for s in stage_order], cut=0, inner='quartile')
        plt.yscale('log')
        plt.ylabel('Episode Length (minutes, log scale)')
        plt.title('Distribution of Episode (Bout) Lengths by Stage')
        plt.tight_layout()
        plt.savefig(out_dir / 'episode_length_distribution.png', dpi=160)
        plt.close()
    else:
        print("Warning: No episode data generated.")

    print(f"Stage duration analysis complete. Processed {len(df)} recordings.")
    print(f"Results saved to: {out_dir}")
if __name__ == '__main__':
    analyze_stage_durations()
