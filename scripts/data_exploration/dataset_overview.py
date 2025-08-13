"""General dataset overview: labs, participants, runs, signals.
Generates summary tables and five core plots.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from helpers import load_metadata, load_participant_data, ensure_output_directory

plt.style.use('seaborn-v0_8')


def analyze_dataset_overview(output_dir="results/data_exploration", sampling_rate=128, max_participants_scan=500):
    print("="*60)
    print("DATASET OVERVIEW")
    print("="*60)
    out_base = Path(ensure_output_directory(output_dir)) / "dataset_overview"
    out_base.mkdir(exist_ok=True)

    meta = load_metadata()
    meta['run'] = meta['run'].astype(int)

    # 1. Basic counts
    n_labs = meta['lab'].nunique()
    n_participants = meta['participant_id'].nunique()
    runs_per_participant = meta.groupby('participant_id')['run'].nunique()
    total_runs = len(meta)
    participants_per_lab = meta.groupby('lab')['participant_id'].nunique().sort_values(ascending=False)
    runs_per_lab = meta.groupby('lab')['run'].count().sort_values(ascending=False)

    basic_df = pd.DataFrame({
        'metric': [
            'n_labs','n_participants','total_runs','median_runs_per_participant','mean_runs_per_participant'
        ],
        'value': [
            n_labs, n_participants, total_runs, runs_per_participant.median(), runs_per_participant.mean()
        ]
    })
    basic_df.to_csv(out_base / 'basic_counts.csv', index=False)

    participants_per_lab.to_csv(out_base / 'participants_per_lab.csv', header=['n_participants'])
    runs_per_lab.to_csv(out_base / 'runs_per_lab.csv', header=['n_runs'])

    # 2. Scan subset for signal availability (avoid loading everything)
    sampled_meta = meta.groupby('participant_id').head(1).head(max_participants_scan)
    signal_presence = []
    unique_signals = set()
    for _, row in sampled_meta.iterrows():
        pid, run_id = row['participant_id'], row['run']
        try:
            data = load_participant_data(pid, run_id)
        except Exception:
            continue
        present = [k for k in data.keys() if k != 'labels']
        for sig in present:
            unique_signals.add(sig)
        signal_presence.append({'participant_id': pid, 'run': run_id, **{sig: (sig in present) for sig in ['EEG1','EEG2','EEG3','EEG4','EMG']}})

    signal_df = pd.DataFrame(signal_presence)
    if not signal_df.empty:
        signal_df.to_csv(out_base / 'signal_presence_sample.csv', index=False)

    signal_counts = {sig: (signal_df[sig].sum() if sig in signal_df else 0) for sig in ['EEG1','EEG2','EEG3','EEG4','EMG']}
    signal_counts_df = pd.DataFrame([signal_counts])
    signal_counts_df.to_csv(out_base / 'signal_counts_sample.csv', index=False)

    # Derived table summarizing participants per #runs bucket
    # Distribution of number of runs per participant
    run_bucket = runs_per_participant.value_counts().sort_index()
    run_bucket_df = run_bucket.reset_index()
    run_bucket_df.columns = ['n_runs','n_participants']
    run_bucket_df.to_csv(out_base / 'participants_by_run_count.csv', index=False)

    # ---------------- Plots (5) ----------------
    sns.set_palette('viridis')

    # Plot 1: Participants per lab
    plt.figure(figsize=(8,4))
    participants_per_lab.plot(kind='bar')
    plt.ylabel('Participants')
    plt.title('Participants per Lab')
    plt.tight_layout()
    plt.savefig(out_base / 'participants_per_lab.png', dpi=160)
    plt.close()

    # Plot 2: Runs per lab
    plt.figure(figsize=(8,4))
    runs_per_lab.plot(kind='bar', color='teal')
    plt.ylabel('Runs')
    plt.title('Runs per Lab')
    plt.tight_layout()
    plt.savefig(out_base / 'runs_per_lab.png', dpi=160)
    plt.close()

    # Plot 3: Distribution of runs per participant
    plt.figure(figsize=(6,4))
    sns.histplot(runs_per_participant, bins=range(1, runs_per_participant.max()+2), kde=False, color='#4ECDC4', edgecolor='black')
    plt.xlabel('Runs per Participant')
    plt.ylabel('Count of Participants')
    plt.title('Distribution of Runs per Participant')
    plt.tight_layout()
    plt.savefig(out_base / 'runs_per_participant_distribution.png', dpi=160)
    plt.close()

    # Plot 4: Signal availability heatmap (sample)
    if not signal_df.empty:
        plt.figure(figsize=(6,4))
        heat_data = signal_df[['EEG1','EEG2','EEG3','EEG4','EMG']].mean().to_frame(name='availability')
        sns.heatmap(heat_data, annot=True, fmt='.2f', cmap='YlGnBu', cbar=False)
        plt.title('Signal Availability Fraction (Sample)')
        plt.tight_layout()
        plt.savefig(out_base / 'signal_availability_heatmap.png', dpi=160)
        plt.close()

    # Plot 5: Participants by number of runs
    if not run_bucket_df.empty:
        plt.figure(figsize=(6,4))
        sns.barplot(data=run_bucket_df, x='n_runs', y='n_participants', color='#FF6B6B')
        plt.xlabel('Number of Runs')
        plt.ylabel('Participants')
        plt.title('Participants by Run Count')
        for i, row in run_bucket_df.iterrows():
            plt.text(i, row['n_participants'], int(row['n_participants']), ha='center', va='bottom', fontsize=8)
        plt.tight_layout()
        plt.savefig(out_base / 'participants_by_run_count.png', dpi=160)
        plt.close()
    else:
        print("Warning: run_bucket_df is empty; skipping participants_by_run_count plot.")

    # Combined overview summary markdown
    overview_lines = [
        f"Number of labs: {n_labs}",
        f"Number of participants: {n_participants}",
        f"Total runs: {total_runs}",
        f"Median runs per participant: {runs_per_participant.median():.1f}",
        f"Mean runs per participant: {runs_per_participant.mean():.2f}",
        f"Signals (sampled): {', '.join(sorted(unique_signals)) if unique_signals else 'N/A'}"
    ]
    (out_base / 'overview_summary.txt').write_text("\n".join(overview_lines))
    print("\n".join(overview_lines))
    print(f"Dataset overview tables & plots written to {out_base}")

if __name__ == '__main__':
    analyze_dataset_overview()
