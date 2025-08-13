"""Sleep stage transition analysis.
Computes and visualizes transition statistics across all recordings.
Generates three key plots:
 1. Full transition probability matrix (including self-transitions)
 2. Change-only transition probability matrix (excluding self-transitions)
 3. Top N most frequent stage-to-stage changes (bar plot)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from helpers import load_metadata, load_participant_data, calculate_stage_transitions, ensure_output_directory, STAGE_ORDER, SLEEP_STAGE_MAPPING

plt.style.use('seaborn-v0_8')


def analyze_stage_transitions(output_dir="results/data_exploration", exclude_lab1=True, top_n=10):
    print("="*60)
    print("ANALYZING SLEEP STAGE TRANSITIONS")
    print("="*60)
    output_dir = ensure_output_directory(output_dir)
    out_dir = Path(output_dir) / "stage_transitions"
    out_dir.mkdir(exist_ok=True)

    metadata = load_metadata()
    if exclude_lab1:
        before = len(metadata)
        metadata = metadata[metadata['lab'] != 'lab_1']
        print(f"Excluded lab_1: {before - len(metadata)} records removed")

    # Prepare aggregation containers
    stage_index = {name: i for i, name in enumerate(STAGE_ORDER)}
    n = len(STAGE_ORDER)
    agg_counts = np.zeros((n, n), dtype=np.float64)
    agg_change_counts = np.zeros((n, n), dtype=np.float64)
    total_recordings = 0

    per_record_summary = []

    for (pid, run_id), group in metadata.groupby(['participant_id', 'run']):
        try:
            data = load_participant_data(pid, run_id)
        except Exception as e:
            print(f"  Skipping {pid} run {run_id}: load error {e}")
            continue
        labels = data.get('labels')
        if labels is None or len(labels) < 2:
            continue
        try:
            trans = calculate_stage_transitions(labels)
        except Exception as e:
            print(f"  Transition calc failed {pid} run {run_id}: {e}")
            continue
        rec_counts = trans['transition_counts']
        rec_change_counts = trans['change_only_counts']
        rec_stage_names = trans['stage_names']  # subset ordering

        # Map into global matrices
        # Build index mapping from rec subset to global indices
        rec_indices = [stage_index[name] for name in rec_stage_names]
        # Add counts
        for local_i, global_i in enumerate(rec_indices):
            for local_j, global_j in enumerate(rec_indices):
                agg_counts[global_i, global_j] += rec_counts[local_i, local_j]
                agg_change_counts[global_i, global_j] += rec_change_counts[local_i, local_j]
        total_recordings += 1

        per_record_summary.append({
            'participant_id': pid,
            'run': run_id,
            'lab': group.iloc[0]['lab'],
            'total_transitions': trans['total_transitions'],
            'total_changes': trans['total_changes']
        })

    if total_recordings == 0:
        print("No recordings processed for transitions.")
        return

    # Compute probabilities
    row_sums = agg_counts.sum(axis=1, keepdims=True)
    prob_matrix = np.divide(agg_counts, row_sums, out=np.zeros_like(agg_counts), where=row_sums>0)

    change_row_sums = agg_change_counts.sum(axis=1, keepdims=True)
    change_prob_matrix = np.divide(agg_change_counts, change_row_sums, out=np.zeros_like(agg_change_counts), where=change_row_sums>0)

    stages = STAGE_ORDER

    # Save CSV outputs
    counts_df = pd.DataFrame(agg_counts, index=stages, columns=stages)
    counts_df.to_csv(out_dir / 'transition_counts.csv')
    prob_df = pd.DataFrame(prob_matrix, index=stages, columns=stages)
    prob_df.to_csv(out_dir / 'transition_probabilities.csv')
    change_counts_df = pd.DataFrame(agg_change_counts, index=stages, columns=stages)
    change_counts_df.to_csv(out_dir / 'change_only_counts.csv')
    change_prob_df = pd.DataFrame(change_prob_matrix, index=stages, columns=stages)
    change_prob_df.to_csv(out_dir / 'change_only_probabilities.csv')
    pd.DataFrame(per_record_summary).to_csv(out_dir / 'per_record_transition_summary.csv', index=False)

    # Derive top transitions (exclude self) from change-only counts
    top_pairs = []
    for i, src in enumerate(stages):
        for j, dst in enumerate(stages):
            if i == j:
                continue
            count = agg_change_counts[i, j]
            if count > 0:
                prob = change_prob_matrix[i, j]
                top_pairs.append({'source': src, 'target': dst, 'count': count, 'probability': prob})
    top_pairs_sorted = sorted(top_pairs, key=lambda x: x['count'], reverse=True)[:top_n]
    pd.DataFrame(top_pairs_sorted).to_csv(out_dir / 'top_transitions.csv', index=False)

    # Plot 1: Full transition probability matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(prob_df, annot=True, fmt='.2f', cmap='Blues', cbar_kws={'label':'Probability'})
    plt.title('Sleep Stage Transition Probabilities (Including Self)')
    plt.xlabel('To Stage')
    plt.ylabel('From Stage')
    plt.tight_layout()
    plt.savefig(out_dir / 'transition_probability_matrix.png', dpi=160)
    plt.close()

    # Plot 2: Change-only transition probability matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(change_prob_df, annot=True, fmt='.2f', cmap='PuBuGn', cbar_kws={'label':'Probability'})
    plt.title('Change-Only Transition Probabilities (Excluding Self)')
    plt.xlabel('To Stage')
    plt.ylabel('From Stage')
    plt.tight_layout()
    plt.savefig(out_dir / 'change_only_transition_probability_matrix.png', dpi=160)
    plt.close()

    # Plot 3: Top N transitions (bar)
    if top_pairs_sorted:
        top_df = pd.DataFrame(top_pairs_sorted)
        top_df['pair'] = top_df['source'] + '→' + top_df['target']
        plt.figure(figsize=(min(12, 1+0.8*len(top_df)), 5))
        sns.barplot(data=top_df, x='pair', y='count', palette='viridis')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Change Count')
        plt.xlabel('Transition')
        plt.title(f'Top {len(top_df)} Sleep Stage Changes')
        for idx, row in top_df.iterrows():
            plt.text(idx, row['count'], f"{row['probability']:.2f}", ha='center', va='bottom', fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / 'top_transitions_bar.png', dpi=160)
        plt.close()

    print(f"Stage transition analysis complete. Processed {total_recordings} recordings.")
    print(f"Results saved to: {out_dir}")

if __name__ == '__main__':
    analyze_stage_transitions()
