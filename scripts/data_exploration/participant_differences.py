"""
Analysis of signal differences between participants.
This module focuses on identifying and visualizing the key signal characteristics
that distinguish different participants and their individual sleep patterns.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from helpers import (
    load_metadata, load_participant_data, calculate_signal_statistics,
    plot_signal_statistics_comparison, create_summary_table, ensure_output_directory,
    SLEEP_STAGE_MAPPING, SLEEP_STAGE_COLORS, calculate_spectral_confidence_intervals,
    STAGE_ORDER
)

def analyze_participant_differences(output_dir="results/data_exploration_2"):
    """
    Comprehensive analysis of signal differences between participants.
    
    Args:
        output_dir (str): Directory to save results
    """
    print("=" * 60)
    print("ANALYZING SIGNAL DIFFERENCES BETWEEN PARTICIPANTS")
    print("=" * 60)
    
    # Ensure output directory exists
    output_dir = ensure_output_directory(output_dir)
    participant_output_dir = Path(output_dir) / "participant_differences"
    participant_output_dir.mkdir(exist_ok=True)
    
    # Load metadata
    metadata = load_metadata()
    # Exclude lab_1 (outlier) from participant analysis
    initial_rows = metadata.shape[0]
    metadata = metadata[metadata['lab'] != 'lab_1']
    removed_rows = initial_rows - metadata.shape[0]
    print(f"Excluded lab_1 from participant differences analysis (removed {removed_rows} records).")
    
    # Select diverse participants from different labs for comprehensive analysis
    selected_participants = select_representative_participants(metadata)
    print(f"Selected participants for analysis: {[p[0] for p in selected_participants]}")
    
    # Collect comprehensive data from all participants
    all_participant_stats = {}
    
    for participant_id, lab in selected_participants:
        print(f"\nProcessing {participant_id} from {lab}...")
        participant_runs = metadata[metadata['participant_id'] == participant_id]
        
        participant_data = {}
        
        for _, run_info in participant_runs.iterrows():
            run_id = run_info['run']
            
            try:
                # Load participant data
                data = load_participant_data(participant_id, run_id)
                labels = data['labels']
                
                # Process each signal type
                for signal_name, signal_data in data.items():
                    if signal_name == 'labels':
                        continue
                    
                    print(f"  Analyzing {signal_name}...")
                    
                    # Calculate statistics for the entire signal (will be separated by stage inside function)
                    stats_result = calculate_signal_statistics(signal_data, labels)
                    
                    # Store statistics for each stage
                    for stage_name, stage_stats in stats_result.items():
                        key = f"{participant_id}_{signal_name}_{stage_name}"
                        participant_data[key] = {
                            'participant_id': participant_id,
                            'lab': lab,
                            'signal': signal_name,
                            'stage': stage_name,
                            **stage_stats
                        }
                
                all_participant_stats.update(participant_data)
                
            except Exception as e:
                print(f"    Error processing {participant_id} run {run_id}: {e}")
                continue
    
    if not all_participant_stats:
        print("No valid data found for analysis!")
        return
    
    print(f"\nCollected data from {len(selected_participants)} participants")
    
    # Create visualizations
    create_participant_signal_overview(all_participant_stats, participant_output_dir)
    create_participant_frequency_analysis(all_participant_stats, participant_output_dir)
    create_participant_sleep_profile_analysis(all_participant_stats, participant_output_dir)
    create_participant_spectral_profiles(all_participant_stats, participant_output_dir)
    create_participant_variability_analysis(all_participant_stats, participant_output_dir)
    
    # Create summary statistics
    create_participant_summary_statistics(all_participant_stats, participant_output_dir)
    
    print(f"\nParticipant difference analysis completed. Results saved to: {participant_output_dir}")


def select_representative_participants(metadata, max_participants=15):
    """
    Select representative participants for analysis, prioritizing single lab to control for lab differences.
    
    Args:
        metadata (pd.DataFrame): Participant metadata
        max_participants (int): Maximum number of participants to select
    
    Returns:
        list: List of (participant_id, lab) tuples
    """
    # Find the lab with the most participants
    lab_counts = metadata['lab'].value_counts()
    selected_lab = lab_counts.index[0]
    
    print(f"Selected {selected_lab} for within-lab analysis (has {lab_counts.iloc[0]} participants)")
    
    # Get participants from the selected lab
    lab_participants = metadata[metadata['lab'] == selected_lab]['participant_id'].unique()
    
    # Select up to max_participants
    n_select = min(max_participants, len(lab_participants))
    selected_participants = sorted(lab_participants)[:n_select]
    
    # Return with lab information
    result = [(participant_id, selected_lab) for participant_id in selected_participants]
    
    return result


def create_participant_signal_overview(participant_stats, output_dir):
    """Create overview plots comparing signal characteristics across participants."""
    print("Creating participant signal overview...")
    
    # Prepare data for plotting
    plot_data = []
    for key, data_point in participant_stats.items():
        plot_data.append(data_point)
    
    df = pd.DataFrame(plot_data)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Signal Characteristics Across Participants', fontsize=16, fontweight='bold')
    
    # Plot 1: EEG signal amplitudes by participant
    ax1 = axes[0, 0]
    eeg_data = df[df['signal'].str.contains('EEG')]
    if not eeg_data.empty:
        # Select a subset of participants for readability
        top_participants = eeg_data['participant_id'].value_counts().head(10).index
        eeg_subset = eeg_data[eeg_data['participant_id'].isin(top_participants)]
        
        sns.boxplot(data=eeg_subset, x='participant_id', y='mean', hue='stage', ax=ax1)
        ax1.set_title('EEG Signal Amplitude by Participant')
        ax1.set_ylabel('Mean Amplitude')
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend(title='Sleep Stage', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 2: EMG signal amplitudes by participant
    ax2 = axes[0, 1]
    emg_data = df[df['signal'] == 'EMG']
    if not emg_data.empty:
        top_participants = emg_data['participant_id'].value_counts().head(10).index
        emg_subset = emg_data[emg_data['participant_id'].isin(top_participants)]
        
        sns.boxplot(data=emg_subset, x='participant_id', y='mean', hue='stage', ax=ax2)
        ax2.set_title('EMG Signal Amplitude by Participant')
        ax2.set_ylabel('Mean Amplitude')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend(title='Sleep Stage', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 3: Signal variability across participants
    ax3 = axes[0, 2]
    if not df.empty:
        participant_variability = df.groupby('participant_id')['std'].mean().sort_values(ascending=False)
        top_variable = participant_variability.head(10)
        
        sns.barplot(x=top_variable.index, y=top_variable.values, ax=ax3)
        ax3.set_title('Signal Variability by Participant')
        ax3.set_ylabel('Mean Standard Deviation')
        ax3.set_xlabel('Participant ID')
        ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: SNR distribution across participants
    ax4 = axes[1, 0]
    if 'snr' in df.columns:
        participant_snr = df.groupby('participant_id')['snr'].mean().sort_values(ascending=False)
        top_snr = participant_snr.head(10)
        
        sns.barplot(x=top_snr.index, y=top_snr.values, ax=ax4)
        ax4.set_title('Signal Quality (SNR) by Participant')
        ax4.set_ylabel('Mean SNR (dB)')
        ax4.set_xlabel('Participant ID')
        ax4.tick_params(axis='x', rotation=45)
    
    # Plot 5: Stage discrimination by participant
    ax5 = axes[1, 1]
    if not df.empty:
        discrimination_data = []
        for participant in df['participant_id'].unique():
            participant_data = df[df['participant_id'] == participant]
            stage_means = participant_data.groupby('stage')['mean'].mean()
            if len(stage_means) > 1:
                discrimination_index = stage_means.max() - stage_means.min()
                discrimination_data.append({
                    'participant_id': participant,
                    'discrimination_index': discrimination_index
                })
        
        if discrimination_data:
            disc_df = pd.DataFrame(discrimination_data)
            disc_df = disc_df.sort_values('discrimination_index', ascending=False).head(10)
            
            sns.barplot(data=disc_df, x='participant_id', y='discrimination_index', ax=ax5)
            ax5.set_title('Stage Discrimination by Participant')
            ax5.set_ylabel('Discrimination Index')
            ax5.set_xlabel('Participant ID')
            ax5.tick_params(axis='x', rotation=45)
    
    # Plot 6: Signal consistency by participant
    ax6 = axes[1, 2]
    if not df.empty:
        df['cv'] = df['std'] / df['mean'].abs()
        df['cv'] = df['cv'].replace([np.inf, -np.inf], np.nan)
        
        participant_consistency = df.groupby('participant_id')['cv'].mean().sort_values()
        top_consistent = participant_consistency.head(10)
        
        sns.barplot(x=top_consistent.index, y=top_consistent.values, ax=ax6)
        ax6.set_title('Signal Consistency by Participant')
        ax6.set_ylabel('Mean Coefficient of Variation')
        ax6.set_xlabel('Participant ID')
        ax6.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'participant_EEG_signal_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    # plt.close()  # Disabled for automated analysis


def create_participant_frequency_analysis(participant_stats, output_dir):
    """Create frequency domain analysis comparing participants."""
    print("Creating participant frequency analysis...")
    
    # Collect spectral data
    spectral_data = []
    for key, data_point in participant_stats.items():
        if 'spectral_power' in data_point and data_point['spectral_power']:
            for band, power in data_point['spectral_power'].items():
                spectral_data.append({
                    'participant_id': data_point['participant_id'],
                    'lab': data_point['lab'],
                    'signal': data_point['signal'],
                    'stage': data_point['stage'],
                    'frequency_band': band,
                    'power': power
                })
    
    if not spectral_data:
        print("No spectral data available for frequency analysis")
        return
    
    df_spectral = pd.DataFrame(spectral_data)
    
    # Create frequency analysis plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Frequency Domain Analysis Across Participants', fontsize=16, fontweight='bold')
    
    # Plot 1: Delta power distribution across participants
    ax1 = axes[0, 0]
    delta_data = df_spectral[df_spectral['frequency_band'] == 'delta']
    if not delta_data.empty:
        # Select top participants for readability
        top_participants = delta_data['participant_id'].value_counts().head(8).index
        delta_subset = delta_data[delta_data['participant_id'].isin(top_participants)]
        
        sns.boxplot(data=delta_subset, x='participant_id', y='power', ax=ax1)
        ax1.set_title('Delta Band Power Across Participants')
        ax1.set_ylabel('Delta Power')
        ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Alpha power distribution across participants
    ax2 = axes[0, 1]
    alpha_data = df_spectral[df_spectral['frequency_band'] == 'alpha']
    if not alpha_data.empty:
        top_participants = alpha_data['participant_id'].value_counts().head(8).index
        alpha_subset = alpha_data[alpha_data['participant_id'].isin(top_participants)]
        
        sns.boxplot(data=alpha_subset, x='participant_id', y='power', ax=ax2)
        ax2.set_title('Alpha Band Power Across Participants')
        ax2.set_ylabel('Alpha Power')
        ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Frequency band ratios by participant
    ax3 = axes[1, 0]
    # Calculate delta/alpha ratio for each participant
    ratio_data = []
    for participant in df_spectral['participant_id'].unique():
        participant_data = df_spectral[df_spectral['participant_id'] == participant]
        
        delta_power = participant_data[participant_data['frequency_band'] == 'delta']['power'].mean()
        alpha_power = participant_data[participant_data['frequency_band'] == 'alpha']['power'].mean()
        
        if not pd.isna(delta_power) and not pd.isna(alpha_power) and alpha_power > 0:
            ratio_data.append({
                'participant_id': participant,
                'delta_alpha_ratio': delta_power / alpha_power
            })
    
    if ratio_data:
        ratio_df = pd.DataFrame(ratio_data)
        ratio_df = ratio_df.sort_values('delta_alpha_ratio', ascending=False).head(10)
        
        sns.barplot(data=ratio_df, x='participant_id', y='delta_alpha_ratio', ax=ax3)
        ax3.set_title('Delta/Alpha Ratio by Participant')
        ax3.set_ylabel('Delta/Alpha Ratio')
        ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Spectral power heatmap for top participants
    ax4 = axes[1, 1]
    # Create a pivot table for heatmap
    top_participants = df_spectral['participant_id'].value_counts().head(8).index
    spectral_subset = df_spectral[df_spectral['participant_id'].isin(top_participants)]
    
    if not spectral_subset.empty:
        pivot_data = spectral_subset.groupby(['participant_id', 'frequency_band'])['power'].mean().unstack()
        if not pivot_data.empty:
            sns.heatmap(pivot_data, annot=True, fmt='.2e', cmap='viridis', ax=ax4)
            ax4.set_title('Spectral Power Profile by Participant')
            ax4.set_ylabel('Participant ID')
            ax4.set_xlabel('Frequency Band')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'participant_frequency_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    # plt.close()  # Disabled for automated analysis


def create_participant_sleep_profile_analysis(participant_stats, output_dir):
    """Create sleep profile analysis for each participant."""
    print("Creating participant sleep profile analysis...")
    
    # Prepare data for sleep profiling
    plot_data = []
    for key, data_point in participant_stats.items():
        plot_data.append(data_point)
    
    df = pd.DataFrame(plot_data)
    
    if df.empty:
        print("No data available for sleep profile analysis")
        return
    
    # Create sleep profile plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Sleep Profile Analysis Across Participants', fontsize=16, fontweight='bold')
    
    # Plot 1: NREM depth by participant (using delta power as proxy)
    ax1 = axes[0, 0]
    nrem_data = df[df['stage'] == 'NREM']
    if not nrem_data.empty:
        nrem_summary = nrem_data.groupby('participant_id')['mean'].mean().sort_values(ascending=False)
        top_nrem = nrem_summary.head(10)
        
        sns.barplot(x=top_nrem.index, y=top_nrem.values, ax=ax1)
        ax1.set_title('NREM Sleep Depth by Participant')
        ax1.set_ylabel('Mean NREM Amplitude')
        ax1.set_xlabel('Participant ID')
        ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: REM intensity by participant
    ax2 = axes[0, 1]
    rem_data = df[df['stage'] == 'REM']
    if not rem_data.empty:
        rem_summary = rem_data.groupby('participant_id')['mean'].mean().sort_values(ascending=False)
        top_rem = rem_summary.head(10)
        
        sns.barplot(x=top_rem.index, y=top_rem.values, ax=ax2)
        ax2.set_title('REM Sleep Intensity by Participant')
        ax2.set_ylabel('Mean REM Amplitude')
        ax2.set_xlabel('Participant ID')
        ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Wake/Sleep transition quality
    ax3 = axes[1, 0]
    # Calculate the difference between awake and sleep states as transition quality
    transition_data = []
    for participant in df['participant_id'].unique():
        participant_data = df[df['participant_id'] == participant]
        
        awake_mean = participant_data[participant_data['stage'] == 'Awake']['mean'].mean()
        sleep_mean = participant_data[participant_data['stage'].isin(['NREM', 'REM'])]['mean'].mean()
        
        if not pd.isna(awake_mean) and not pd.isna(sleep_mean):
            transition_quality = abs(awake_mean - sleep_mean)
            transition_data.append({
                'participant_id': participant,
                'transition_quality': transition_quality
            })
    
    if transition_data:
        transition_df = pd.DataFrame(transition_data)
        transition_df = transition_df.sort_values('transition_quality', ascending=False).head(10)
        
        sns.barplot(data=transition_df, x='participant_id', y='transition_quality', ax=ax3)
        ax3.set_title('Wake/Sleep Transition Quality by Participant')
        ax3.set_ylabel('Transition Quality Index')
        ax3.set_xlabel('Participant ID')
        ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Overall sleep quality score
    ax4 = axes[1, 1]
    # Calculate composite sleep quality score
    quality_data = []
    for participant in df['participant_id'].unique():
        participant_data = df[df['participant_id'] == participant]
        
        # Use coefficient of variation as stability measure (lower is better)
        participant_data['cv'] = participant_data['std'] / participant_data['mean'].abs()
        participant_data['cv'] = participant_data['cv'].replace([np.inf, -np.inf], np.nan)
        
        stability_score = 1 / (1 + participant_data['cv'].mean())  # Higher is more stable
        
        # Use mean amplitude as depth measure
        depth_score = participant_data['mean'].mean()
        
        # Composite score (normalize and combine)
        if not pd.isna(stability_score) and not pd.isna(depth_score):
            quality_score = stability_score * abs(depth_score)
            quality_data.append({
                'participant_id': participant,
                'sleep_quality_score': quality_score
            })
    
    if quality_data:
        quality_df = pd.DataFrame(quality_data)
        quality_df = quality_df.sort_values('sleep_quality_score', ascending=False).head(10)
        
        sns.barplot(data=quality_df, x='participant_id', y='sleep_quality_score', ax=ax4)
        ax4.set_title('Overall Sleep Quality Score by Participant')
        ax4.set_ylabel('Sleep Quality Score')
        ax4.set_xlabel('Participant ID')
        ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'participant_sleep_profiles.png', dpi=150, bbox_inches='tight')
    plt.close()
    # plt.close()  # Disabled for automated analysis


def create_participant_spectral_profiles(participant_stats, output_dir):
    """Create detailed spectral profiles for participants."""
    print("Creating participant spectral profiles...")
    
    # Collect spectral data
    spectral_data = []
    for key, data_point in participant_stats.items():
        if 'spectral_power' in data_point and data_point['spectral_power']:
            base_info = {
                'participant_id': data_point['participant_id'],
                'lab': data_point['lab'],
                'signal': data_point['signal'],
                'stage': data_point['stage']
            }
            
            # Add all frequency bands
            for band, power in data_point['spectral_power'].items():
                base_info[band] = power
            
            spectral_data.append(base_info)
    
    if not spectral_data:
        print("No spectral data available for spectral profile analysis")
        return
    
    df_spectral = pd.DataFrame(spectral_data)
    
    # Create spectral profile plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Detailed Spectral Profiles Across Participants', fontsize=16, fontweight='bold')
    
    # Plot 1: Participant spectral fingerprints
    ax1 = axes[0, 0]
    band_columns = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    available_bands = [col for col in band_columns if col in df_spectral.columns]
    
    if available_bands:
        # Calculate total power and relative power for each participant
        df_spectral['total_power'] = df_spectral[available_bands].sum(axis=1)
        
        for band in available_bands:
            df_spectral[f'{band}_relative'] = df_spectral[band] / df_spectral['total_power']
        
        # Get top participants for visualization
        top_participants = df_spectral['participant_id'].value_counts().head(8).index
        spectral_subset = df_spectral[df_spectral['participant_id'].isin(top_participants)]
        
        # Create stacked bar plot
        participant_band_means = spectral_subset.groupby('participant_id')[[f'{band}_relative' for band in available_bands]].mean()
        participant_band_means.plot(kind='bar', stacked=True, ax=ax1)
        ax1.set_title('Spectral Fingerprint by Participant')
        ax1.set_ylabel('Relative Power')
        ax1.set_xlabel('Participant ID')
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend(title='Frequency Band', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 2: High-frequency vs low-frequency balance
    ax2 = axes[0, 1]
    if 'delta' in df_spectral.columns and 'gamma' in df_spectral.columns:
        df_spectral['low_high_ratio'] = df_spectral['delta'] / (df_spectral['gamma'] + 1e-10)
        
        participant_ratio = df_spectral.groupby('participant_id')['low_high_ratio'].mean().sort_values(ascending=False)
        top_ratio = participant_ratio.head(10)
        
        sns.barplot(x=top_ratio.index, y=top_ratio.values, ax=ax2)
        ax2.set_title('Low/High Frequency Balance by Participant')
        ax2.set_ylabel('Delta/Gamma Ratio')
        ax2.set_xlabel('Participant ID')
        ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Sleep-specific spectral characteristics
    ax3 = axes[1, 0]
    if 'theta' in df_spectral.columns:
        # Focus on theta power during REM sleep
        rem_theta = df_spectral[df_spectral['stage'] == 'REM']
        if not rem_theta.empty:
            rem_theta_summary = rem_theta.groupby('participant_id')['theta'].mean().sort_values(ascending=False)
            top_rem_theta = rem_theta_summary.head(10)
            
            sns.barplot(x=top_rem_theta.index, y=top_rem_theta.values, ax=ax3)
            ax3.set_title('REM Theta Power by Participant')
            ax3.set_ylabel('Mean Theta Power (REM)')
            ax3.set_xlabel('Participant ID')
            ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Arousal indicators
    ax4 = axes[1, 1]
    if 'alpha' in df_spectral.columns and 'beta' in df_spectral.columns:
        # Calculate arousal index during wake periods
        wake_data = df_spectral[df_spectral['stage'] == 'Awake']
        if not wake_data.empty:
            wake_data['arousal_index'] = wake_data['alpha'] + wake_data['beta']
            
            arousal_summary = wake_data.groupby('participant_id')['arousal_index'].mean().sort_values(ascending=False)
            top_arousal = arousal_summary.head(10)
            
            sns.barplot(x=top_arousal.index, y=top_arousal.values, ax=ax4)
            ax4.set_title('Arousal Index by Participant (Wake)')
            ax4.set_ylabel('Alpha + Beta Power')
            ax4.set_xlabel('Participant ID')
            ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'participant_spectral_profiles.png', dpi=150, bbox_inches='tight')
    plt.close()
    # plt.close()  # Disabled for automated analysis


def create_participant_variability_analysis(participant_stats, output_dir):
    """Create analysis of participant variability and individual differences."""
    print("Creating participant variability analysis...")
    
    # Prepare data for variability analysis
    plot_data = []
    for key, data_point in participant_stats.items():
        plot_data.append(data_point)
    
    df = pd.DataFrame(plot_data)
    
    if df.empty:
        print("No data available for variability analysis")
        return
    
    # Create variability analysis plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Participant Variability and Individual Differences', fontsize=16, fontweight='bold')
    
    # Plot 1: Inter-participant variability by signal type
    ax1 = axes[0, 0]
    variability_data = []
    for signal_type in df['signal'].unique():
        signal_data = df[df['signal'] == signal_type]
        participant_means = signal_data.groupby('participant_id')['mean'].mean()
        
        if len(participant_means) > 1:
            inter_participant_cv = participant_means.std() / participant_means.mean()
            variability_data.append({
                'signal': signal_type,
                'inter_participant_cv': inter_participant_cv
            })
    
    if variability_data:
        var_df = pd.DataFrame(variability_data)
        sns.barplot(data=var_df, x='signal', y='inter_participant_cv', ax=ax1)
        ax1.set_title('Inter-Participant Variability by Signal Type')
        ax1.set_ylabel('Coefficient of Variation')
        ax1.set_xlabel('Signal Type')
        ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Intra-participant consistency
    ax2 = axes[0, 1]
    consistency_data = []
    for participant in df['participant_id'].unique():
        participant_data = df[df['participant_id'] == participant]
        
        # Calculate consistency across different stages
        stage_means = participant_data.groupby('stage')['mean'].mean()
        if len(stage_means) > 1:
            intra_participant_cv = stage_means.std() / stage_means.mean()
            consistency_data.append({
                'participant_id': participant,
                'intra_participant_cv': intra_participant_cv
            })
    
    if consistency_data:
        cons_df = pd.DataFrame(consistency_data)
        cons_df = cons_df.sort_values('intra_participant_cv', ascending=False).head(10)
        
        sns.barplot(data=cons_df, x='participant_id', y='intra_participant_cv', ax=ax2)
        ax2.set_title('Intra-Participant Variability (Top 10)')
        ax2.set_ylabel('Coefficient of Variation Across Stages')
        ax2.set_xlabel('Participant ID')
        ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Signal quality distribution across participants
    ax3 = axes[1, 0]
    if 'snr' in df.columns:
        quality_distribution = df.groupby('participant_id')['snr'].mean()
        
        ax3.hist(quality_distribution.values, bins=20, alpha=0.7, edgecolor='black')
        ax3.set_title('Distribution of Signal Quality Across Participants')
        ax3.set_xlabel('Mean SNR (dB)')
        ax3.set_ylabel('Number of Participants')
        ax3.axvline(quality_distribution.mean(), color='red', linestyle='--', 
                   label=f'Mean: {quality_distribution.mean():.1f} dB')
        ax3.legend()
    
    # Plot 4: Outlier identification
    ax4 = axes[1, 1]
    # Identify participants with unusual signal characteristics
    outlier_scores = []
    for participant in df['participant_id'].unique():
        participant_data = df[df['participant_id'] == participant]
        
        # Calculate z-score based on mean amplitude
        overall_mean = df['mean'].mean()
        overall_std = df['mean'].std()
        participant_mean = participant_data['mean'].mean()
        
        z_score = abs((participant_mean - overall_mean) / overall_std) if overall_std > 0 else 0
        
        outlier_scores.append({
            'participant_id': participant,
            'outlier_score': z_score
        })
    
    if outlier_scores:
        outlier_df = pd.DataFrame(outlier_scores)
        outlier_df = outlier_df.sort_values('outlier_score', ascending=False).head(10)
        
        sns.barplot(data=outlier_df, x='participant_id', y='outlier_score', ax=ax4)
        ax4.set_title('Signal Outliers (Top 10 Unusual Participants)')
        ax4.set_ylabel('Outlier Score (Z-score)')
        ax4.set_xlabel('Participant ID')
        ax4.tick_params(axis='x', rotation=45)
        ax4.axhline(y=2, color='red', linestyle='--', label='Outlier Threshold (2σ)')
        ax4.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'participant_variability_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    # plt.close()  # Disabled for automated analysis


def create_participant_summary_statistics(participant_stats, output_dir):
    """Create comprehensive summary statistics for participant differences."""
    print("Creating participant summary statistics...")
    
    # Prepare comprehensive summary data
    summary_data = []
    
    for participant_id in set(data['participant_id'] for data in participant_stats.values()):
        participant_data = {data['participant_id']: data for key, data in participant_stats.items() 
                          if data['participant_id'] == participant_id}
        
        if not participant_data:
            continue
        
        # Get lab information
        lab = list(participant_data.values())[0]['lab']
        
        # Calculate key metrics
        all_means = [data['mean'] for data in participant_data.values()]
        all_stds = [data['std'] for data in participant_data.values()]
        
        # Get spectral power information
        spectral_info = {}
        for data in participant_data.values():
            if 'spectral_power' in data and data['spectral_power']:
                for band, power in data['spectral_power'].items():
                    if band not in spectral_info:
                        spectral_info[band] = []
                    spectral_info[band].append(power)
        
        # Calculate stage-specific metrics
        stage_metrics = {}
        for stage in ['Awake', 'NREM', 'REM', 'Artifact']:
            stage_data = [data for data in participant_data.values() if data['stage'] == stage]
            if stage_data:
                stage_means = [data['mean'] for data in stage_data]
                stage_metrics[f'{stage}_amplitude'] = np.mean(stage_means)
        
        # Calculate sleep quality indicators
        nrem_power = np.mean(spectral_info.get('delta', [0])) if 'delta' in spectral_info else 0
        rem_power = np.mean(spectral_info.get('theta', [0])) if 'theta' in spectral_info else 0
        awake_power = np.mean(spectral_info.get('alpha', [0])) if 'alpha' in spectral_info else 0
        
        # Calculate arousal ratio (higher alpha+beta during wake vs sleep)
        wake_arousal = awake_power + np.mean(spectral_info.get('beta', [0])) if 'beta' in spectral_info else awake_power
        sleep_arousal = (nrem_power + rem_power) / 2 if nrem_power > 0 and rem_power > 0 else 0
        arousal_ratio = wake_arousal / sleep_arousal if sleep_arousal > 0 else 0
        
        # Calculate sleep depth ratio (NREM vs other stages)
        nrem_amplitude = stage_metrics.get('NREM_amplitude', 0)
        other_amplitudes = [stage_metrics.get(f'{stage}_amplitude', 0) 
                          for stage in ['Awake', 'REM'] if f'{stage}_amplitude' in stage_metrics]
        avg_other = np.mean(other_amplitudes) if other_amplitudes else 0
        sleep_depth_ratio = nrem_amplitude / avg_other if avg_other > 0 else 0
        
        # Overall sleep quality score
        sleep_quality_score = nrem_power + rem_power + awake_power
        
        summary_entry = {
            'Participant_ID': participant_id,
            'Laboratory': lab,
            'Delta_Power_NREM': nrem_power,
            'Theta_Power_REM': rem_power,
            'Alpha_Power_Awake': awake_power,
            'Arousal_Ratio_Awake': arousal_ratio,
            'Sleep_Depth_Ratio_NREM': sleep_depth_ratio,
            'Sleep_Quality_Score': sleep_quality_score,
            **stage_metrics
        }
        
        summary_data.append(summary_entry)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    if not summary_df.empty:
        # Sort by sleep quality score
        summary_df = summary_df.sort_values('Sleep_Quality_Score', ascending=False)
        
        # Save summary
        summary_file = output_dir / 'participant_differences_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        print(f"Participant summary saved to: {summary_file}")
        
        # Create detailed statistics
        detailed_stats = []
        for key, data in participant_stats.items():
            detailed_stats.append(data)
        
        detailed_df = pd.DataFrame(detailed_stats)
        detailed_file = output_dir / 'participant_differences_detailed.csv'
        detailed_df.to_csv(detailed_file, index=False)
        print(f"Detailed participant statistics saved to: {detailed_file}")
    
    print(f"Participant difference analysis completed. Results saved to: {output_dir}")


if __name__ == "__main__":
    analyze_participant_differences()
