"""
Analysis of signal differences between laboratories.
This module focuses on identifying and visualizing the key signal characteristics
that distinguish measurements from different laboratories, which helps understand
measurement variability and standardization across research sites.
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

def analyze_lab_differences(output_dir="results/data_exploration", participants_per_lab: int = 10, exclude_lab1: bool = False):
    """
    Comprehensive analysis of signal differences between laboratories.
    
    Args:
        output_dir (str): Directory to save results
    """
    print("=" * 60)
    print("ANALYZING SIGNAL DIFFERENCES BETWEEN LABORATORIES")
    print("=" * 60)
    
    # Ensure output directory exists
    output_dir = ensure_output_directory(output_dir)
    lab_output_dir = Path(output_dir) / "lab_differences"
    lab_output_dir.mkdir(exist_ok=True)
    
    # Load metadata
    metadata = load_metadata()
    
    # Optionally exclude lab_1 from participant selection (still appears in comparisons if kept)
    if exclude_lab1:
        before = len(metadata)
        metadata = metadata[metadata['lab'] != 'lab_1']
        print(f"Excluded lab_1 from selection set: {before - len(metadata)} records removed")

    # Get available labs and select representative participants from each (post-exclusion set)
    labs = metadata['lab'].unique()
    print(f"Available laboratories (selection pool): {sorted(labs)}")
    
    selected_participants = select_lab_representatives(metadata, participants_per_lab=participants_per_lab)
    
    # Collect comprehensive data from all labs
    all_lab_stats = {}
    lab_summary_data = {}
    
    for lab in labs:
        lab_participants = [p for p in selected_participants if p[1] == lab]
        print(f"\nProcessing lab {lab} with {len(lab_participants)} participants...")
        
        lab_data = {}
        
        for participant_id, participant_lab in lab_participants:
            print(f"  Processing {participant_id}...")
            participant_runs = metadata[metadata['participant_id'] == participant_id]
            
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
                        
                        print(f"    Analyzing {signal_name}...")
                        
                        # Calculate statistics for the entire signal
                        stats_result = calculate_signal_statistics(signal_data, labels)
                        
                        # Store statistics for each stage
                        for stage_name, stage_stats in stats_result.items():
                            # Exclude Artifact stage globally (only artifact_characteristics keeps it)
                            if stage_name == 'Artifact':
                                continue
                            key = f"{lab}_{signal_name}_{stage_name}"
                            if key not in lab_data:
                                lab_data[key] = []
                            
                            lab_data[key].append({
                                'lab': lab,
                                'participant_id': participant_id,
                                'signal': signal_name,
                                'stage': stage_name,
                                **stage_stats
                            })
                    
                except Exception as e:
                    print(f"      Error processing {participant_id} run {run_id}: {e}")
                    continue
        
        # Aggregate lab statistics
        for key, data_list in lab_data.items():
            if data_list:  # Only process if we have data
                all_lab_stats[key] = data_list
                
                # Calculate lab-level aggregated statistics
                lab_name, signal_name, stage_name = key.split('_', 2)
                
                # Aggregate key metrics
                means = [d['mean'] for d in data_list]
                stds = [d['std'] for d in data_list]
                spectral_powers = [d.get('spectral_power', {}) for d in data_list if d.get('spectral_power')]
                
                summary_key = f"{lab_name}_{signal_name}_{stage_name}"
                lab_summary_data[summary_key] = {
                    'lab': lab_name,
                    'signal': signal_name,
                    'stage': stage_name,
                    'n_recordings': len(data_list),
                    'mean_amplitude': np.mean(means),
                    'std_amplitude': np.std(means),
                    'cv_amplitude': np.std(means) / np.mean(means) if np.mean(means) != 0 else 0,
                    'mean_std': np.mean(stds),
                    'std_std': np.std(stds),
                }
                
                # Add spectral power information if available
                if spectral_powers and any(spectral_powers):
                    for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
                        band_powers = [sp.get(band, 0) for sp in spectral_powers if sp and band in sp]
                        if band_powers:
                            lab_summary_data[summary_key][f'{band}_power_mean'] = np.mean(band_powers)
                            lab_summary_data[summary_key][f'{band}_power_std'] = np.std(band_powers)
                            lab_summary_data[summary_key][f'{band}_power_cv'] = np.std(band_powers) / np.mean(band_powers) if np.mean(band_powers) != 0 else 0
    
    if not all_lab_stats:
        print("No valid data found for analysis!")
        return
    
    print(f"\nCollected data from {len(labs)} laboratories")
    
    # Create visualizations
    create_lab_signal_overview(all_lab_stats, lab_output_dir)
    create_lab_frequency_analysis(all_lab_stats, lab_output_dir)
    create_lab_measurement_quality(all_lab_stats, lab_output_dir)
    create_lab_variability_analysis(all_lab_stats, lab_output_dir)
    create_lab_standardization_analysis(all_lab_stats, lab_output_dir)
    create_lab_psd_comparison(all_lab_stats, lab_output_dir)
    
    # Create summary statistics
    create_lab_summary_statistics(lab_summary_data, all_lab_stats, lab_output_dir)
    
    print(f"\nLab difference analysis completed. Results saved to: {lab_output_dir}")


def select_lab_representatives(metadata, participants_per_lab=10):
    """
    Select representative participants from each laboratory for analysis.
    
    Args:
        metadata (pd.DataFrame): Participant metadata
        participants_per_lab (int): Number of participants to select per lab
    
    Returns:
        list: List of (participant_id, lab) tuples
    """
    selected_participants = []
    
    for lab in sorted(metadata['lab'].unique()):
        lab_participants = metadata[metadata['lab'] == lab]['participant_id'].unique()
        
        # Select up to participants_per_lab from each lab
        n_select = min(participants_per_lab, len(lab_participants))
        
        # Sort for reproducible selection
        selected_from_lab = sorted(lab_participants)[:n_select]
        
        for participant_id in selected_from_lab:
            selected_participants.append((participant_id, lab))
        
        print(f"Selected {n_select} participants from {lab}: {selected_from_lab}")
    
    return selected_participants


def create_lab_signal_overview(lab_stats, output_dir):
    """Create overview plots comparing signal characteristics across laboratories."""
    print("Creating lab signal overview...")
    
    # Prepare data for plotting
    plot_data = []
    for key, data_list in lab_stats.items():
        for data_point in data_list:
            plot_data.append(data_point)
    
    df = pd.DataFrame(plot_data)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Signal Characteristics Across Laboratories', fontsize=16, fontweight='bold')
    
    # Plot 1: Mean amplitude by lab and signal type
    ax1 = axes[0, 0]
    if 'mean' in df.columns:
        sns.boxplot(data=df, x='lab', y='mean', hue='signal', ax=ax1)
        ax1.set_title('Signal Amplitude Distribution by Lab')
        ax1.set_ylabel('Mean Amplitude')
        ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Signal variability by lab
    ax2 = axes[0, 1]
    if 'std' in df.columns:
        sns.boxplot(data=df, x='lab', y='std', hue='signal', ax=ax2)
        ax2.set_title('Signal Variability by Lab')
        ax2.set_ylabel('Standard Deviation')
        ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: SNR comparison across labs
    ax3 = axes[0, 2]
    if 'snr' in df.columns:
        sns.boxplot(data=df, x='lab', y='snr', hue='signal', ax=ax3)
        ax3.set_title('Signal-to-Noise Ratio by Lab')
        ax3.set_ylabel('SNR (dB)')
        ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Stage-specific analysis - NREM
    ax4 = axes[1, 0]
    nrem_data = df[df['stage'] == 'NREM']
    if not nrem_data.empty and 'mean' in nrem_data.columns:
        sns.boxplot(data=nrem_data, x='lab', y='mean', hue='signal', ax=ax4)
        ax4.set_title('NREM Stage Signal Amplitude by Lab')
        ax4.set_ylabel('Mean Amplitude (NREM)')
        ax4.tick_params(axis='x', rotation=45)
    
    # Plot 5: Stage-specific analysis - REM
    ax5 = axes[1, 1]
    rem_data = df[df['stage'] == 'REM']
    if not rem_data.empty and 'mean' in rem_data.columns:
        sns.boxplot(data=rem_data, x='lab', y='mean', hue='signal', ax=ax5)
        ax5.set_title('REM Stage Signal Amplitude by Lab')
        ax5.set_ylabel('Mean Amplitude (REM)')
        ax5.tick_params(axis='x', rotation=45)
    
    # Plot 6: Stage-specific analysis - Awake
    ax6 = axes[1, 2]
    awake_data = df[df['stage'] == 'Awake']
    if not awake_data.empty and 'mean' in awake_data.columns:
        sns.boxplot(data=awake_data, x='lab', y='mean', hue='signal', ax=ax6)
        ax6.set_title('Awake Stage Signal Amplitude by Lab')
        ax6.set_ylabel('Mean Amplitude (Awake)')
        ax6.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'lab_signal_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    # plt.show()  # Disabled for automated analysis


def create_lab_frequency_analysis(lab_stats, output_dir):
    """Create frequency domain analysis comparing laboratories."""
    print("Creating lab frequency analysis...")
    
    # Collect spectral data
    spectral_data = []
    for key, data_list in lab_stats.items():
        for data_point in data_list:
            if 'spectral_power' in data_point and data_point['spectral_power']:
                for band, power in data_point['spectral_power'].items():
                    spectral_data.append({
                        'lab': data_point['lab'],
                        'signal': data_point['signal'],
                        'stage': data_point['stage'],
                        'participant_id': data_point['participant_id'],
                        'frequency_band': band,
                        'power': power
                    })
    
    if not spectral_data:
        print("No spectral data available for frequency analysis")
        return
    
    df_spectral = pd.DataFrame(spectral_data)
    
    # Create frequency analysis plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Frequency Domain Analysis Across Laboratories', fontsize=16, fontweight='bold')
    
    # Plot 1: Delta band power across labs
    ax1 = axes[0, 0]
    delta_data = df_spectral[df_spectral['frequency_band'] == 'delta']
    if not delta_data.empty:
        sns.boxplot(data=delta_data, x='lab', y='power', hue='signal', ax=ax1)
        ax1.set_title('Delta Band Power (0.5-4 Hz)')
        ax1.set_ylabel('Power')
        ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Alpha band power across labs
    ax2 = axes[0, 1]
    alpha_data = df_spectral[df_spectral['frequency_band'] == 'alpha']
    if not alpha_data.empty:
        sns.boxplot(data=alpha_data, x='lab', y='power', hue='signal', ax=ax2)
        ax2.set_title('Alpha Band Power (8-12 Hz)')
        ax2.set_ylabel('Power')
        ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Beta band power across labs
    ax3 = axes[1, 0]
    beta_data = df_spectral[df_spectral['frequency_band'] == 'beta']
    if not beta_data.empty:
        sns.boxplot(data=beta_data, x='lab', y='power', hue='signal', ax=ax3)
        ax3.set_title('Beta Band Power (13-30 Hz)')
        ax3.set_ylabel('Power')
        ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Frequency band ratios by lab
    ax4 = axes[1, 1]
    # Calculate delta/alpha ratio as a standardization metric
    ratio_data = []
    for lab in df_spectral['lab'].unique():
        for signal in df_spectral['signal'].unique():
            for stage in df_spectral['stage'].unique():
                lab_signal_stage = df_spectral[
                    (df_spectral['lab'] == lab) & 
                    (df_spectral['signal'] == signal) & 
                    (df_spectral['stage'] == stage)
                ]
                
                delta_power = lab_signal_stage[lab_signal_stage['frequency_band'] == 'delta']['power'].mean()
                alpha_power = lab_signal_stage[lab_signal_stage['frequency_band'] == 'alpha']['power'].mean()
                
                if not pd.isna(delta_power) and not pd.isna(alpha_power) and alpha_power > 0:
                    ratio_data.append({
                        'lab': lab,
                        'signal': signal,
                        'stage': stage,
                        'delta_alpha_ratio': delta_power / alpha_power
                    })
    
    if ratio_data:
        df_ratio = pd.DataFrame(ratio_data)
        sns.boxplot(data=df_ratio, x='lab', y='delta_alpha_ratio', hue='signal', ax=ax4)
        ax4.set_title('Delta/Alpha Power Ratio')
        ax4.set_ylabel('Ratio')
        ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'lab_frequency_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    # plt.show()  # Disabled for automated analysis


def create_lab_measurement_quality(lab_stats, output_dir):
    """Create measurement quality comparison across laboratories."""
    print("Creating lab measurement quality analysis...")
    
    # Prepare quality metrics data
    quality_data = []
    for key, data_list in lab_stats.items():
        for data_point in data_list:
            quality_metrics = {
                'lab': data_point['lab'],
                'signal': data_point['signal'],
                'stage': data_point['stage'],
                'participant_id': data_point['participant_id'],
                'snr': data_point.get('snr', np.nan),
                'kurtosis': data_point.get('kurtosis', np.nan),
                'skewness': data_point.get('skewness', np.nan),
                'mean': data_point.get('mean', np.nan),
                'std': data_point.get('std', np.nan)
            }
            
            # Calculate coefficient of variation as quality metric
            if quality_metrics['mean'] != 0:
                quality_metrics['cv'] = quality_metrics['std'] / abs(quality_metrics['mean'])
            else:
                quality_metrics['cv'] = np.nan
            
            quality_data.append(quality_metrics)
    
    df_quality = pd.DataFrame(quality_data)
    
    # Create quality analysis plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Measurement Quality Across Laboratories', fontsize=16, fontweight='bold')
    
    # Plot 1: Signal-to-Noise Ratio
    ax1 = axes[0, 0]
    if 'snr' in df_quality.columns and not df_quality['snr'].isna().all():
        sns.boxplot(data=df_quality, x='lab', y='snr', hue='signal', ax=ax1)
        ax1.set_title('Signal-to-Noise Ratio by Laboratory')
        ax1.set_ylabel('SNR (dB)')
        ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Coefficient of Variation
    ax2 = axes[0, 1]
    if 'cv' in df_quality.columns and not df_quality['cv'].isna().all():
        sns.boxplot(data=df_quality, x='lab', y='cv', hue='signal', ax=ax2)
        ax2.set_title('Coefficient of Variation by Laboratory')
        ax2.set_ylabel('CV (std/mean)')
        ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Signal Distribution Shape (Kurtosis)
    ax3 = axes[1, 0]
    if 'kurtosis' in df_quality.columns and not df_quality['kurtosis'].isna().all():
        sns.boxplot(data=df_quality, x='lab', y='kurtosis', hue='signal', ax=ax3)
        ax3.set_title('Signal Distribution Kurtosis by Laboratory')
        ax3.set_ylabel('Kurtosis')
        ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Signal Distribution Shape (Skewness)
    ax4 = axes[1, 1]
    if 'skewness' in df_quality.columns and not df_quality['skewness'].isna().all():
        sns.boxplot(data=df_quality, x='lab', y='skewness', hue='signal', ax=ax4)
        ax4.set_title('Signal Distribution Skewness by Laboratory')
        ax4.set_ylabel('Skewness')
        ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'lab_measurement_quality.png', dpi=150, bbox_inches='tight')
    plt.close()
    # plt.show()  # Disabled for automated analysis


def create_lab_variability_analysis(lab_stats, output_dir):
    """Create inter-laboratory variability analysis."""
    print("Creating lab variability analysis...")
    
    # Calculate inter-lab variability metrics
    variability_data = []
    
    # Group by signal and stage to calculate inter-lab variability
    for signal in ['EEG1', 'EEG2', 'EEG3', 'EEG4', 'EMG']:
        for stage in STAGE_ORDER:
            stage_signal_data = {}
            
            # Collect data for each lab
            for key, data_list in lab_stats.items():
                if f"_{signal}_{stage}" in key:
                    lab = key.split('_')[0]
                    means = [d['mean'] for d in data_list]
                    if means:
                        stage_signal_data[lab] = {
                            'mean_amplitude': np.mean(means),
                            'std_amplitude': np.std(means),
                            'n_recordings': len(means)
                        }
            
            if len(stage_signal_data) > 1:  # Need at least 2 labs for comparison
                lab_means = [data['mean_amplitude'] for data in stage_signal_data.values()]
                lab_stds = [data['std_amplitude'] for data in stage_signal_data.values()]
                
                variability_data.append({
                    'signal': signal,
                    'stage': stage,
                    'inter_lab_cv': np.std(lab_means) / np.mean(lab_means) if np.mean(lab_means) != 0 else 0,
                    'mean_inter_lab_std': np.std(lab_means),
                    'mean_intra_lab_std': np.mean(lab_stds),
                    'n_labs': len(stage_signal_data),
                    'total_recordings': sum(data['n_recordings'] for data in stage_signal_data.values())
                })
    
    if not variability_data:
        print("Insufficient data for variability analysis")
        return
    
    df_variability = pd.DataFrame(variability_data)
    
    # Create variability plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Inter-Laboratory Variability Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Inter-lab coefficient of variation by signal type
    ax1 = axes[0, 0]
    sns.barplot(data=df_variability, x='signal', y='inter_lab_cv', hue='stage', ax=ax1)
    ax1.set_title('Inter-Laboratory Coefficient of Variation')
    ax1.set_ylabel('CV (between labs)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(title='Sleep Stage', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 2: Ratio of inter-lab to intra-lab variability
    ax2 = axes[0, 1]
    df_variability['variability_ratio'] = df_variability['mean_inter_lab_std'] / df_variability['mean_intra_lab_std']
    sns.barplot(data=df_variability, x='signal', y='variability_ratio', hue='stage', ax=ax2)
    ax2.set_title('Inter-lab / Intra-lab Variability Ratio')
    ax2.set_ylabel('Ratio')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(title='Sleep Stage', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 3: Heatmap of inter-lab CV by signal and stage
    ax3 = axes[1, 0]
    pivot_cv = df_variability.pivot(index='signal', columns='stage', values='inter_lab_cv')
    sns.heatmap(pivot_cv, annot=True, fmt='.3f', cmap='Reds', ax=ax3)
    ax3.set_title('Inter-Lab CV Heatmap')
    ax3.set_ylabel('Signal Type')
    ax3.set_xlabel('Sleep Stage')
    
    # Plot 4: Number of recordings per analysis
    ax4 = axes[1, 1]
    sns.barplot(data=df_variability, x='signal', y='total_recordings', hue='stage', ax=ax4)
    ax4.set_title('Number of Recordings per Analysis')
    ax4.set_ylabel('Total Recordings')
    ax4.tick_params(axis='x', rotation=45)
    ax4.legend(title='Sleep Stage', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'lab_variability_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    # plt.show()  # Disabled for automated analysis


def create_lab_standardization_analysis(lab_stats, output_dir):
    """Create analysis of standardization differences between laboratories."""
    print("Creating lab standardization analysis...")
    
    # Collect data for standardization analysis
    standardization_data = []
    
    for key, data_list in lab_stats.items():
        lab, signal, stage = key.split('_', 2)
        
        if data_list:
            means = [d['mean'] for d in data_list]
            stds = [d['std'] for d in data_list]
            
            # Calculate standardization metrics
            standardization_data.append({
                'lab': lab,
                'signal': signal,
                'stage': stage,
                'mean_signal_amplitude': np.mean(means),
                'std_signal_amplitude': np.std(means),
                'mean_signal_variability': np.mean(stds),
                'n_recordings': len(data_list),
                'amplitude_range': np.max(means) - np.min(means) if len(means) > 1 else 0,
                'consistency_score': 1 / (1 + np.std(means)) if len(means) > 1 else 1  # Higher is more consistent
            })
    
    df_std = pd.DataFrame(standardization_data)
    
    # Create standardization plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Laboratory Standardization Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Signal amplitude consistency across labs
    ax1 = axes[0, 0]
    eeg_data = df_std[df_std['signal'].str.contains('EEG')]
    if not eeg_data.empty:
        sns.boxplot(data=eeg_data, x='lab', y='consistency_score', hue='stage', ax=ax1)
        ax1.set_title('EEG Signal Consistency by Laboratory')
        ax1.set_ylabel('Consistency Score (higher = more consistent)')
        ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Signal amplitude ranges across labs
    ax2 = axes[0, 1]
    if not eeg_data.empty:
        sns.boxplot(data=eeg_data, x='lab', y='amplitude_range', hue='stage', ax=ax2)
        ax2.set_title('EEG Signal Amplitude Range by Laboratory')
        ax2.set_ylabel('Amplitude Range')
        ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: EMG standardization
    ax3 = axes[1, 0]
    emg_data = df_std[df_std['signal'] == 'EMG']
    if not emg_data.empty:
        sns.boxplot(data=emg_data, x='lab', y='consistency_score', hue='stage', ax=ax3)
        ax3.set_title('EMG Signal Consistency by Laboratory')
        ax3.set_ylabel('Consistency Score')
        ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Overall standardization score by lab
    ax4 = axes[1, 1]
    lab_scores = df_std.groupby('lab')['consistency_score'].mean().reset_index()
    lab_scores = lab_scores.sort_values('consistency_score', ascending=False)
    sns.barplot(data=lab_scores, x='lab', y='consistency_score', ax=ax4)
    ax4.set_title('Overall Standardization Score by Laboratory')
    ax4.set_ylabel('Mean Consistency Score')
    ax4.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(lab_scores['consistency_score']):
        ax4.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'lab_standardization_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    # plt.show()  # Disabled for automated analysis


def create_lab_summary_statistics(lab_summary_data, all_lab_stats, output_dir):
    """Create comprehensive summary statistics for laboratory differences."""
    print("Creating lab summary statistics...")
    
    # Create detailed summary CSV
    summary_df = pd.DataFrame(lab_summary_data).T
    summary_df = summary_df.reset_index(drop=True)
    
    # Sort by lab and signal for better organization
    if not summary_df.empty:
        summary_df = summary_df.sort_values(['lab', 'signal', 'stage'])
    
    summary_file = output_dir / 'lab_differences_detailed.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"Detailed lab statistics saved to: {summary_file}")
    
    # Create high-level summary
    if not summary_df.empty:
        high_level_summary = []
        
        for lab in summary_df['lab'].unique():
            lab_data = summary_df[summary_df['lab'] == lab]
            
            # Calculate lab-level metrics
            lab_summary = {
                'Lab': lab,
                'Total_Recordings': lab_data['n_recordings'].sum(),
                'Mean_EEG_Amplitude': lab_data[lab_data['signal'].str.contains('EEG')]['mean_amplitude'].mean(),
                'EEG_Amplitude_CV': lab_data[lab_data['signal'].str.contains('EEG')]['cv_amplitude'].mean(),
                'Mean_EMG_Amplitude': lab_data[lab_data['signal'] == 'EMG']['mean_amplitude'].mean(),
                'EMG_Amplitude_CV': lab_data[lab_data['signal'] == 'EMG']['cv_amplitude'].mean(),
            }
            
            # Add frequency band information if available
            for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
                band_col = f'{band}_power_mean'
                if band_col in lab_data.columns:
                    lab_summary[f'Mean_{band.capitalize()}_Power'] = lab_data[band_col].mean()
            
            high_level_summary.append(lab_summary)
        
        high_level_df = pd.DataFrame(high_level_summary)
        high_level_file = output_dir / 'lab_differences_summary.csv'
        high_level_df.to_csv(high_level_file, index=False)
        print(f"Lab summary saved to: {high_level_file}")
    
    print(f"Lab difference analysis completed. Results saved to: {output_dir}")


def create_lab_psd_comparison(lab_stats, output_dir):
    """Create detailed PSD comparison across labs (0.5–45 Hz) with 95% CI similar to stage PSD analysis.

    Aggregates EEG channel PSDs across stages per lab, interpolates to a common frequency grid,
    and produces four subplots: overall, low (0.5–10 Hz), high (10–45 Hz), and normalized PSD.
    """
    print("Creating lab PSD comparison (detailed with CI)...")
    # Collect PSDs per lab
    lab_psds = {}
    lab_freqs = {}
    # Common grid (0.5–45 Hz, 0.5 Hz resolution)
    common_freqs = np.arange(0.5, 45.5, 0.5)

    for key, data_list in lab_stats.items():
        # Instead of parsing key (which embeds underscores inside lab name), rely on stored entry metadata
        for entry in data_list:
            lab = entry.get('lab')
            signal = entry.get('signal')
            if lab is None or signal is None or not str(signal).startswith('EEG'):
                continue
            psd = entry.get('psd')
            freqs = entry.get('frequencies')
            if psd is None or freqs is None:
                continue
            mask = (freqs >= 0.5) & (freqs <= 45)
            if not np.any(mask):
                continue
            try:
                interp_psd = np.interp(common_freqs, freqs[mask], psd[mask], left=np.nan, right=np.nan)
            except Exception:
                continue
            interp_psd = np.nan_to_num(interp_psd, nan=0.0)
            lab_psds.setdefault(lab, []).append(interp_psd)
            lab_freqs[lab] = common_freqs

    if not lab_psds:
        print("No PSD data available for lab PSD comparison.")
        return

    labs = sorted(lab_psds.keys())
    # Prepare figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.reshape(2,2)
    color_palette = sns.color_palette('tab20', n_colors=len(labs))
    colors = {lab: color_palette[i] for i, lab in enumerate(labs)}

    def compute_ci(arr_list):
        arr = np.array(arr_list)
        if arr.shape[0] == 0:
            return None, None, None
        mean_psd = arr.mean(axis=0)
        if arr.shape[0] == 1:
            return mean_psd, mean_psd, mean_psd
        # 95% CI via t approx
        sem = arr.std(axis=0, ddof=1)/np.sqrt(arr.shape[0])
        t = 1.96
        return mean_psd, mean_psd - t*sem, mean_psd + t*sem

    # Overall (log scale)
    ax = axes[0,0]
    for lab in labs:
        mean_psd, ci_low, ci_high = compute_ci(lab_psds[lab])
        if mean_psd is None: continue
        ax.plot(common_freqs, mean_psd, label=lab, color=colors[lab], linewidth=2)
        ax.fill_between(common_freqs, ci_low, ci_high, color=colors[lab], alpha=0.15)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD (linear)')
    ax.set_yscale('log')
    ax.set_title('Lab Mean PSD (0.5–45 Hz) ±95% CI')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    # Low freq 0.5–10 Hz
    ax = axes[0,1]
    low_mask = (common_freqs >= 0.5) & (common_freqs <= 10)
    for lab in labs:
        mean_psd, ci_low, ci_high = compute_ci(lab_psds[lab])
        if mean_psd is None: continue
        ax.plot(common_freqs[low_mask], mean_psd[low_mask], label=lab, color=colors[lab], linewidth=2)
        ax.fill_between(common_freqs[low_mask], ci_low[low_mask], ci_high[low_mask], color=colors[lab], alpha=0.15)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD')
    ax.set_title('Low Frequency (0.5–10 Hz) ±95% CI')
    ax.grid(True, alpha=0.3)

    # High freq 10–45 Hz
    ax = axes[1,0]
    high_mask = (common_freqs >= 10) & (common_freqs <= 45)
    for lab in labs:
        mean_psd, ci_low, ci_high = compute_ci(lab_psds[lab])
        if mean_psd is None: continue
        ax.plot(common_freqs[high_mask], mean_psd[high_mask], label=lab, color=colors[lab], linewidth=2)
        ax.fill_between(common_freqs[high_mask], ci_low[high_mask], ci_high[high_mask], color=colors[lab], alpha=0.15)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD')
    ax.set_title('High Frequency (10–45 Hz) ±95% CI')
    ax.grid(True, alpha=0.3)

    # Normalized PSD
    ax = axes[1,1]
    norm_mask = (common_freqs >= 0.5) & (common_freqs <= 45)
    for lab in labs:
        arr = np.array(lab_psds[lab])
        if arr.shape[0] == 0:
            continue
        # Normalize each sample first
        norm_samples = []
        for psd in arr:
            total = psd[norm_mask].sum()
            if total > 0:
                norm_samples.append(psd[norm_mask]/total)
        if not norm_samples:
            continue
        mean_psd, ci_low, ci_high = compute_ci(norm_samples)
        f = common_freqs[norm_mask]
        ax.plot(f, mean_psd, label=lab, color=colors[lab], linewidth=2)
        ax.fill_between(f, ci_low, ci_high, color=colors[lab], alpha=0.15)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Normalized PSD')
    ax.set_title('Normalized PSD (0.5–45 Hz) ±95% CI')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'lab_detailed_psd_analysis_with_CI.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Lab PSD comparison figure saved.")


if __name__ == "__main__":
    analyze_lab_differences()
