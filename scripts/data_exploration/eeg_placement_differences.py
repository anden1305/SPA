"""
Analysis of signal differences between EEG placements.
This module focuses on identifying and visualizing the key signal characteristics
that distinguish different EEG electrode placements across sleep stages.
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

def analyze_eeg_placement_differences(output_dir="results/data_exploration"):
    """
    Comprehensive analysis of signal differences between EEG placements.
    
    Args:
        output_dir (str): Directory to save results
    """
    print("=" * 60)
    print("ANALYZING SIGNAL DIFFERENCES BETWEEN EEG PLACEMENTS")
    print("=" * 60)
    
    # Ensure output directory exists
    output_dir = ensure_output_directory(output_dir)
    placement_output_dir = Path(output_dir) / "eeg_placement_differences"
    placement_output_dir.mkdir(exist_ok=True)
    
    # Load metadata
    metadata = load_metadata()
    # Exclude lab_1 as requested
    initial_count = metadata.shape[0]
    metadata = metadata[metadata['lab'] != 'lab_1']
    removed = initial_count - metadata.shape[0]
    print(f"Excluded lab_1 from EEG placement analysis (removed {removed} records).")
    
    # Analyze EEG placement types
    placement_types = analyze_placement_types(metadata)
    print(f"Identified EEG placement types: {list(placement_types.keys())}")
    
    # Select representative participants for each placement type
    selected_participants = select_placement_representatives(metadata, placement_types)
    
    # Collect comprehensive data from all EEG placements
    all_placement_stats = {}
    placement_summary_data = {}
    
    for placement_type, participants in selected_participants.items():
        print(f"\nProcessing {placement_type} placement with {len(participants)} participants...")
        
        placement_data = {}
        
        for participant_id, lab in participants:
            print(f"  Processing {participant_id} from {lab}...")
            participant_runs = metadata[metadata['participant_id'] == participant_id]
            
            for _, run_info in participant_runs.iterrows():
                run_id = run_info['run']
                
                try:
                    # Load participant data
                    data = load_participant_data(participant_id, run_id)
                    labels = data['labels']
                    
                    # Get placement information for this run
                    run_placements = get_run_placements(run_info)
                    
                    # Process each EEG signal with its placement
                    for signal_name, signal_data in data.items():
                        if signal_name == 'labels' or 'EMG' in signal_name:
                            continue
                        
                        # Get the placement for this specific signal
                        signal_placement = run_placements.get(signal_name)
                        if not signal_placement:
                            continue
                        
                        print(f"    Analyzing {signal_name} ({signal_placement})...")
                        
                        # Calculate statistics for the entire signal
                        stats_result = calculate_signal_statistics(signal_data, labels)
                        
                        # Store statistics for each stage
                        for stage_name, stage_stats in stats_result.items():
                            if stage_name == 'Artifact':
                                continue
                            key = f"{placement_type}_{signal_placement}_{stage_name}"
                            if key not in placement_data:
                                placement_data[key] = []
                            
                            placement_data[key].append({
                                'placement_type': placement_type,
                                'signal_placement': signal_placement,
                                'participant_id': participant_id,
                                'lab': lab,
                                'signal_name': signal_name,
                                'stage': stage_name,
                                **stage_stats
                            })
                    
                except Exception as e:
                    print(f"      Error processing {participant_id} run {run_id}: {e}")
                    continue
        
        # Aggregate placement statistics
        for key, data_list in placement_data.items():
            if data_list:  # Only process if we have data
                all_placement_stats[key] = data_list
                
                # Calculate placement-level aggregated statistics
                placement_type, signal_placement, stage_name = key.rsplit('_', 2)
                
                # Aggregate key metrics
                means = [d['mean'] for d in data_list]
                stds = [d['std'] for d in data_list]
                spectral_powers = [d.get('spectral_power', {}) for d in data_list if d.get('spectral_power')]
                
                summary_key = f"{placement_type}_{signal_placement}_{stage_name}"
                placement_summary_data[summary_key] = {
                    'placement_type': placement_type,
                    'signal_placement': signal_placement,
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
                            placement_summary_data[summary_key][f'{band}_power_mean'] = np.mean(band_powers)
                            placement_summary_data[summary_key][f'{band}_power_std'] = np.std(band_powers)
                            placement_summary_data[summary_key][f'{band}_power_cv'] = np.std(band_powers) / np.mean(band_powers) if np.mean(band_powers) != 0 else 0
    
    if not all_placement_stats:
        print("No valid data found for analysis!")
        return
    
    print(f"\nCollected data from {len(selected_participants)} placement types")
    
    # Create visualizations
    create_placement_signal_overview(all_placement_stats, placement_output_dir)
    create_placement_frequency_analysis(all_placement_stats, placement_output_dir)
    create_placement_stage_comparison(all_placement_stats, placement_output_dir)
    create_placement_band_power_analysis(all_placement_stats, placement_output_dir)
    create_placement_spatial_analysis(all_placement_stats, placement_output_dir)
    
    # Create summary statistics
    create_placement_summary_statistics(placement_summary_data, all_placement_stats, placement_output_dir)
    
    # New analyses: Detailed PSD and extended band power comparisons
    create_placement_detailed_psd_analysis(all_placement_stats, placement_output_dir)
    create_placement_band_power_differences(all_placement_stats, placement_output_dir)
    
    print(f"\nEEG placement difference analysis completed. Results saved to: {placement_output_dir}")


def analyze_placement_types(metadata):
    """
    Analyze the different EEG placement types in the dataset.
    
    Args:
        metadata (pd.DataFrame): Participant metadata
    
    Returns:
        dict: Dictionary mapping placement types to their descriptions
    """
    placement_types = {}
    
    for _, row in metadata.iterrows():
        # Check each EEG channel and its type
        for i in range(1, 5):  # EEG1 to EEG4
            eeg_col = f'EEG{i}'
            type_col = f'EEG{i}_TYPE'
            
            if row[eeg_col] and pd.notna(row[type_col]) and row[type_col] != '':
                placement = row[type_col].strip()
                
                if placement not in placement_types:
                    placement_types[placement] = {
                        'description': placement,
                        'participants': set(),
                        'channels': set()
                    }
                
                placement_types[placement]['participants'].add(row['participant_id'])
                placement_types[placement]['channels'].add(f'EEG{i}')
    
    # Convert sets to lists for easier handling
    for placement in placement_types:
        placement_types[placement]['participants'] = list(placement_types[placement]['participants'])
        placement_types[placement]['channels'] = list(placement_types[placement]['channels'])
    
    return placement_types


def get_run_placements(run_info):
    """
    Get the EEG placements for a specific run.
    
    Args:
        run_info (pd.Series): Run information from metadata
    
    Returns:
        dict: Mapping of signal names to their placements
    """
    placements = {}
    
    for i in range(1, 5):  # EEG1 to EEG4
        eeg_col = f'EEG{i}'
        type_col = f'EEG{i}_TYPE'
        
        if run_info[eeg_col] and pd.notna(run_info[type_col]) and run_info[type_col] != '':
            placements[eeg_col] = run_info[type_col].strip()
    
    return placements


def select_placement_representatives(metadata, placement_types, participants_per_type=15):
    """
    Select representative participants for each EEG placement type.
    
    Args:
        metadata (pd.DataFrame): Participant metadata
        placement_types (dict): Available placement types
        participants_per_type (int): Number of participants to select per type
    
    Returns:
        dict: Dictionary mapping placement types to selected participants
    """
    selected_participants = {}
    
    for placement_type, info in placement_types.items():
        participants = info['participants']
        
        # Get lab information for these participants
        participants_with_labs = []
        for participant_id in participants:
            lab = metadata[metadata['participant_id'] == participant_id]['lab'].iloc[0]
            participants_with_labs.append((participant_id, lab))
        
        # Select up to participants_per_type
        n_select = min(participants_per_type, len(participants_with_labs))
        selected = sorted(participants_with_labs)[:n_select]
        
        selected_participants[placement_type] = selected
        
        print(f"Selected {n_select} participants for {placement_type}: {[p[0] for p in selected]}")
    
    return selected_participants


def create_placement_signal_overview(placement_stats, output_dir):
    """Create overview plots comparing signal characteristics across EEG placements."""
    print("Creating EEG placement signal overview...")
    
    # Prepare data for plotting
    plot_data = []
    for key, data_list in placement_stats.items():
        for data_point in data_list:
            plot_data.append(data_point)
    
    df = pd.DataFrame(plot_data)
    
    if df.empty:
        print("No data available for placement signal overview")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('EEG Signal Characteristics Across Placements', fontsize=16, fontweight='bold')
    
    # Plot 1: Mean amplitude by placement and stage
    ax1 = axes[0, 0]
    if 'mean' in df.columns and 'signal_placement' in df.columns:
        sns.boxplot(data=df, x='signal_placement', y='mean', hue='stage', ax=ax1)
        ax1.set_title('Signal Amplitude by EEG Placement')
        ax1.set_ylabel('Mean Amplitude')
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend(title='Sleep Stage', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 2: Signal variability by placement
    ax2 = axes[0, 1]
    if 'std' in df.columns:
        sns.boxplot(data=df, x='signal_placement', y='std', hue='stage', ax=ax2)
        ax2.set_title('Signal Variability by EEG Placement')
        ax2.set_ylabel('Standard Deviation')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend(title='Sleep Stage', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 3: SNR comparison across placements
    ax3 = axes[0, 2]
    if 'snr' in df.columns:
        sns.boxplot(data=df, x='signal_placement', y='snr', hue='stage', ax=ax3)
        ax3.set_title('Signal-to-Noise Ratio by EEG Placement')
        ax3.set_ylabel('SNR (dB)')
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend(title='Sleep Stage', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 4: Kurtosis by placement
    ax4 = axes[1, 0]
    if 'kurtosis' in df.columns:
        sns.boxplot(data=df, x='signal_placement', y='kurtosis', hue='stage', ax=ax4)
        ax4.set_title('Signal Distribution Kurtosis by EEG Placement')
        ax4.set_ylabel('Kurtosis')
        ax4.tick_params(axis='x', rotation=45)
        ax4.legend(title='Sleep Stage', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 5: Skewness by placement
    ax5 = axes[1, 1]
    if 'skewness' in df.columns:
        sns.boxplot(data=df, x='signal_placement', y='skewness', hue='stage', ax=ax5)
        ax5.set_title('Signal Distribution Skewness by EEG Placement')
        ax5.set_ylabel('Skewness')
        ax5.tick_params(axis='x', rotation=45)
        ax5.legend(title='Sleep Stage', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 6: Coefficient of variation
    ax6 = axes[1, 2]
    if 'mean' in df.columns and 'std' in df.columns:
        df['cv'] = df['std'] / df['mean'].abs()
        df['cv'] = df['cv'].replace([np.inf, -np.inf], np.nan)
        sns.boxplot(data=df, x='signal_placement', y='cv', hue='stage', ax=ax6)
        ax6.set_title('Coefficient of Variation by EEG Placement')
        ax6.set_ylabel('CV (std/|mean|)')
        ax6.tick_params(axis='x', rotation=45)
        ax6.legend(title='Sleep Stage', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'eeg_placement_signal_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    # plt.show()  # Disabled for automated analysis


def create_placement_frequency_analysis(placement_stats, output_dir):
    """Create frequency domain analysis comparing EEG placements."""
    print("Creating EEG placement frequency analysis...")
    
    # Collect spectral data (derive from flat *_power keys if nested dict absent)
    spectral_data = []
    band_key_map = {
        'delta': 'delta_power',
        'theta': 'theta_power',
        'alpha': 'alpha_power',
        'beta': 'beta_power',
        'gamma': 'gamma_power'
    }
    for key, data_list in placement_stats.items():
        for data_point in data_list:
            # Prefer nested spectral_power if present, else build from keys
            if 'spectral_power' in data_point and data_point['spectral_power']:
                for band, power in data_point['spectral_power'].items():
                    spectral_data.append({
                        'placement_type': data_point['placement_type'],
                        'signal_placement': data_point['signal_placement'],
                        'stage': data_point['stage'],
                        'participant_id': data_point['participant_id'],
                        'lab': data_point['lab'],
                        'frequency_band': band,
                        'power': power
                    })
            else:
                for band, flat_key in band_key_map.items():
                    if flat_key in data_point and data_point[flat_key] is not None:
                        spectral_data.append({
                            'placement_type': data_point['placement_type'],
                            'signal_placement': data_point['signal_placement'],
                            'stage': data_point['stage'],
                            'participant_id': data_point['participant_id'],
                            'lab': data_point['lab'],
                            'frequency_band': band,
                            'power': data_point[flat_key]
                        })
    
    if not spectral_data:
        print("No spectral data available for frequency analysis")
        return
    
    df_spectral = pd.DataFrame(spectral_data)
    
    # Create frequency analysis plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Frequency Domain Analysis Across EEG Placements', fontsize=16, fontweight='bold')
    
    # Plot 1: Delta band power across placements
    ax1 = axes[0, 0]
    delta_data = df_spectral[df_spectral['frequency_band'] == 'delta']
    if not delta_data.empty:
        sns.boxplot(data=delta_data, x='signal_placement', y='power', hue='stage', ax=ax1)
        ax1.set_title('Delta Band Power (0.5-4 Hz)')
        ax1.set_ylabel('Power')
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend(title='Sleep Stage', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 2: Theta band power across placements
    ax2 = axes[0, 1]
    theta_data = df_spectral[df_spectral['frequency_band'] == 'theta']
    if not theta_data.empty:
        sns.boxplot(data=theta_data, x='signal_placement', y='power', hue='stage', ax=ax2)
        ax2.set_title('Theta Band Power (4-8 Hz)')
        ax2.set_ylabel('Power')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend(title='Sleep Stage', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 3: Alpha band power across placements
    ax3 = axes[0, 2]
    alpha_data = df_spectral[df_spectral['frequency_band'] == 'alpha']
    if not alpha_data.empty:
        sns.boxplot(data=alpha_data, x='signal_placement', y='power', hue='stage', ax=ax3)
        ax3.set_title('Alpha Band Power (8-12 Hz)')
        ax3.set_ylabel('Power')
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend(title='Sleep Stage', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 4: Beta band power across placements
    ax4 = axes[1, 0]
    beta_data = df_spectral[df_spectral['frequency_band'] == 'beta']
    if not beta_data.empty:
        sns.boxplot(data=beta_data, x='signal_placement', y='power', hue='stage', ax=ax4)
        ax4.set_title('Beta Band Power (13-30 Hz)')
        ax4.set_ylabel('Power')
        ax4.tick_params(axis='x', rotation=45)
        ax4.legend(title='Sleep Stage', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 5: Gamma band power across placements
    ax5 = axes[1, 1]
    gamma_data = df_spectral[df_spectral['frequency_band'] == 'gamma']
    if not gamma_data.empty:
        sns.boxplot(data=gamma_data, x='signal_placement', y='power', hue='stage', ax=ax5)
        ax5.set_title('Gamma Band Power (30-50 Hz)')
        ax5.set_ylabel('Power')
        ax5.tick_params(axis='x', rotation=45)
        ax5.legend(title='Sleep Stage', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 6: Spectral power heatmap
    ax6 = axes[1, 2]
    # Create a pivot table for the heatmap
    pivot_data = df_spectral.groupby(['signal_placement', 'frequency_band'])['power'].mean().unstack()
    if not pivot_data.empty:
        sns.heatmap(pivot_data, annot=True, fmt='.2e', cmap='viridis', ax=ax6)
        ax6.set_title('Mean Spectral Power by Placement')
        ax6.set_ylabel('EEG Placement')
        ax6.set_xlabel('Frequency Band')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'eeg_placement_frequency_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    # plt.show()  # Disabled for automated analysis


def create_placement_stage_comparison(placement_stats, output_dir):
    """Create stage-specific comparison across EEG placements."""
    print("Creating EEG placement stage comparison...")
    
    # Prepare data for plotting
    plot_data = []
    for key, data_list in placement_stats.items():
        for data_point in data_list:
            plot_data.append(data_point)
    
    df = pd.DataFrame(plot_data)
    
    if df.empty:
        print("No data available for placement stage comparison")
        return
    
    # Create stage-specific plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Sleep Stage Comparison Across EEG Placements', fontsize=16, fontweight='bold')
    
    # Plot 1: NREM stage analysis
    ax1 = axes[0, 0]
    nrem_data = df[df['stage'] == 'NREM']
    if not nrem_data.empty and 'mean' in nrem_data.columns:
        sns.boxplot(data=nrem_data, x='signal_placement', y='mean', ax=ax1)
        ax1.set_title('NREM Stage Signal Amplitude by EEG Placement')
        ax1.set_ylabel('Mean Amplitude (NREM)')
        ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: REM stage analysis
    ax2 = axes[0, 1]
    rem_data = df[df['stage'] == 'REM']
    if not rem_data.empty and 'mean' in rem_data.columns:
        sns.boxplot(data=rem_data, x='signal_placement', y='mean', ax=ax2)
        ax2.set_title('REM Stage Signal Amplitude by EEG Placement')
        ax2.set_ylabel('Mean Amplitude (REM)')
        ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Awake stage analysis
    ax3 = axes[1, 0]
    awake_data = df[df['stage'] == 'Awake']
    if not awake_data.empty and 'mean' in awake_data.columns:
        sns.boxplot(data=awake_data, x='signal_placement', y='mean', ax=ax3)
        ax3.set_title('Awake Stage Signal Amplitude by EEG Placement')
        ax3.set_ylabel('Mean Amplitude (Awake)')
        ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Stage discrimination by placement
    ax4 = axes[1, 1]
    if 'mean' in df.columns:
        # Calculate stage discrimination index (difference between max and min stage amplitudes)
        discrimination_data = []
        for placement in df['signal_placement'].unique():
            placement_data = df[df['signal_placement'] == placement]
            stage_means = placement_data.groupby('stage')['mean'].mean()
            if len(stage_means) > 1:
                discrimination_index = stage_means.max() - stage_means.min()
                discrimination_data.append({
                    'signal_placement': placement,
                    'discrimination_index': discrimination_index
                })
        
        if discrimination_data:
            disc_df = pd.DataFrame(discrimination_data)
            sns.barplot(data=disc_df, x='signal_placement', y='discrimination_index', ax=ax4)
            ax4.set_title('Stage Discrimination by EEG Placement')
            ax4.set_ylabel('Discrimination Index (max - min amplitude)')
            ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'eeg_placement_stage_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    # plt.show()  # Disabled for automated analysis


def create_placement_band_power_analysis(placement_stats, output_dir):
    """Create detailed band power analysis across EEG placements."""
    print("Creating EEG placement band power analysis...")
    
    # Collect detailed spectral data
    band_power_data = []
    for key, data_list in placement_stats.items():
        for data_point in data_list:
            base_info = {
                'placement_type': data_point['placement_type'],
                'signal_placement': data_point['signal_placement'],
                'stage': data_point['stage'],
                'participant_id': data_point['participant_id'],
                'lab': data_point['lab']
            }
            record = base_info.copy()
            added = False
            # Check nested dict first
            if 'spectral_power' in data_point and data_point['spectral_power']:
                for band, power in data_point['spectral_power'].items():
                    record[band] = power
                    added = True
            else:
                # Use flat keys
                for band, flat_key in [('delta','delta_power'),('theta','theta_power'),('alpha','alpha_power'),('beta','beta_power'),('gamma','gamma_power')]:
                    if flat_key in data_point:
                        record[band] = data_point[flat_key]
                        added = True
            if added:
                band_power_data.append(record)
    
    if not band_power_data:
        print("No band power data available for analysis")
        return
    
    df_bands = pd.DataFrame(band_power_data)
    
    # Create band power analysis plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('EEG Band Power Analysis Across Placements', fontsize=16, fontweight='bold')
    
    # Plot 1: Delta/Theta ratio (sleep indicator)
    ax1 = axes[0, 0]
    if 'delta' in df_bands.columns and 'theta' in df_bands.columns:
        df_bands['delta_theta_ratio'] = df_bands['delta'] / (df_bands['theta'] + 1e-10)
        sns.boxplot(data=df_bands, x='signal_placement', y='delta_theta_ratio', hue='stage', ax=ax1)
        ax1.set_title('Delta/Theta Ratio by EEG Placement')
        ax1.set_ylabel('Delta/Theta Ratio')
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend(title='Sleep Stage', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 2: Alpha/Beta ratio (arousal indicator)
    ax2 = axes[0, 1]
    if 'alpha' in df_bands.columns and 'beta' in df_bands.columns:
        df_bands['alpha_beta_ratio'] = df_bands['alpha'] / (df_bands['beta'] + 1e-10)
        sns.boxplot(data=df_bands, x='signal_placement', y='alpha_beta_ratio', hue='stage', ax=ax2)
        ax2.set_title('Alpha/Beta Ratio by EEG Placement')
        ax2.set_ylabel('Alpha/Beta Ratio')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend(title='Sleep Stage', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 3: Total power by placement
    ax3 = axes[1, 0]
    band_columns = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    available_bands = [col for col in band_columns if col in df_bands.columns]
    
    if available_bands:
        df_bands['total_power'] = df_bands[available_bands].sum(axis=1)
        sns.boxplot(data=df_bands, x='signal_placement', y='total_power', hue='stage', ax=ax3)
        ax3.set_title('Total Spectral Power by EEG Placement')
        ax3.set_ylabel('Total Power')
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend(title='Sleep Stage', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 4: Band power distribution
    ax4 = axes[1, 1]
    if available_bands:
        # Calculate relative power for each band
        for band in available_bands:
            df_bands[f'{band}_relative'] = df_bands[band] / df_bands['total_power']
        
        # Create stacked bar plot
        placement_band_means = df_bands.groupby('signal_placement')[[f'{band}_relative' for band in available_bands]].mean()
        placement_band_means.plot(kind='bar', stacked=True, ax=ax4)
        ax4.set_title('Relative Band Power Distribution by EEG Placement')
        ax4.set_ylabel('Relative Power')
        ax4.set_xlabel('EEG Placement')
        ax4.tick_params(axis='x', rotation=45)
        ax4.legend(title='Frequency Band', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'eeg_placement_band_power_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    # plt.show()  # Disabled for automated analysis


def create_placement_spatial_analysis(placement_stats, output_dir):
    """Create spatial analysis of EEG placements and their characteristics."""
    print("Creating EEG placement spatial analysis...")
    
    # Prepare data for spatial analysis
    plot_data = []
    for key, data_list in placement_stats.items():
        for data_point in data_list:
            plot_data.append(data_point)
    
    df = pd.DataFrame(plot_data)
    
    if df.empty:
        print("No data available for spatial analysis")
        return
    
    # Create spatial analysis plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('EEG Placement Spatial Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Frontal vs Parietal comparison
    ax1 = axes[0, 0]
    frontal_data = df[df['signal_placement'].str.contains('F', na=False)]
    parietal_data = df[df['signal_placement'].str.contains('P', na=False)]
    
    if not frontal_data.empty and not parietal_data.empty:
        comparison_data = []
        for _, row in frontal_data.iterrows():
            comparison_data.append({**dict(row), 'region': 'Frontal'})
        for _, row in parietal_data.iterrows():
            comparison_data.append({**dict(row), 'region': 'Parietal'})
        
        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)
            sns.boxplot(data=comp_df, x='region', y='mean', hue='stage', ax=ax1)
            ax1.set_title('Frontal vs Parietal EEG Amplitude')
            ax1.set_ylabel('Mean Amplitude')
            ax1.legend(title='Sleep Stage')
    
    # Plot 2: Central vs Peripheral placement analysis
    ax2 = axes[0, 1]
    if 'mean' in df.columns:
        # Group placements by central vs peripheral
        central_placements = ['EEG PFCF']  # Central placements
        peripheral_placements = ['EEG P', 'EEG F']  # Peripheral placements
        
        region_data = []
        for _, row in df.iterrows():
            if row['signal_placement'] in central_placements:
                region_data.append({**dict(row), 'position': 'Central'})
            elif row['signal_placement'] in peripheral_placements:
                region_data.append({**dict(row), 'position': 'Peripheral'})
        
        if region_data:
            region_df = pd.DataFrame(region_data)
            sns.boxplot(data=region_df, x='position', y='mean', hue='stage', ax=ax2)
            ax2.set_title('Central vs Peripheral EEG Amplitude')
            ax2.set_ylabel('Mean Amplitude')
            ax2.legend(title='Sleep Stage')
    
    # Plot 3: Signal quality by placement
    ax3 = axes[1, 0]
    if 'snr' in df.columns:
        placement_quality = df.groupby('signal_placement')['snr'].mean().sort_values(ascending=False)
        sns.barplot(x=placement_quality.index, y=placement_quality.values, ax=ax3)
        ax3.set_title('Signal Quality (SNR) by EEG Placement')
        ax3.set_ylabel('Mean SNR (dB)')
        ax3.set_xlabel('EEG Placement')
        ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Placement consistency analysis
    ax4 = axes[1, 1]
    if 'std' in df.columns and 'mean' in df.columns:
        df['cv'] = df['std'] / df['mean'].abs()
        df['cv'] = df['cv'].replace([np.inf, -np.inf], np.nan)
        
        placement_consistency = df.groupby('signal_placement')['cv'].mean().sort_values()
        sns.barplot(x=placement_consistency.index, y=placement_consistency.values, ax=ax4)
        ax4.set_title('Signal Consistency by EEG Placement')
        ax4.set_ylabel('Mean Coefficient of Variation')
        ax4.set_xlabel('EEG Placement')
        ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'eeg_placement_spatial_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    # plt.show()  # Disabled for automated analysis


def create_placement_summary_statistics(placement_summary_data, all_placement_stats, output_dir):
    """Create comprehensive summary statistics for EEG placement differences."""
    print("Creating EEG placement summary statistics...")
    
    # Create detailed summary CSV
    summary_df = pd.DataFrame(placement_summary_data).T
    summary_df = summary_df.reset_index(drop=True)
    
    # Sort by placement type and stage for better organization
    if not summary_df.empty:
        summary_df = summary_df.sort_values(['placement_type', 'signal_placement', 'stage'])
    
    summary_file = output_dir / 'eeg_placement_differences_detailed.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"Detailed EEG placement statistics saved to: {summary_file}")
    
    # Create high-level summary
    if not summary_df.empty:
        high_level_summary = []
        
        for placement in summary_df['signal_placement'].unique():
            placement_data = summary_df[summary_df['signal_placement'] == placement]
            
            # Calculate placement-level metrics
            placement_summary = {
                'EEG_Placement': placement,
                'Total_Recordings': placement_data['n_recordings'].sum(),
                'Mean_Amplitude': placement_data['mean_amplitude'].mean(),
                'Amplitude_CV': placement_data['cv_amplitude'].mean(),
                'SNR_Quality': placement_data.get('snr_mean', pd.Series([np.nan])).mean(),
            }
            
            # Add frequency band information if available
            for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
                band_col = f'{band}_power_mean'
                if band_col in placement_data.columns:
                    placement_summary[f'Mean_{band.capitalize()}_Power'] = placement_data[band_col].mean()
            
            # Add stage-specific metrics
            for stage in STAGE_ORDER:
                stage_data = placement_data[placement_data['stage'] == stage]
                if not stage_data.empty:
                    placement_summary[f'{stage}_Amplitude'] = stage_data['mean_amplitude'].mean()
            
            high_level_summary.append(placement_summary)
        
        high_level_df = pd.DataFrame(high_level_summary)
        high_level_file = output_dir / 'eeg_placement_differences_summary.csv'
        high_level_df.to_csv(high_level_file, index=False)
        print(f"EEG placement summary saved to: {high_level_file}")
    
    print(f"EEG placement difference analysis completed. Results saved to: {output_dir}")


def create_placement_detailed_psd_analysis(all_placement_stats, output_dir):
    """Create detailed PSD comparison across EEG placements with confidence intervals."""
    print("Creating detailed PSD analysis across EEG placements...")
    # Collect PSDs per placement per stage
    placement_stage_psds = {}
    placement_stage_freqs = {}
    for key, data_list in all_placement_stats.items():
        try:
            _, signal_placement, stage = key.rsplit('_', 2)
        except ValueError:
            continue
        for rec in data_list:
            psd = rec.get('psd')
            freqs = rec.get('frequencies')
            if psd is None or freqs is None:
                continue
            placement_key = (signal_placement, stage)
            if placement_key not in placement_stage_psds:
                placement_stage_psds[placement_key] = []
                placement_stage_freqs[placement_key] = freqs
            placement_stage_psds[placement_key].append(psd)
    if not placement_stage_psds:
        print("No PSD data available for placement detailed analysis.")
        return
    # Determine stages and placements present
    placements = sorted({p for (p, s) in placement_stage_psds.keys()})
    stages = [s for s in STAGE_ORDER if any(stage == s for (_, stage) in placement_stage_psds.keys())]
    # Plot PSD per stage comparing placements
    fig, axes = plt.subplots(len(stages), 2, figsize=(14, 4*len(stages)))
    if len(stages) == 1:
        axes = np.array([axes])
    fig.suptitle('EEG Placement PSD Comparison by Sleep Stage (95% CI)', fontsize=16, fontweight='bold')
    # Define common frequency grid for interpolation (0.5-45 Hz, 450 points)
    common_freqs = np.linspace(0.5, 45, 450)
    for i, stage in enumerate(stages):
        ax_full = axes[i, 0]
        ax_low = axes[i, 1]
        for placement in placements:
            key_tuple = (placement, stage)
            if key_tuple not in placement_stage_psds:
                continue
            psd_list = []
            freqs = placement_stage_freqs[key_tuple]
            if freqs is None:
                continue
            # Interpolate each PSD onto common grid
            for raw_psd in placement_stage_psds[key_tuple]:
                mask_valid = (freqs >= 0.5) & (freqs <= 45)
                try:
                    interp_psd = np.interp(common_freqs, freqs[mask_valid], raw_psd[mask_valid])
                    psd_list.append(interp_psd)
                except Exception:
                    continue
            if not psd_list:
                continue
            mean_psd, ci_lower, ci_upper = calculate_spectral_confidence_intervals(psd_list)
            mask_low = (common_freqs >= 0.5) & (common_freqs <= 10)
            color = None
            # Assign color by placement for consistency using seaborn palette
            palette = sns.color_palette('tab10', n_colors=len(placements))
            placement_index = placements.index(placement)
            color = palette[placement_index]
            # Full band (log scale)
            ax_full.semilogy(common_freqs, mean_psd, label=placement, color=color, linewidth=1.8)
            if ci_lower is not None and ci_upper is not None:
                ax_full.fill_between(common_freqs, ci_lower, ci_upper, color=color, alpha=0.15)
            # Low band
            ax_low.plot(common_freqs[mask_low], mean_psd[mask_low], label=placement, color=color, linewidth=1.8)
            if ci_lower is not None and ci_upper is not None:
                ax_low.fill_between(common_freqs[mask_low], ci_lower[mask_low], ci_upper[mask_low], color=color, alpha=0.15)
        ax_full.set_title(f'{stage} PSD (0.5-45 Hz)', fontweight='bold')
        ax_full.set_xlabel('Frequency (Hz)')
        ax_full.set_ylabel('PSD (log scale)')
        ax_full.grid(alpha=0.3)
        ax_low.set_title(f'{stage} Low-Frequency PSD (0.5-10 Hz)', fontweight='bold')
        ax_low.set_xlabel('Frequency (Hz)')
        ax_low.set_ylabel('PSD')
        ax_low.grid(alpha=0.3)
        if i == 0:
            ax_full.legend(title='Placement', fontsize=8)
            ax_low.legend(title='Placement', fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_dir / 'eeg_placement_detailed_psd.png', dpi=200, bbox_inches='tight')
    plt.close()
    # plt.show()

    # Aggregate placement-averaged PSD across all stages
    placement_psd_all = {}
    freqs_ref = None
    for (placement, stage), psd_list in placement_stage_psds.items():
        if placement not in placement_psd_all:
            placement_psd_all[placement] = []
        freqs = placement_stage_freqs[(placement, stage)]
        mask_valid = (freqs >= 0.5) & (freqs <= 45)
        try:
            psd_array = np.vstack([p[mask_valid] for p in psd_list if len(p)==len(freqs)])
            mean_raw = psd_array.mean(axis=0)
            interp_mean = np.interp(common_freqs, freqs[mask_valid], mean_raw)
            freqs_ref = common_freqs
            placement_psd_all[placement].append(interp_mean)
        except Exception:
            continue
    # Compute overall mean per placement
    combined = {}
    for placement, psd_lists in placement_psd_all.items():
        combined[placement] = np.mean(np.vstack(psd_lists), axis=0)
    if combined and freqs_ref is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        mask = slice(None)
        palette = sns.color_palette('tab10', n_colors=len(combined))
        for idx, (placement, psd_mean) in enumerate(combined.items()):
            ax.semilogy(freqs_ref[mask], psd_mean[mask], label=placement, linewidth=2, color=palette[idx])
        ax.set_title('Overall Mean PSD by EEG Placement (0.5-45 Hz)', fontweight='bold')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD (log scale)')
        ax.grid(alpha=0.3)
        ax.legend(title='Placement')
        plt.tight_layout()
        plt.savefig(output_dir / 'eeg_placement_overall_psd.png', dpi=200, bbox_inches='tight')
        plt.close()
        # plt.show()

def create_placement_band_power_differences(all_placement_stats, output_dir):
    """Extended band power comparisons across placements with statistical tests."""
    print("Creating extended band power comparison across placements...")
    records = []
    for key, data_list in all_placement_stats.items():
        try:
            placement_type, signal_placement, stage = key.rsplit('_', 2)
        except ValueError:
            continue
        for rec in data_list:
            sp = rec.get('spectral_power')
            if not sp:
                continue
            row = {
                'signal_placement': signal_placement,
                'stage': stage,
                'participant_id': rec.get('participant_id')
            }
            row.update({band: sp.get(band) for band in ['delta','theta','alpha','beta','gamma'] if band in sp})
            records.append(row)
    if not records:
        print("No spectral power records for extended comparison.")
        return
    df = pd.DataFrame(records)
    # Melt for combined violin/box plots
    melt = df.melt(id_vars=['signal_placement','stage','participant_id'], var_name='band', value_name='power')
    # Plot per band across placements
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.violinplot(data=melt, x='band', y='power', hue='signal_placement', split=False, inner='quart', ax=ax)
    ax.set_title('Band Power Distribution by Placement (All Stages)', fontweight='bold')
    ax.set_ylabel('Power')
    ax.grid(alpha=0.3)
    ax.legend(title='Placement', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_dir / 'eeg_placement_band_power_distribution.png', dpi=180, bbox_inches='tight')
    plt.close()
    # Stage-specific comparison grids
    stages = sorted(df['stage'].unique())
    bands = ['delta','theta','alpha','beta','gamma']
    for band in bands:
        band_df = melt[melt['band']==band]
        if band_df.empty:
            continue
        fig, axes = plt.subplots(1, len(stages), figsize=(4*len(stages), 5), sharey=True)
        if len(stages)==1:
            axes = [axes]
        for i, stage in enumerate(stages):
            sd = band_df[band_df['stage']==stage]
            sns.boxplot(data=sd, x='signal_placement', y='power', ax=axes[i])
            axes[i].set_title(f'{band.capitalize()} Power - {stage}')
            axes[i].tick_params(axis='x', rotation=45)
        fig.suptitle(f'{band.capitalize()} Band Power by Placement and Stage', fontweight='bold')
        plt.tight_layout(rect=[0,0,1,0.95])
        plt.savefig(output_dir / f'eeg_placement_{band}_power_by_stage.png', dpi=170, bbox_inches='tight')
        plt.close()
    # Statistical comparison (ANOVA) saving table
    stats_rows = []
    from scipy.stats import f_oneway
    for band in bands:
        band_df = melt[melt['band']==band]
        if band_df.empty:
            continue
        for stage in stages:
            subset = band_df[band_df['stage']==stage]
            groups = [g['power'].dropna().values for _, g in subset.groupby('signal_placement') if len(g['power'].dropna())>1]
            if len(groups) < 2:
                continue
            try:
                f_stat, p_val = f_oneway(*groups)
                stats_rows.append({'band': band, 'stage': stage, 'f_stat': f_stat, 'p_value': p_val})
            except Exception:
                continue
    if stats_rows:
        stats_df = pd.DataFrame(stats_rows)
        stats_df.to_csv(output_dir / 'eeg_placement_band_power_anova.csv', index=False)
        print('Saved band power ANOVA results.')

if __name__ == "__main__":
    analyze_eeg_placement_differences()
