"""
Analysis of signal differences between sleep stages.
This module focuses on identifying and visualizing the key signal characteristics
that distinguish different sleep stages.
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
    calculate_stage_transitions, STAGE_ORDER, downsample_labels
)

def analyze_stage_differences(output_dir="results/data_exploration_2"):
    """
    Comprehensive analysis of signal differences between sleep stages.
    
    Args:
        output_dir (str): Directory to save results
    """
    print("=" * 60)
    print("ANALYZING SIGNAL DIFFERENCES BETWEEN SLEEP STAGES")
    print("=" * 60)
    
    # Ensure output directory exists
    output_dir = ensure_output_directory(output_dir)
    stage_output_dir = Path(output_dir) / "stage_differences"
    stage_output_dir.mkdir(exist_ok=True)
    
    # Load metadata to get representative samples
    metadata = load_metadata()
    # Exclude lab_1 (outlier) from this analysis
    initial_rows = metadata.shape[0]
    metadata = metadata[metadata['lab'] != 'lab_1']
    removed_rows = initial_rows - metadata.shape[0]
    print(f"Excluded lab_1 from stage differences analysis (removed {removed_rows} records).")
    
    # Select representative participants from different labs for analysis
    selected_participants = []
    for lab in metadata['lab'].unique():
        lab_participants = metadata[metadata['lab'] == lab]['participant_id'].unique()
        # Select first participant from each lab for initial analysis
        selected_participants.append((lab_participants[0], lab))
    
    print(f"Selected participants for analysis: {[p[0] for p in selected_participants]}")
    
    # Collect signal statistics across participants
    all_eeg_stats = {}
    all_emg_stats = {}
    
    for participant_id, lab in selected_participants:
        print(f"\nProcessing {participant_id} from {lab}...")
        
        # Get participant runs
        participant_runs = metadata[metadata['participant_id'] == participant_id]
        
        for _, run_info in participant_runs.iterrows():
            run_id = run_info['run']
            
            try:
                # Load participant data
                data = load_participant_data(participant_id, run_id)
                
                # Analyze EEG channels
                for channel in ['EEG1', 'EEG2', 'EEG3', 'EEG4']:
                    if channel in data:
                        print(f"  Analyzing {channel}...")
                        
                        # Calculate statistics for this channel
                        eeg_stats = calculate_signal_statistics(
                            data[channel], data['labels'], sampling_rate=128
                        )
                        
                        # Store statistics
                        key = f"{participant_id}_{run_id}_{channel}"
                        all_eeg_stats[key] = eeg_stats
                
                # Analyze EMG if available
                if 'EMG' in data:
                    print(f"  Analyzing EMG...")
                    emg_stats = calculate_signal_statistics(
                        data['EMG'], data['labels'], sampling_rate=128
                    )
                    key = f"{participant_id}_{run_id}_EMG"
                    all_emg_stats[key] = emg_stats
                    
            except Exception as e:
                print(f"  Error processing {participant_id} run {run_id}: {e}")
                continue
    
    # Aggregate statistics across all recordings
    print(f"\nAggregating statistics from {len(all_eeg_stats)} EEG recordings...")
    
    aggregated_eeg_stats = aggregate_statistics_across_recordings(all_eeg_stats)
    aggregated_emg_stats = aggregate_statistics_across_recordings(all_emg_stats)
    
    # Create comprehensive visualizations
    create_stage_difference_plots(aggregated_eeg_stats, "EEG", stage_output_dir)
    if aggregated_emg_stats:
        create_stage_difference_plots(aggregated_emg_stats, "EMG", stage_output_dir)
    
    # Create frequency domain analysis
    create_frequency_domain_analysis(aggregated_eeg_stats, stage_output_dir)
    
    # Create detailed power spectral density analysis
    create_detailed_psd_analysis(all_eeg_stats, stage_output_dir)
    
    # Create fine-grained frequency band analysis (1 Hz bins)
    create_fine_grained_band_analysis(all_eeg_stats, stage_output_dir)
    
    # Create transition matrix analysis
    create_transition_analysis(selected_participants, metadata, stage_output_dir)
    
    # Generate summary tables
    create_comprehensive_summary_tables(aggregated_eeg_stats, aggregated_emg_stats, stage_output_dir)
    
    print(f"\nStage difference analysis completed. Results saved to: {stage_output_dir}")

def aggregate_statistics_across_recordings(all_stats):
    """
    Aggregate statistics across multiple recordings for each sleep stage.
    
    Args:
        all_stats (dict): Dictionary of statistics for each recording
    
    Returns:
        dict: Aggregated statistics for each sleep stage
    """
    if not all_stats:
        return {}
    
    # Get all possible sleep stages
    all_stages = set()
    for recording_stats in all_stats.values():
        all_stages.update(recording_stats.keys())
    
    aggregated = {}
    
    for stage in all_stages:
        stage_data = {}
        
        # Collect all values for each metric across recordings
        for metric in ['mean', 'std', 'median', 'variance', 'skewness', 'kurtosis',
                      'delta_power', 'theta_power', 'alpha_power', 'beta_power', 
                      'gamma_power', 'power_50hz', 'power_60hz', 'delta_power_rel',
                      'theta_power_rel', 'alpha_power_rel', 'beta_power_rel',
                      'gamma_power_rel', 'total_power', 'peak_frequency', 
                      'spectral_centroid', 'spectral_bandwidth']:
            values = []
            for recording_stats in all_stats.values():
                if stage in recording_stats and metric in recording_stats[stage]:
                    values.append(recording_stats[stage][metric])
            
            if values:
                stage_data[f'{metric}_mean'] = np.mean(values)
                stage_data[f'{metric}_std'] = np.std(values)
                stage_data[f'{metric}_values'] = values
                
                # Calculate confidence interval across recordings
                if len(values) > 1:
                    ci_lower, ci_upper = stats.t.interval(
                        0.95, len(values)-1, loc=np.mean(values), scale=stats.sem(values)
                    )
                    stage_data[f'{metric}_ci_lower'] = ci_lower
                    stage_data[f'{metric}_ci_upper'] = ci_upper
                else:
                    stage_data[f'{metric}_ci_lower'] = stage_data[f'{metric}_mean']
                    stage_data[f'{metric}_ci_upper'] = stage_data[f'{metric}_mean']
        
        aggregated[stage] = stage_data
    
    return aggregated

def create_stage_difference_plots(stats_data, signal_type, output_dir):
    """
    Create comprehensive plots showing signal differences between stages.
    
    Args:
        stats_data (dict): Aggregated statistics data
        signal_type (str): Type of signal (EEG or EMG)
        output_dir (Path): Output directory
    """
    if not stats_data:
        print(f"No {signal_type} data available for plotting.")
        return
    
    print(f"Creating {signal_type} stage difference plots...")
    
    # Use consistent stage ordering
    stages = [stage for stage in STAGE_ORDER if stage in stats_data]
    colors = [SLEEP_STAGE_COLORS[stage] for stage in stages]
    
    # 1. Amplitude characteristics comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{signal_type} Signal Characteristics by Sleep Stage', fontsize=16, fontweight='bold')
    
    # Mean amplitude with confidence intervals
    ax = axes[0, 0]
    means = [stats_data[stage]['mean_mean'] for stage in stages]
    ci_lower = [stats_data[stage]['mean_ci_lower'] for stage in stages]
    ci_upper = [stats_data[stage]['mean_ci_upper'] for stage in stages]
    errors = [[means[i] - ci_lower[i] for i in range(len(means))],
              [ci_upper[i] - means[i] for i in range(len(means))]]
    
    bars = ax.bar(stages, means, color=colors, alpha=0.8, edgecolor='black')
    ax.errorbar(stages, means, yerr=errors, fmt='none', color='black', capsize=5)
    ax.set_title('Mean Amplitude', fontweight='bold')
    ax.set_ylabel('Amplitude (mV)')
    ax.grid(axis='y', alpha=0.3)
    
    # Standard deviation
    ax = axes[0, 1]
    stds = [stats_data[stage]['std_mean'] for stage in stages]
    bars = ax.bar(stages, stds, color=colors, alpha=0.8, edgecolor='black')
    ax.set_title('Signal Variability (Std Dev)', fontweight='bold')
    ax.set_ylabel('Standard Deviation (mV)')
    ax.grid(axis='y', alpha=0.3)
    
    # Variance
    ax = axes[1, 0]
    variances = [stats_data[stage]['variance_mean'] for stage in stages]
    bars = ax.bar(stages, variances, color=colors, alpha=0.8, edgecolor='black')
    ax.set_title('Signal Variance', fontweight='bold')
    ax.set_ylabel('Variance (mV²)')
    ax.grid(axis='y', alpha=0.3)
    
    # Skewness
    ax = axes[1, 1]
    skewness = [stats_data[stage]['skewness_mean'] for stage in stages]
    bars = ax.bar(stages, skewness, color=colors, alpha=0.8, edgecolor='black')
    ax.set_title('Signal Skewness', fontweight='bold')
    ax.set_ylabel('Skewness')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{signal_type}_amplitude_characteristics.png', dpi=300, bbox_inches='tight')
    plt.close()
    # plt.close()
    # plt.show()  # Disabled for automated analysis  # Disabled for automated analysis

def create_frequency_domain_analysis(stats_data, output_dir):
    """
    Create comprehensive frequency domain analysis plots.
    
    Args:
        stats_data (dict): Aggregated statistics data
        output_dir (Path): Output directory
    """
    if not stats_data:
        return
    
    print("Creating comprehensive frequency domain analysis...")
    
    # Use consistent stage ordering
    stages = [stage for stage in STAGE_ORDER if stage in stats_data]
    colors = [SLEEP_STAGE_COLORS[stage] for stage in stages]
    
    # 1. Absolute Power Analysis
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('EEG Absolute Power by Sleep Stage', fontsize=16, fontweight='bold')
    
    frequency_bands = [
        ('delta_power', 'Delta (0.5-4 Hz)', axes[0, 0]),
        ('theta_power', 'Theta (4-8 Hz)', axes[0, 1]),
        ('alpha_power', 'Alpha (8-12 Hz)', axes[0, 2]),
        ('beta_power', 'Beta (12-30 Hz)', axes[1, 0]),
        ('gamma_power', 'Gamma (30-100 Hz)', axes[1, 1]),
        ('total_power', 'Total Power (0.5-100 Hz)', axes[1, 2])
    ]
    
    for band, title, ax in frequency_bands:
        if all(f'{band}_mean' in stats_data[stage] for stage in stages):
            powers = [stats_data[stage][f'{band}_mean'] for stage in stages]
            ci_lower = [stats_data[stage][f'{band}_ci_lower'] for stage in stages]
            ci_upper = [stats_data[stage][f'{band}_ci_upper'] for stage in stages]
            errors = [[powers[i] - ci_lower[i] for i in range(len(powers))],
                     [ci_upper[i] - powers[i] for i in range(len(powers))]]
            
            bars = ax.bar(stages, powers, color=colors, alpha=0.8, edgecolor='black')
            ax.errorbar(stages, powers, yerr=errors, fmt='none', color='black', capsize=5)
            ax.set_title(title, fontweight='bold')
            ax.set_ylabel('Power Spectral Density')
            ax.grid(axis='y', alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, power in zip(bars, powers):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(powers)*0.01,
                       f'{power:.2e}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'EEG_absolute_power_bands.png', dpi=300, bbox_inches='tight')
    plt.close()
    # plt.close()
    # plt.show()  # Disabled for automated analysis  # Disabled for automated analysis
    
    # 2. Relative Power Analysis
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('EEG Relative Power by Sleep Stage (%)', fontsize=16, fontweight='bold')
    
    relative_bands = [
        ('delta_power_rel', 'Delta (0.5-4 Hz)', axes[0, 0]),
        ('theta_power_rel', 'Theta (4-8 Hz)', axes[0, 1]),
        ('alpha_power_rel', 'Alpha (8-12 Hz)', axes[0, 2]),
        ('beta_power_rel', 'Beta (12-30 Hz)', axes[1, 0]),
        ('gamma_power_rel', 'Gamma (30-100 Hz)', axes[1, 1])
    ]
    
    for band, title, ax in relative_bands:
        if all(f'{band}_mean' in stats_data[stage] for stage in stages):
            powers = [stats_data[stage][f'{band}_mean'] * 100 for stage in stages]  # Convert to percentage
            bars = ax.bar(stages, powers, color=colors, alpha=0.8, edgecolor='black')
            ax.set_title(title, fontweight='bold')
            ax.set_ylabel('Relative Power (%)')
            ax.grid(axis='y', alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
    
    # Create combined relative power plot
    ax = axes[1, 2]
    create_combined_relative_power_plot(stats_data, ax)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'EEG_relative_power_bands.png', dpi=300, bbox_inches='tight')
    plt.close()
    # plt.show()  # Disabled for automated analysis
    
    # 4. Band Power Analysis
    create_band_power_analysis(stats_data, output_dir)
    
    # 6. Spectral Characteristics
    create_spectral_characteristics_plot(stats_data, output_dir)
    
    # 7. Interference Analysis
    create_interference_analysis(stats_data, output_dir)

def create_combined_relative_power_plot(stats_data, ax):
    """Create a stacked bar plot for relative power distribution."""
    stages = [stage for stage in STAGE_ORDER if stage in stats_data]
    bands = ['delta_power_rel', 'theta_power_rel', 'alpha_power_rel', 'beta_power_rel', 'gamma_power_rel']
    band_labels = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    
    # Calculate relative power for each stage
    relative_powers = {}
    for stage in stages:
        if all(f'{band}_mean' in stats_data[stage] for band in bands):
            # Get the raw relative power values (should already be proportions)
            raw_powers = [stats_data[stage][f'{band}_mean'] for band in bands]
            # Normalize to ensure they sum to 1 (100%)
            total_power = sum(raw_powers)
            if total_power > 0:
                relative_powers[stage] = [(p/total_power) * 100 for p in raw_powers]
            else:
                relative_powers[stage] = [0] * len(bands)
        else:
            relative_powers[stage] = [0] * len(bands)
    
    # Create stacked bar plot
    bottom = np.zeros(len(stages))
    colors = plt.cm.Set3(np.linspace(0, 1, len(bands)))
    
    for i, (band, label) in enumerate(zip(bands, band_labels)):
        values = [relative_powers[stage][i] for stage in stages]
        ax.bar(stages, values, bottom=bottom, label=label, color=colors[i], alpha=0.8)
        bottom += values
    
    ax.set_title('Combined Relative Power Distribution', fontweight='bold')
    ax.set_ylabel('Relative Power (%)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

def create_band_power_analysis(stats_data, output_dir):
    """
    Create comprehensive band power analysis focusing on frequency band characteristics.
    
    Args:
        stats_data (dict): Aggregated statistics data  
        output_dir (Path): Output directory
    """
    if not stats_data:
        return
    
    print("Creating comprehensive band power analysis...")
    
    # Use consistent stage ordering
    stages = [stage for stage in STAGE_ORDER if stage in stats_data]
    colors = [SLEEP_STAGE_COLORS[stage] for stage in stages]
    
    # Define frequency bands for analysis
    bands = {
        'Delta (0.5-4 Hz)': 'delta_power',
        'Theta (4-8 Hz)': 'theta_power', 
        'Alpha (8-12 Hz)': 'alpha_power',
        'Beta (12-30 Hz)': 'beta_power',
        'Gamma (30-100 Hz)': 'gamma_power'
    }
    
    # 1. Band Power Distribution Overview
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('EEG Band Power Analysis by Sleep Stage', fontsize=16, fontweight='bold')
    
    # Individual band power plots
    for idx, (band_name, band_key) in enumerate(bands.items()):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        if all(f'{band_key}_mean' in stats_data[stage] for stage in stages):
            powers = [stats_data[stage][f'{band_key}_mean'] for stage in stages]
            ci_lower = [stats_data[stage][f'{band_key}_ci_lower'] for stage in stages]
            ci_upper = [stats_data[stage][f'{band_key}_ci_upper'] for stage in stages]
            errors = [[powers[i] - ci_lower[i] for i in range(len(powers))],
                     [ci_upper[i] - powers[i] for i in range(len(powers))]]
            
            bars = ax.bar(stages, powers, color=colors, alpha=0.8, edgecolor='black')
            ax.errorbar(stages, powers, yerr=errors, fmt='none', color='black', capsize=5)
            ax.set_title(f'{band_name} Power', fontweight='bold')
            ax.set_ylabel('Band Power (µV²/Hz)')
            ax.grid(axis='y', alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            
            # Set y-axis to log scale for better visualization
            ax.set_yscale('log')
    
    # Power ratios analysis
    ax = axes[1, 2]
    create_band_power_ratios_plot(stats_data, ax, stages, colors)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'EEG_band_power_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    # plt.show()  # Disabled for automated analysis
    
    # 2. Band Power Comparisons
    create_band_power_comparisons(stats_data, output_dir, stages, colors)
    
    # 3. Band Power Statistical Analysis
    create_band_power_statistics(stats_data, output_dir, stages)

def create_band_power_ratios_plot(stats_data, ax, stages, colors):
    """Create analysis of band power ratios."""
    # Calculate meaningful ratios
    ratios = {}
    ratio_names = []
    
    for stage in stages:
        if all(f'{band}_mean' in stats_data[stage] for band in ['theta_power', 'delta_power', 'alpha_power', 'beta_power']):
            # Common sleep research ratios
            theta = stats_data[stage]['theta_power_mean']
            delta = stats_data[stage]['delta_power_mean']
            alpha = stats_data[stage]['alpha_power_mean'] 
            beta = stats_data[stage]['beta_power_mean']
            
            if stage not in ratios:
                ratios[stage] = []
            
            # Theta/Delta ratio (important for sleep staging)
            ratios[stage].append(theta / delta if delta > 0 else 0)
            
            # Alpha/Theta ratio (arousal indicator)
            ratios[stage].append(alpha / theta if theta > 0 else 0)
            
            # Beta/Alpha ratio (activation indicator)
            ratios[stage].append(beta / alpha if alpha > 0 else 0)
    
    ratio_names = ['Theta/Delta', 'Alpha/Theta', 'Beta/Alpha']
    
    # Create grouped bar plot
    x = np.arange(len(ratio_names))
    width = 0.2
    
    for i, stage in enumerate(stages):
        if stage in ratios:
            ax.bar(x + i * width, ratios[stage], width, label=stage, 
                  color=colors[i], alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Power Ratios')
    ax.set_ylabel('Ratio Value')
    ax.set_title('Band Power Ratios', fontweight='bold')
    ax.set_xticks(x + width * (len(stages) - 1) / 2)
    ax.set_xticklabels(ratio_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_yscale('log')

def create_band_power_comparisons(stats_data, output_dir, stages, colors):
    """Create detailed band power comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Band Power Detailed Comparisons', fontsize=16, fontweight='bold')
    
    # 1. Low frequency bands (Delta vs Theta)
    ax = axes[0, 0]
    create_band_comparison_scatter(stats_data, ax, stages, colors, 
                                  'delta_power', 'theta_power', 
                                  'Delta Power', 'Theta Power')
    
    # 2. Mid frequency bands (Alpha vs Beta)
    ax = axes[0, 1] 
    create_band_comparison_scatter(stats_data, ax, stages, colors,
                                  'alpha_power', 'beta_power',
                                  'Alpha Power', 'Beta Power')
    
    # 3. Low vs High frequency (Delta vs Gamma)
    ax = axes[1, 0]
    create_band_comparison_scatter(stats_data, ax, stages, colors,
                                  'delta_power', 'gamma_power', 
                                  'Delta Power', 'Gamma Power')
    
    # 4. Band power distribution
    ax = axes[1, 1]
    create_band_power_distribution(stats_data, ax, stages, colors)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'EEG_band_power_comparisons.png', dpi=300, bbox_inches='tight')
    plt.close()
    # plt.show()  # Disabled for automated analysis

def create_band_comparison_scatter(stats_data, ax, stages, colors, band1, band2, label1, label2):
    """Create scatter plot comparing two frequency bands."""
    for i, stage in enumerate(stages):
        if f'{band1}_mean' in stats_data[stage] and f'{band2}_mean' in stats_data[stage]:
            x_power = stats_data[stage][f'{band1}_mean']
            y_power = stats_data[stage][f'{band2}_mean'] 
            
            # Use error bars from confidence intervals
            x_err = stats_data[stage][f'{band1}_ci_upper'] - x_power
            y_err = stats_data[stage][f'{band2}_ci_upper'] - y_power
            
            ax.errorbar(x_power, y_power, xerr=x_err, yerr=y_err,
                       fmt='o', color=colors[i], label=stage, markersize=8,
                       alpha=0.8, capsize=5)
    
    ax.set_xlabel(f'{label1} (µV²/Hz)')
    ax.set_ylabel(f'{label2} (µV²/Hz)')
    ax.set_title(f'{label1} vs {label2}', fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()

def create_band_power_distribution(stats_data, ax, stages, colors):
    """Create normalized band power distribution plot."""
    bands = ['delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power']
    band_labels = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    
    # Calculate total power for each stage
    stage_totals = {}
    for stage in stages:
        if all(f'{band}_mean' in stats_data[stage] for band in bands):
            total = sum(stats_data[stage][f'{band}_mean'] for band in bands)
            stage_totals[stage] = total
    
    # Create normalized distribution
    x = np.arange(len(stages))
    width = 0.15
    
    for i, (band, label) in enumerate(zip(bands, band_labels)):
        values = []
        for stage in stages:
            if stage in stage_totals and stage_totals[stage] > 0:
                normalized_power = stats_data[stage][f'{band}_mean'] / stage_totals[stage] * 100
                values.append(normalized_power)
            else:
                values.append(0)
        
        ax.bar(x + i * width, values, width, label=label, alpha=0.8)
    
    ax.set_xlabel('Sleep Stage')
    ax.set_ylabel('Normalized Power Distribution (%)')
    ax.set_title('Band Power Distribution by Stage', fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(stages)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

def create_band_power_statistics(stats_data, output_dir, stages):
    """Create comprehensive band power statistics table."""
    print("Creating band power statistics...")
    
    bands = {
        'Delta (0.5-4 Hz)': 'delta_power',
        'Theta (4-8 Hz)': 'theta_power',
        'Alpha (8-12 Hz)': 'alpha_power', 
        'Beta (12-30 Hz)': 'beta_power',
        'Gamma (30-100 Hz)': 'gamma_power'
    }
    
    # Create comprehensive statistics table
    stats_table = []
    
    for band_name, band_key in bands.items():
        for stage in stages:
            if f'{band_key}_mean' in stats_data[stage]:
                stats_table.append({
                    'Frequency_Band': band_name,
                    'Sleep_Stage': stage,
                    'Mean_Power': f"{stats_data[stage][f'{band_key}_mean']:.2e}",
                    'Std_Power': f"{stats_data[stage][f'{band_key}_std']:.2e}",
                    'CI_Lower': f"{stats_data[stage][f'{band_key}_ci_lower']:.2e}",
                    'CI_Upper': f"{stats_data[stage][f'{band_key}_ci_upper']:.2e}",
                    'Relative_Power': f"{stats_data[stage][f'{band_key}_rel_mean']*100:.2f}%"
                })
    
    # Save to CSV
    stats_df = pd.DataFrame(stats_table)
    stats_df.to_csv(output_dir / 'EEG_band_power_statistics.csv', index=False)
    print(f"Band power statistics saved to: {output_dir / 'EEG_band_power_statistics.csv'}")
    
    # Create summary statistics
    create_band_power_summary(stats_data, output_dir, stages, bands)

def create_band_power_summary(stats_data, output_dir, stages, bands):
    """Create summary of key band power findings."""
    summary = []
    
    # Overall band power ranking by stage
    for stage in stages:
        stage_powers = {}
        for band_name, band_key in bands.items():
            if f'{band_key}_mean' in stats_data[stage]:
                stage_powers[band_name] = stats_data[stage][f'{band_key}_mean']
        
        # Find dominant band
        if stage_powers:
            dominant_band = max(stage_powers, key=stage_powers.get)
            dominant_power = stage_powers[dominant_band]
            
            summary.append({
                'Sleep_Stage': stage,
                'Dominant_Band': dominant_band,
                'Dominant_Power': f"{dominant_power:.2e}",
                'Total_Power': f"{sum(stage_powers.values()):.2e}",
                'Delta_Dominance': f"{stage_powers.get('Delta (0.5-4 Hz)', 0) / sum(stage_powers.values()) * 100:.1f}%",
                'Theta_Dominance': f"{stage_powers.get('Theta (4-8 Hz)', 0) / sum(stage_powers.values()) * 100:.1f}%",
                'Alpha_Dominance': f"{stage_powers.get('Alpha (8-12 Hz)', 0) / sum(stage_powers.values()) * 100:.1f}%",
                'Beta_Dominance': f"{stage_powers.get('Beta (12-30 Hz)', 0) / sum(stage_powers.values()) * 100:.1f}%",
                'Gamma_Dominance': f"{stage_powers.get('Gamma (30-100 Hz)', 0) / sum(stage_powers.values()) * 100:.1f}%"
            })
    
    # Save summary
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(output_dir / 'EEG_band_power_summary.csv', index=False)
    print(f"Band power summary saved to: {output_dir / 'EEG_band_power_summary.csv'}")

def create_fine_grained_band_analysis(all_stats, output_dir):
    """
    Create fine-grained frequency band analysis with 1 Hz wide bands from 0-50 Hz.
    
    Args:
        all_stats (dict): Raw statistics from all recordings containing PSD data
        output_dir (Path): Output directory
    """
    print("Creating fine-grained frequency band analysis (1 Hz bands)...")
    
    # Collect PSD data for each stage
    stage_psds = {}
    stage_freqs = {}
    
    for recording_id, recording_stats in all_stats.items():
        for stage, stage_stats in recording_stats.items():
            if 'psd' in stage_stats and 'frequencies' in stage_stats:
                if stage not in stage_psds:
                    stage_psds[stage] = []
                    stage_freqs[stage] = stage_stats['frequencies']
                stage_psds[stage].append(stage_stats['psd'])
    
    if not stage_psds:
        print("No PSD data available for fine-grained analysis.")
        return
    
    # Use consistent stage ordering
    stages = [stage for stage in STAGE_ORDER if stage in stage_psds]
    colors = [SLEEP_STAGE_COLORS[stage] for stage in stages]
    
    # Define 1 Hz frequency bands from 0-50 Hz
    freq_bands = [(i, i+1) for i in range(50)]  # [0-1], [1-2], ..., [49-50]
    band_labels = [f"{i}-{i+1} Hz" for i in range(50)]
    
    # Calculate band power for each stage
    stage_band_powers = {}
    stage_band_errors = {}
    
    for stage in stages:
        if stage in stage_psds:
            # Calculate mean PSD and confidence intervals
            psds = np.array(stage_psds[stage])
            freqs = stage_freqs[stage]
            
            # Calculate band powers for each frequency band
            band_powers = []
            band_lower_ci = []
            band_upper_ci = []
            
            for freq_min, freq_max in freq_bands:
                # Find frequency indices for this band
                freq_mask = (freqs >= freq_min) & (freqs < freq_max)
                
                if np.any(freq_mask):
                    # Calculate band power for each recording
                    band_power_values = []
                    for psd in psds:
                        band_power = np.mean(psd[freq_mask])
                        band_power_values.append(band_power)
                    
                    # Calculate statistics
                    mean_power = np.mean(band_power_values)
                    std_power = np.std(band_power_values, ddof=1) if len(band_power_values) > 1 else 0
                    
                    # 95% confidence interval
                    if len(band_power_values) > 1:
                        t_stat = stats.t.ppf(0.975, len(band_power_values) - 1)
                        margin_error = t_stat * std_power / np.sqrt(len(band_power_values))
                        lower_ci = mean_power - margin_error
                        upper_ci = mean_power + margin_error
                    else:
                        lower_ci = upper_ci = mean_power
                    
                    band_powers.append(mean_power)
                    band_lower_ci.append(lower_ci)
                    band_upper_ci.append(upper_ci)
                else:
                    # No data in this frequency range
                    band_powers.append(0)
                    band_lower_ci.append(0)
                    band_upper_ci.append(0)
            
            stage_band_powers[stage] = band_powers
            stage_band_errors[stage] = {
                'lower': band_lower_ci,
                'upper': band_upper_ci
            }
    
    # Create the comprehensive plot
    fig, ax = plt.subplots(1, 1, figsize=(20, 8))
    fig.suptitle('Fine-Grained Frequency Band Power Analysis (1 Hz bands)', fontsize=16, fontweight='bold')
    
    # Set up the bar plot
    x = np.arange(len(freq_bands))
    width = 0.2
    
    for i, stage in enumerate(stages):
        if stage in stage_band_powers:
            powers = stage_band_powers[stage]
            lower_ci = stage_band_errors[stage]['lower']
            upper_ci = stage_band_errors[stage]['upper']
            
            # Calculate error bars
            errors = [[powers[j] - lower_ci[j] for j in range(len(powers))],
                     [upper_ci[j] - powers[j] for j in range(len(powers))]]
            
            # Create bars for this stage
            bars = ax.bar(x + i * width, powers, width, label=stage, 
                         color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Add error bars
            ax.errorbar(x + i * width, powers, yerr=errors, fmt='none', 
                       color='black', capsize=2, linewidth=0.8, alpha=0.7)
    
    # Customize the plot
    ax.set_xlabel('Frequency Band (Hz)', fontweight='bold')
    ax.set_ylabel('Band Power (µV²/Hz)', fontweight='bold')
    ax.set_title('Band Power Distribution Across 1 Hz Frequency Bands (0-50 Hz)', fontweight='bold')
    ax.set_yscale('log')  # Log scale for better visualization
    ax.grid(axis='y', alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Set x-axis ticks and labels
    tick_indices = range(0, 50, 5)  # Show every 5th frequency band
    tick_labels = [f"{i}-{i+1}" for i in tick_indices]
    ax.set_xticks([i + width * (len(stages) - 1) / 2 for i in tick_indices])
    ax.set_xticklabels(tick_labels, rotation=45)
    
    # Add frequency range annotation
    ax.text(0.02, 0.98, 'Frequency Range: 0-50 Hz (1 Hz bands)', 
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'EEG_fine_grained_band_power_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    # plt.show()  # Disabled for automated analysis
    
    # Save detailed results to CSV
    create_fine_grained_band_csv(stage_band_powers, stage_band_errors, freq_bands, stages, output_dir)
    
    print(f"Fine-grained band analysis completed. Results saved to: {output_dir}")

def create_fine_grained_band_csv(stage_band_powers, stage_band_errors, freq_bands, stages, output_dir):
    """Save fine-grained band power results to CSV."""
    print("Saving fine-grained band power results to CSV...")
    
    # Create comprehensive results table
    results = []
    
    for i, (freq_min, freq_max) in enumerate(freq_bands):
        band_name = f"{freq_min}-{freq_max} Hz"
        
        for stage in stages:
            if stage in stage_band_powers:
                results.append({
                    'Frequency_Band': band_name,
                    'Freq_Min_Hz': freq_min,
                    'Freq_Max_Hz': freq_max,
                    'Sleep_Stage': stage,
                    'Mean_Power': f"{stage_band_powers[stage][i]:.6e}",
                    'CI_Lower': f"{stage_band_errors[stage]['lower'][i]:.6e}",
                    'CI_Upper': f"{stage_band_errors[stage]['upper'][i]:.6e}",
                    'Power_uV2_Hz': stage_band_powers[stage][i]
                })
    
    # Save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'EEG_fine_grained_band_power_1Hz.csv', index=False)
    print(f"Fine-grained band power results saved to: {output_dir / 'EEG_fine_grained_band_power_1Hz.csv'}")
    
    # Create summary statistics
    summary_stats = []
    for stage in stages:
        if stage in stage_band_powers:
            powers = stage_band_powers[stage]
            # Find peak frequency band
            max_power_idx = np.argmax(powers)
            peak_freq_band = f"{max_power_idx}-{max_power_idx+1} Hz"
            max_power = powers[max_power_idx]
            
            # Calculate total power across all bands
            total_power = np.sum(powers)
            
            # Find frequency bands with significant power (top 10%)
            power_threshold = np.percentile(powers, 90)
            significant_bands = [i for i, p in enumerate(powers) if p >= power_threshold]
            
            summary_stats.append({
                'Sleep_Stage': stage,
                'Peak_Frequency_Band': peak_freq_band,
                'Peak_Power': f"{max_power:.6e}",
                'Total_Power_0_50Hz': f"{total_power:.6e}",
                'Significant_Bands_Count': len(significant_bands),
                'Significant_Bands': ', '.join([f"{i}-{i+1} Hz" for i in significant_bands[:5]])  # Show top 5
            })
    
    # Save summary
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(output_dir / 'EEG_fine_grained_band_power_summary.csv', index=False)
    print(f"Fine-grained band power summary saved to: {output_dir / 'EEG_fine_grained_band_power_summary.csv'}")

def create_spectral_characteristics_plot(stats_data, output_dir):
    """Create plots for spectral characteristics like peak frequency and centroid."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('EEG Spectral Characteristics by Sleep Stage', fontsize=16, fontweight='bold')
    
    stages = [stage for stage in STAGE_ORDER if stage in stats_data]
    colors = [SLEEP_STAGE_COLORS[stage] for stage in stages]
    
    # Peak frequency
    ax = axes[0]
    if all('peak_frequency_mean' in stats_data[stage] for stage in stages):
        peak_freqs = [stats_data[stage]['peak_frequency_mean'] for stage in stages]
        bars = ax.bar(stages, peak_freqs, color=colors, alpha=0.8, edgecolor='black')
        ax.set_title('Peak Frequency', fontweight='bold')
        ax.set_ylabel('Frequency (Hz)')
        ax.grid(axis='y', alpha=0.3)
        
        for bar, freq in zip(bars, peak_freqs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(peak_freqs)*0.01,
                   f'{freq:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Spectral centroid
    ax = axes[1]
    if all('spectral_centroid_mean' in stats_data[stage] for stage in stages):
        centroids = [stats_data[stage]['spectral_centroid_mean'] for stage in stages]
        bars = ax.bar(stages, centroids, color=colors, alpha=0.8, edgecolor='black')
        ax.set_title('Spectral Centroid', fontweight='bold')
        ax.set_ylabel('Frequency (Hz)')
        ax.grid(axis='y', alpha=0.3)
        
        for bar, centroid in zip(bars, centroids):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(centroids)*0.01,
                   f'{centroid:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Spectral bandwidth
    ax = axes[2]
    if all('spectral_bandwidth_mean' in stats_data[stage] for stage in stages):
        bandwidths = [stats_data[stage]['spectral_bandwidth_mean'] for stage in stages]
        bars = ax.bar(stages, bandwidths, color=colors, alpha=0.8, edgecolor='black')
        ax.set_title('Spectral Bandwidth', fontweight='bold')
        ax.set_ylabel('Bandwidth (Hz)')
        ax.grid(axis='y', alpha=0.3)
        
        for bar, bandwidth in zip(bars, bandwidths):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(bandwidths)*0.01,
                   f'{bandwidth:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'EEG_spectral_characteristics.png', dpi=300, bbox_inches='tight')
    plt.close()
    # plt.show()  # Disabled for automated analysis

def create_interference_analysis(stats_data, output_dir):
    """Create analysis of power line interference at 50Hz and 60Hz."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Power Line Interference Analysis by Sleep Stage', fontsize=16, fontweight='bold')
    
    stages = [stage for stage in STAGE_ORDER if stage in stats_data]
    colors = [SLEEP_STAGE_COLORS[stage] for stage in stages]
    
    # 50Hz interference
    ax = axes[0]
    if all('power_50hz_mean' in stats_data[stage] for stage in stages):
        power_50hz = [stats_data[stage]['power_50hz_mean'] for stage in stages]
        bars = ax.bar(stages, power_50hz, color=colors, alpha=0.8, edgecolor='black')
        ax.set_title('50Hz Interference', fontweight='bold')
        ax.set_ylabel('Power Spectral Density')
        ax.grid(axis='y', alpha=0.3)
        
        for bar, power in zip(bars, power_50hz):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(power_50hz)*0.01,
                   f'{power:.2e}', ha='center', va='bottom', fontsize=8)
    
    # 60Hz interference
    ax = axes[1]
    if all('power_60hz_mean' in stats_data[stage] for stage in stages):
        power_60hz = [stats_data[stage]['power_60hz_mean'] for stage in stages]
        bars = ax.bar(stages, power_60hz, color=colors, alpha=0.8, edgecolor='black')
        ax.set_title('60Hz Interference', fontweight='bold')
        ax.set_ylabel('Power Spectral Density')
        ax.grid(axis='y', alpha=0.3)
        
        for bar, power in zip(bars, power_60hz):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(power_60hz)*0.01,
                   f'{power:.2e}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'EEG_interference_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    # plt.show()  # Disabled for automated analysis

def create_detailed_psd_analysis(all_stats, output_dir):
    """
    Create detailed power spectral density analysis across stages with confidence intervals.
    
    Args:
        all_stats (dict): Raw statistics from all recordings
        output_dir (Path): Output directory
    """
    print("Creating detailed power spectral density analysis with confidence intervals...")
    
    # Collect PSD data for each stage
    stage_psds = {}
    stage_freqs = {}
    
    for recording_id, recording_stats in all_stats.items():
        for stage, stage_stats in recording_stats.items():
            if 'psd' in stage_stats and 'frequencies' in stage_stats:
                if stage not in stage_psds:
                    stage_psds[stage] = []
                    stage_freqs[stage] = stage_stats['frequencies']
                stage_psds[stage].append(stage_stats['psd'])
    
    if not stage_psds:
        print("No PSD data available for detailed analysis.")
        return
    
    # Create detailed PSD comparison plot with confidence intervals
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Detailed Power Spectral Density by Sleep Stage (with 95% CI)', fontsize=16, fontweight='bold')
    
    # Use consistent ordering of stages
    stages = [stage for stage in STAGE_ORDER if stage in stage_psds]
    
    # Plot 1: Average PSD across all stages (log scale) with confidence intervals
    ax = axes[0, 0]
    for stage in stages:
        if stage_psds[stage]:
            mean_psd, ci_lower, ci_upper = calculate_spectral_confidence_intervals(stage_psds[stage])
            freqs = stage_freqs[stage]
            # Limit to meaningful frequency range
            freq_mask = (freqs >= 0.5) & (freqs <= 45)
            
            color = SLEEP_STAGE_COLORS[stage]
            ax.semilogy(freqs[freq_mask], mean_psd[freq_mask], 
                       label=stage, color=color, linewidth=2)
            
            # Add confidence interval
            if ci_lower is not None and ci_upper is not None:
                ax.fill_between(freqs[freq_mask], ci_lower[freq_mask], ci_upper[freq_mask],
                               color=color, alpha=0.2)
    
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density (log scale)')
    ax.set_title('Average PSD with 95% CI (0.5-45 Hz)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Low frequency detail (0.5-10 Hz) with confidence intervals
    ax = axes[0, 1]
    for stage in stages:
        if stage_psds[stage]:
            mean_psd, ci_lower, ci_upper = calculate_spectral_confidence_intervals(stage_psds[stage])
            freqs = stage_freqs[stage]
            freq_mask = (freqs >= 0.5) & (freqs <= 10)
            
            color = SLEEP_STAGE_COLORS[stage]
            ax.plot(freqs[freq_mask], mean_psd[freq_mask], 
                   label=stage, color=color, linewidth=2)
            
            # Add confidence interval
            if ci_lower is not None and ci_upper is not None:
                ax.fill_between(freqs[freq_mask], ci_lower[freq_mask], ci_upper[freq_mask],
                               color=color, alpha=0.2)
    
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density')
    ax.set_title('Low Frequency Detail with 95% CI (0.5-10 Hz)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: High frequency detail (10-45 Hz) with confidence intervals
    ax = axes[1, 0]
    for stage in stages:
        if stage_psds[stage]:
            mean_psd, ci_lower, ci_upper = calculate_spectral_confidence_intervals(stage_psds[stage])
            freqs = stage_freqs[stage]
            freq_mask = (freqs >= 10) & (freqs <= 45)
            
            color = SLEEP_STAGE_COLORS[stage]
            ax.plot(freqs[freq_mask], mean_psd[freq_mask], 
                   label=stage, color=color, linewidth=2)
            
            # Add confidence interval
            if ci_lower is not None and ci_upper is not None:
                ax.fill_between(freqs[freq_mask], ci_lower[freq_mask], ci_upper[freq_mask],
                               color=color, alpha=0.2)
    
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density')
    ax.set_title('High Frequency Detail with 95% CI (10-45 Hz)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Normalized PSD (relative to total power) with confidence intervals
    ax = axes[1, 1]
    for stage in stages:
        if stage_psds[stage]:
            # Normalize each PSD individually before calculating confidence intervals
            normalized_psds = []
            for psd in stage_psds[stage]:
                freqs = stage_freqs[stage]
                freq_mask = (freqs >= 0.5) & (freqs <= 45)
                total_power = np.sum(psd[freq_mask])
                if total_power > 0:
                    normalized_psds.append(psd[freq_mask] / total_power)
            
            if normalized_psds:
                mean_psd, ci_lower, ci_upper = calculate_spectral_confidence_intervals(normalized_psds)
                freqs_masked = freqs[freq_mask]
                
                color = SLEEP_STAGE_COLORS[stage]
                ax.plot(freqs_masked, mean_psd, label=stage, color=color, linewidth=2)
                
                # Add confidence interval
                if ci_lower is not None and ci_upper is not None:
                    ax.fill_between(freqs_masked, ci_lower, ci_upper, color=color, alpha=0.2)
    
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Normalized Power Spectral Density')
    ax.set_title('Normalized PSD with 95% CI (0.5-45 Hz)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'EEG_detailed_psd_analysis_with_CI.png', dpi=300, bbox_inches='tight')
    plt.close()
    # plt.show()  # Disabled for automated analysis

def create_comprehensive_summary_tables(eeg_stats, emg_stats, output_dir):
    """
    Create comprehensive summary tables for stage differences.
    
    Args:
        eeg_stats (dict): EEG statistics
        emg_stats (dict): EMG statistics  
        output_dir (Path): Output directory
    """
    print("Creating comprehensive summary tables...")
    
    # EEG summary table
    if eeg_stats:
        eeg_summary = {}
        for stage in [s for s in STAGE_ORDER if s in eeg_stats]:
            eeg_summary[stage] = {
                'Mean Amplitude': f"{eeg_stats[stage]['mean_mean']:.6f} ± {eeg_stats[stage]['mean_std']:.6f}",
                'Std Deviation': f"{eeg_stats[stage]['std_mean']:.6f}",
                'Variance': f"{eeg_stats[stage]['variance_mean']:.8f}",
                'Skewness': f"{eeg_stats[stage]['skewness_mean']:.4f}",
                'Delta Power (abs)': f"{eeg_stats[stage]['delta_power_mean']:.2e}",
                'Theta Power (abs)': f"{eeg_stats[stage]['theta_power_mean']:.2e}",
                'Alpha Power (abs)': f"{eeg_stats[stage]['alpha_power_mean']:.2e}",
                'Beta Power (abs)': f"{eeg_stats[stage]['beta_power_mean']:.2e}",
                'Gamma Power (abs)': f"{eeg_stats[stage]['gamma_power_mean']:.2e}",
                'Delta Power (%)': f"{eeg_stats[stage]['delta_power_rel_mean']*100:.1f}%",
                'Theta Power (%)': f"{eeg_stats[stage]['theta_power_rel_mean']*100:.1f}%",
                'Alpha Power (%)': f"{eeg_stats[stage]['alpha_power_rel_mean']*100:.1f}%",
                'Beta Power (%)': f"{eeg_stats[stage]['beta_power_rel_mean']*100:.1f}%",
                'Gamma Power (%)': f"{eeg_stats[stage]['gamma_power_rel_mean']*100:.1f}%",
                'Total Power': f"{eeg_stats[stage]['total_power_mean']:.2e}",
                'Peak Frequency': f"{eeg_stats[stage]['peak_frequency_mean']:.2f} Hz",
                'Spectral Centroid': f"{eeg_stats[stage]['spectral_centroid_mean']:.2f} Hz",
                'Spectral Bandwidth': f"{eeg_stats[stage]['spectral_bandwidth_mean']:.2f} Hz",
                '50Hz Interference': f"{eeg_stats[stage]['power_50hz_mean']:.2e}",
                '60Hz Interference': f"{eeg_stats[stage]['power_60hz_mean']:.2e}"
            }
        
        eeg_df = pd.DataFrame(eeg_summary).T
        eeg_df.to_csv(output_dir / 'EEG_stage_differences_comprehensive.csv')
        print(f"Comprehensive EEG summary saved to: {output_dir / 'EEG_stage_differences_comprehensive.csv'}")
    
    # EMG summary table
    if emg_stats:
        emg_summary = {}
        for stage in [s for s in STAGE_ORDER if s in emg_stats]:
            emg_summary[stage] = {
                'Mean Amplitude': f"{emg_stats[stage]['mean_mean']:.6f} ± {emg_stats[stage]['mean_std']:.6f}",
                'Std Deviation': f"{emg_stats[stage]['std_mean']:.6f}",
                'Variance': f"{emg_stats[stage]['variance_mean']:.8f}",
                'Skewness': f"{emg_stats[stage]['skewness_mean']:.4f}"
            }
        
        emg_df = pd.DataFrame(emg_summary).T
        emg_df.to_csv(output_dir / 'EMG_stage_differences_summary.csv')
        print(f"EMG summary saved to: {output_dir / 'EMG_stage_differences_summary.csv'}")

def create_transition_analysis(selected_participants, metadata, output_dir):
    """
    Create comprehensive transition matrix analysis.
    
    Args:
        selected_participants (list): List of (participant_id, lab) tuples
        metadata (pd.DataFrame): Metadata dataframe
        output_dir (Path): Output directory
    """
    print("Creating sleep stage transition analysis...")
    
    all_transitions = []
    all_labels = []
    
    # Collect transition data from all participants
    for participant_id, lab in selected_participants:
        print(f"  Analyzing transitions for {participant_id}...")
        
        participant_runs = metadata[metadata['participant_id'] == participant_id]
        
        for _, run_info in participant_runs.iterrows():
            run_id = run_info['run']
            
            try:
                # Load participant data
                data = load_participant_data(participant_id, run_id)
                labels = data['labels']
                
                # Downsample labels to 4-second intervals for meaningful transition analysis
                downsampled_labels = downsample_labels(labels, sample_rate=128, target_interval_seconds=4)
                
                # Calculate transitions for this recording
                transition_data = calculate_stage_transitions(downsampled_labels)
                all_transitions.append(transition_data)
                all_labels.extend(downsampled_labels)
                
            except Exception as e:
                print(f"    Error processing {participant_id} run {run_id}: {e}")
                continue
    
    if not all_transitions:
        print("No transition data available.")
        return
    
    # Aggregate transition matrices across all recordings
    aggregated_transitions = aggregate_transition_matrices(all_transitions)
    
    # Create transition matrix visualizations
    create_transition_matrix_plots(aggregated_transitions, output_dir)
    
    # Create transition statistics
    create_transition_statistics(aggregated_transitions, output_dir)
    
    print(f"Transition analysis completed. Results saved to: {output_dir}")

def aggregate_transition_matrices(all_transitions):
    """
    Aggregate transition matrices from multiple recordings.
    
    Args:
        all_transitions (list): List of transition data dictionaries
    
    Returns:
        dict: Aggregated transition data
    """
    if not all_transitions:
        return {}
    
    # Get all unique stages across all recordings
    all_unique_stages = set()
    for transition_data in all_transitions:
        all_unique_stages.update(transition_data['unique_stages'])
    
    # Sort stages for consistent ordering
    sorted_stages = sorted(list(all_unique_stages))
    stage_names = [SLEEP_STAGE_MAPPING[stage] for stage in sorted_stages]
    n_stages = len(sorted_stages)
    
    # Initialize aggregated matrices
    total_transition_counts = np.zeros((n_stages, n_stages))
    total_change_counts = np.zeros((n_stages, n_stages))
    
    # Sum up all transition counts, handling different stage sets
    for transition_data in all_transitions:
        recording_stages = transition_data['unique_stages']
        recording_transition_counts = transition_data['transition_counts']
        recording_change_counts = transition_data['change_only_counts']
        
        # Map recording stages to global stage indices
        for i, from_stage in enumerate(recording_stages):
            global_i = sorted_stages.index(from_stage)
            for j, to_stage in enumerate(recording_stages):
                global_j = sorted_stages.index(to_stage)
                total_transition_counts[global_i, global_j] += recording_transition_counts[i, j]
                total_change_counts[global_i, global_j] += recording_change_counts[i, j]
    
    # Calculate aggregated probabilities
    # All transitions (including self-transitions)
    row_sums = np.sum(total_transition_counts, axis=1)
    aggregated_transition_probs = np.zeros_like(total_transition_counts)
    for i in range(n_stages):
        if row_sums[i] > 0:
            aggregated_transition_probs[i, :] = total_transition_counts[i, :] / row_sums[i]
    
    # Change-only transitions (excluding self-transitions)
    change_row_sums = np.sum(total_change_counts, axis=1)
    aggregated_change_probs = np.zeros_like(total_change_counts)
    for i in range(n_stages):
        if change_row_sums[i] > 0:
            aggregated_change_probs[i, :] = total_change_counts[i, :] / change_row_sums[i]
    
    return {
        'stage_names': stage_names,
        'transition_counts': total_transition_counts,
        'transition_probs': aggregated_transition_probs,
        'change_only_counts': total_change_counts,
        'change_only_probs': aggregated_change_probs,
        'total_transitions': np.sum(total_transition_counts),
        'total_changes': np.sum(total_change_counts),
        'n_recordings': len(all_transitions)
    }

def create_transition_matrix_plots(transition_data, output_dir):
    """
    Create visualizations for transition matrices.
    
    Args:
        transition_data (dict): Aggregated transition data
        output_dir (Path): Output directory
    """
    if not transition_data:
        return
    
    stage_names = transition_data['stage_names']
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Sleep Stage Transition Matrices (4-second intervals)', fontsize=16, fontweight='bold')
    
    # Plot 1: All transitions (including self-transitions)
    ax = axes[0]
    transition_probs = transition_data['transition_probs']
    
    im1 = ax.imshow(transition_probs, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(stage_names)))
    ax.set_yticks(range(len(stage_names)))
    ax.set_xticklabels(stage_names, rotation=45)
    ax.set_yticklabels(stage_names)
    ax.set_xlabel('To Stage')
    ax.set_ylabel('From Stage')
    ax.set_title('All Transitions (4s intervals)\n(Including Self-Transitions)', fontweight='bold')
    
    # Add text annotations
    for i in range(len(stage_names)):
        for j in range(len(stage_names)):
            text = ax.text(j, i, f'{transition_probs[i, j]:.3f}',
                          ha="center", va="center", color="white" if transition_probs[i, j] > 0.5 else "black",
                          fontweight='bold')
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax, shrink=0.8)
    cbar1.set_label('Transition Probability')
    
    # Plot 2: Change-only transitions (excluding self-transitions)
    ax = axes[1]
    change_probs = transition_data['change_only_probs']
    
    im2 = ax.imshow(change_probs, cmap='Reds', vmin=0, vmax=1)
    ax.set_xticks(range(len(stage_names)))
    ax.set_yticks(range(len(stage_names)))
    ax.set_xticklabels(stage_names, rotation=45)
    ax.set_yticklabels(stage_names)
    ax.set_xlabel('To Stage')
    ax.set_ylabel('From Stage')
    ax.set_title('Change-Only Transitions (4s intervals)\n(Excluding Self-Transitions)', fontweight='bold')
    
    # Add text annotations
    for i in range(len(stage_names)):
        for j in range(len(stage_names)):
            if i != j:  # Only show non-diagonal elements
                text = ax.text(j, i, f'{change_probs[i, j]:.3f}',
                              ha="center", va="center", color="white" if change_probs[i, j] > 0.5 else "black",
                              fontweight='bold')
    
    # Add colorbar
    cbar2 = plt.colorbar(im2, ax=ax, shrink=0.8)
    cbar2.set_label('Transition Probability')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sleep_stage_transition_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    # plt.show()  # Disabled for automated analysis

def create_transition_statistics(transition_data, output_dir):
    """
    Create comprehensive transition statistics and save to CSV.
    
    Args:
        transition_data (dict): Aggregated transition data
        output_dir (Path): Output directory
    """
    if not transition_data:
        return
    
    stage_names = transition_data['stage_names']
    
    # Create comprehensive transition statistics table
    stats_data = []
    
    # Overall statistics
    stats_data.append({
        'Metric': 'Sampling Interval',
        'Value': '4 seconds',
        'Description': 'Temporal resolution of downsampled labels for transition analysis'
    })
    
    stats_data.append({
        'Metric': 'Total Transitions',
        'Value': f"{transition_data['total_transitions']:,}",
        'Description': 'Total number of time points analyzed (4-second intervals)'
    })
    
    stats_data.append({
        'Metric': 'Total Stage Changes',
        'Value': f"{transition_data['total_changes']:,}",
        'Description': 'Number of actual stage transitions (excluding self-transitions)'
    })
    
    stats_data.append({
        'Metric': 'Change Rate',
        'Value': f"{(transition_data['total_changes'] / transition_data['total_transitions'] * 100):.2f}%",
        'Description': 'Percentage of time points with stage changes'
    })
    
    stats_data.append({
        'Metric': 'Recordings Analyzed',
        'Value': f"{transition_data['n_recordings']}",
        'Description': 'Number of recordings included in analysis'
    })
    
    # Stage-specific statistics
    transition_probs = transition_data['transition_probs']
    change_probs = transition_data['change_only_probs']
    
    for i, stage in enumerate(stage_names):
        # Self-transition probability (stability)
        self_prob = transition_probs[i, i]
        stats_data.append({
            'Metric': f'{stage} Stability',
            'Value': f"{self_prob:.3f}",
            'Description': f'Probability of staying in {stage} stage'
        })
        
        # Most likely transition target (excluding self)
        change_row = change_probs[i, :]
        if np.sum(change_row) > 0:
            max_transition_idx = np.argmax(change_row)
            max_transition_prob = change_row[max_transition_idx]
            target_stage = stage_names[max_transition_idx]
            stats_data.append({
                'Metric': f'{stage} → Most Likely',
                'Value': f"{target_stage} ({max_transition_prob:.3f})",
                'Description': f'Most likely transition from {stage} when change occurs'
            })
    
    # Save statistics to CSV
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(output_dir / 'transition_statistics.csv', index=False)
    print(f"Transition statistics saved to: {output_dir / 'transition_statistics.csv'}")
    
    # Save detailed transition matrices to CSV
    transition_matrix_df = pd.DataFrame(transition_data['transition_probs'], 
                                       index=stage_names, columns=stage_names)
    transition_matrix_df.to_csv(output_dir / 'all_transitions_matrix.csv')
    
    change_matrix_df = pd.DataFrame(transition_data['change_only_probs'], 
                                   index=stage_names, columns=stage_names)
    change_matrix_df.to_csv(output_dir / 'change_only_transitions_matrix.csv')
    
    print(f"Transition matrices saved to: {output_dir}")

if __name__ == "__main__":
    analyze_stage_differences()
