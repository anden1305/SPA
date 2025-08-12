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
        print("No data available for participant analysis.")
        return
    
    print(f"\nCollected data from {len(set([data['participant_id'] for data in all_participant_stats.values()]))} participants")
    
    # Create comprehensive visualizations
    create_participant_signal_overview(all_participant_stats, participant_output_dir)
    create_participant_frequency_analysis(all_participant_stats, participant_output_dir)
    create_participant_sleep_profile_analysis(all_participant_stats, participant_output_dir)
    create_participant_variability_analysis(all_participant_stats, participant_output_dir)
    create_participant_summary_statistics(all_participant_stats, participant_output_dir)
    
    print(f"\nParticipant difference analysis completed. Results saved to: {participant_output_dir}")

def select_representative_participants(metadata):
    """
    Select representative participants from a single lab to control for lab differences.
    
    Args:
        metadata (pd.DataFrame): Metadata dataframe
    
    Returns:
        list: List of (participant_id, lab) tuples
    """
    selected = []
    
    # Choose the lab with the most participants for robust analysis
    lab_counts = metadata['lab'].value_counts()
    target_lab = lab_counts.index[0]  # Lab with most participants
    
    print(f"Selected {target_lab} for within-lab analysis (has {lab_counts[target_lab]} participants)")
    
    # Select all participants from the target lab (up to 15 for manageable analysis)
    lab_participants = metadata[metadata['lab'] == target_lab]['participant_id'].unique()
    
    # Take up to 15 participants from the target lab
    for participant in lab_participants[:15]:
        selected.append((participant, target_lab))
    
    return selected

def create_participant_signal_overview(all_stats, output_dir):
    """
    Create overview plots showing signal characteristics across participants.
    
    Args:
        all_stats (dict): All participant statistics
        output_dir (Path): Output directory
    """
    print("Creating participant signal overview...")
    
    # Organize data by signal type
    eeg_data = {}
    emg_data = {}
    
    for key, data in all_stats.items():
        if 'EEG' in data['signal']:
            if data['participant_id'] not in eeg_data:
                eeg_data[data['participant_id']] = {}
            eeg_data[data['participant_id']][data['stage']] = data
        elif 'EMG' in data['signal']:
            if data['participant_id'] not in emg_data:
                emg_data[data['participant_id']] = {}
            emg_data[data['participant_id']][data['stage']] = data
    
    # Create comprehensive participant comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('EEG Signal Characteristics Across Participants', fontsize=16, fontweight='bold')
    
    participants = list(eeg_data.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(participants)))
    
    metrics = [
        ('mean', 'Mean Amplitude (µV)', axes[0, 0]),
        ('std', 'Standard Deviation (µV)', axes[0, 1]),
        ('delta_power', 'Delta Power (µV²/Hz)', axes[0, 2]),
        ('theta_power', 'Theta Power (µV²/Hz)', axes[1, 0]),
        ('alpha_power', 'Alpha Power (µV²/Hz)', axes[1, 1]),
        ('total_power', 'Total Power (µV²/Hz)', axes[1, 2])
    ]
    
    for metric, ylabel, ax in metrics:
        create_participant_metric_plot(eeg_data, participants, colors, metric, ylabel, ax)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'participant_EEG_signal_overview.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_participant_metric_plot(data, participants, colors, metric, ylabel, ax):
    """Create a single metric comparison plot across participants."""
    
    # Collect data for each stage across participants
    stages = ['Awake', 'NREM', 'REM']
    
    for stage in stages:
        values = []
        participant_labels = []
        
        for participant in participants:
            if participant in data and stage in data[participant]:
                if metric in data[participant][stage]:
                    values.append(data[participant][stage][metric])
                    participant_labels.append(participant)
        
        if values:
            # Use different positions for each stage
            stage_offset = stages.index(stage) * 0.25
            x_positions = [i + stage_offset for i in range(len(participant_labels))]
            
            ax.scatter(x_positions, values, label=f'{stage}', alpha=0.7, s=60,
                      color=SLEEP_STAGE_COLORS.get(stage, 'gray'))
    
    ax.set_xlabel('Participants')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{ylabel} by Participant', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log' if 'power' in ylabel.lower() else 'linear')
    
    # Set x-axis labels
    if len(participants) <= 8:  # Only show labels if not too many
        ax.set_xticks(range(len(participants)))
        ax.set_xticklabels(participants, rotation=45)

def create_participant_frequency_analysis(all_stats, output_dir):
    """
    Create comprehensive frequency domain analysis across participants.
    
    Args:
        all_stats (dict): All participant statistics
        output_dir (Path): Output directory
    """
    print("Creating participant frequency analysis...")
    
    # Organize EEG data by participant
    eeg_data = {}
    for key, data in all_stats.items():
        if 'EEG' in data['signal']:
            participant = data['participant_id']
            if participant not in eeg_data:
                eeg_data[participant] = {}
            eeg_data[participant][data['stage']] = data
    
    # Create frequency band comparison across participants
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Frequency Band Power Across Participants', fontsize=16, fontweight='bold')
    
    participants = list(eeg_data.keys())
    
    # Individual frequency bands
    bands = [
        ('delta_power', 'Delta Power (0.5-4 Hz)', axes[0, 0]),
        ('theta_power', 'Theta Power (4-8 Hz)', axes[0, 1]),
        ('alpha_power', 'Alpha Power (8-12 Hz)', axes[1, 0]),
        ('beta_power', 'Beta Power (12-30 Hz)', axes[1, 1])
    ]
    
    for band_key, title, ax in bands:
        create_participant_frequency_band_plot(eeg_data, participants, band_key, title, ax)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'participant_frequency_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create participant-specific spectral profiles
    create_participant_spectral_profiles(eeg_data, participants, output_dir)

def create_participant_frequency_band_plot(data, participants, band_key, title, ax):
    """Create frequency band comparison plot."""
    
    stages = ['Awake', 'NREM', 'REM']
    x = np.arange(len(participants))
    width = 0.25
    
    for i, stage in enumerate(stages):
        values = []
        for participant in participants:
            if participant in data and stage in data[participant]:
                if band_key in data[participant][stage]:
                    values.append(data[participant][stage][band_key])
                else:
                    values.append(0)
            else:
                values.append(0)
        
        ax.bar(x + i * width, values, width, label=stage, alpha=0.8,
               color=SLEEP_STAGE_COLORS.get(stage, 'gray'))
    
    ax.set_xlabel('Participants')
    ax.set_ylabel('Power (µV²/Hz)')
    ax.set_title(title, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(participants, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')

def create_participant_spectral_profiles(eeg_data, participants, output_dir):
    """Create individual spectral profiles for each participant."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Individual Participant Spectral Profiles', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for i, participant in enumerate(participants[:6]):  # Show first 6 participants
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        if participant in eeg_data:
            bands = ['delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power']
            band_labels = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
            
            for stage in ['Awake', 'NREM', 'REM']:
                if stage in eeg_data[participant]:
                    powers = []
                    for band in bands:
                        if band in eeg_data[participant][stage]:
                            powers.append(eeg_data[participant][stage][band])
                        else:
                            powers.append(0)
                    
                    ax.plot(band_labels, powers, marker='o', label=stage, linewidth=2,
                           color=SLEEP_STAGE_COLORS.get(stage, 'gray'))
        
        ax.set_title(f'{participant}', fontweight='bold')
        ax.set_ylabel('Power (µV²/Hz)')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    # Hide unused subplots
    for i in range(len(participants), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'participant_spectral_profiles.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_participant_sleep_profile_analysis(all_stats, output_dir):
    """
    Create analysis of sleep-specific characteristics per participant.
    
    Args:
        all_stats (dict): All participant statistics
        output_dir (Path): Output directory
    """
    print("Creating participant sleep profile analysis...")
    
    # Organize data by participant
    participant_profiles = {}
    for key, data in all_stats.items():
        participant = data['participant_id']
        if participant not in participant_profiles:
            participant_profiles[participant] = {'lab': data['lab'], 'stages': {}}
        
        if 'EEG' in data['signal']:  # Focus on EEG for sleep profiles
            participant_profiles[participant]['stages'][data['stage']] = data
    
    # Create sleep efficiency and stage characteristics analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Participant Sleep Profile Characteristics', fontsize=16, fontweight='bold')
    
    participants = list(participant_profiles.keys())
    
    # Sleep stage ratios and characteristics
    create_sleep_stage_ratios(participant_profiles, participants, axes[0, 0])
    create_sleep_intensity_analysis(participant_profiles, participants, axes[0, 1])
    create_arousal_indicators(participant_profiles, participants, axes[1, 0])
    create_sleep_efficiency_metrics(participant_profiles, participants, axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'participant_sleep_profiles.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_sleep_stage_ratios(profiles, participants, ax):
    """Create sleep stage ratio analysis."""
    
    # Calculate theta/delta ratios (sleep depth indicator)
    ratios = {}
    for participant in participants:
        if participant in profiles:
            ratios[participant] = {}
            for stage in ['Awake', 'NREM', 'REM']:
                if stage in profiles[participant]['stages']:
                    stage_stats = profiles[participant]['stages'][stage]
                    theta = stage_stats.get('theta_power', 0)
                    delta = stage_stats.get('delta_power', 1)  # Avoid division by zero
                    ratios[participant][stage] = theta / delta if delta > 0 else 0
    
    stages = ['Awake', 'NREM', 'REM']
    x = np.arange(len(participants))
    width = 0.25
    
    for i, stage in enumerate(stages):
        values = [ratios.get(p, {}).get(stage, 0) for p in participants]
        ax.bar(x + i * width, values, width, label=stage, alpha=0.8,
               color=SLEEP_STAGE_COLORS.get(stage, 'gray'))
    
    ax.set_xlabel('Participants')
    ax.set_ylabel('Theta/Delta Ratio')
    ax.set_title('Sleep Depth Indicator (Theta/Delta Ratio)', fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(participants, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

def create_sleep_intensity_analysis(profiles, participants, ax):
    """Create sleep intensity analysis based on delta power."""
    
    # Use delta power as sleep intensity indicator
    delta_powers = {}
    for participant in participants:
        if participant in profiles:
            delta_powers[participant] = {}
            for stage in ['NREM', 'REM']:  # Focus on sleep stages
                if stage in profiles[participant]['stages']:
                    delta_powers[participant][stage] = profiles[participant]['stages'][stage].get('delta_power', 0)
    
    stages = ['NREM', 'REM']
    x = np.arange(len(participants))
    width = 0.4
    
    for i, stage in enumerate(stages):
        values = [delta_powers.get(p, {}).get(stage, 0) for p in participants]
        ax.bar(x + i * width, values, width, label=stage, alpha=0.8,
               color=SLEEP_STAGE_COLORS.get(stage, 'gray'))
    
    ax.set_xlabel('Participants')
    ax.set_ylabel('Delta Power (µV²/Hz)')
    ax.set_title('Sleep Intensity (Delta Power)', fontweight='bold')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(participants, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')

def create_arousal_indicators(profiles, participants, ax):
    """Create arousal indicator analysis."""
    
    # Use beta/alpha ratio as arousal indicator
    arousal_ratios = {}
    for participant in participants:
        if participant in profiles:
            arousal_ratios[participant] = {}
            for stage in ['Awake', 'NREM', 'REM']:
                if stage in profiles[participant]['stages']:
                    stage_stats = profiles[participant]['stages'][stage]
                    beta = stage_stats.get('beta_power', 0)
                    alpha = stage_stats.get('alpha_power', 1)  # Avoid division by zero
                    arousal_ratios[participant][stage] = beta / alpha if alpha > 0 else 0
    
    stages = ['Awake', 'NREM', 'REM']
    x = np.arange(len(participants))
    width = 0.25
    
    for i, stage in enumerate(stages):
        values = [arousal_ratios.get(p, {}).get(stage, 0) for p in participants]
        ax.bar(x + i * width, values, width, label=stage, alpha=0.8,
               color=SLEEP_STAGE_COLORS.get(stage, 'gray'))
    
    ax.set_xlabel('Participants')
    ax.set_ylabel('Beta/Alpha Ratio')
    ax.set_title('Arousal Indicators (Beta/Alpha Ratio)', fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(participants, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

def create_sleep_efficiency_metrics(profiles, participants, ax):
    """Create sleep efficiency metrics."""
    
    # Calculate total power as sleep efficiency indicator
    total_powers = {}
    for participant in participants:
        if participant in profiles:
            total_powers[participant] = {}
            for stage in ['Awake', 'NREM', 'REM']:
                if stage in profiles[participant]['stages']:
                    total_powers[participant][stage] = profiles[participant]['stages'][stage].get('total_power', 0)
    
    stages = ['Awake', 'NREM', 'REM']
    x = np.arange(len(participants))
    width = 0.25
    
    for i, stage in enumerate(stages):
        values = [total_powers.get(p, {}).get(stage, 0) for p in participants]
        ax.bar(x + i * width, values, width, label=stage, alpha=0.8,
               color=SLEEP_STAGE_COLORS.get(stage, 'gray'))
    
    ax.set_xlabel('Participants')
    ax.set_ylabel('Total Power (µV²/Hz)')
    ax.set_title('Sleep Efficiency (Total Power)', fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(participants, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')

def create_participant_variability_analysis(all_stats, output_dir):
    """
    Create analysis of variability across participants.
    
    Args:
        all_stats (dict): All participant statistics
        output_dir (Path): Output directory
    """
    print("Creating participant variability analysis...")
    
    # Organize data for variability analysis
    eeg_data = {}
    for key, data in all_stats.items():
        if 'EEG' in data['signal']:
            participant = data['participant_id']
            stage = data['stage']
            
            if stage not in eeg_data:
                eeg_data[stage] = {}
            if participant not in eeg_data[stage]:
                eeg_data[stage][participant] = data
    
    # Create variability plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Participant Variability Analysis', fontsize=16, fontweight='bold')
    
    # Inter-participant variability for different metrics
    create_variability_plot(eeg_data, 'mean', 'Mean Amplitude Variability', axes[0, 0])
    create_variability_plot(eeg_data, 'delta_power', 'Delta Power Variability', axes[0, 1])
    create_variability_plot(eeg_data, 'alpha_power', 'Alpha Power Variability', axes[1, 0])
    create_coefficient_of_variation_plot(eeg_data, axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'participant_variability_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_variability_plot(data, metric, title, ax):
    """Create individual variability plot."""
    
    stages = ['Awake', 'NREM', 'REM']
    stage_variability = {}
    
    for stage in stages:
        if stage in data:
            values = []
            for participant, stats in data[stage].items():
                if metric in stats:
                    values.append(stats[metric])
            
            if values:
                stage_variability[stage] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'cv': np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
                }
    
    # Plot coefficient of variation for each stage
    stages_with_data = list(stage_variability.keys())
    cv_values = [stage_variability[stage]['cv'] for stage in stages_with_data]
    colors = [SLEEP_STAGE_COLORS.get(stage, 'gray') for stage in stages_with_data]
    
    bars = ax.bar(stages_with_data, cv_values, color=colors, alpha=0.8)
    ax.set_ylabel('Coefficient of Variation')
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, cv in zip(bars, cv_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(cv_values)*0.01,
                f'{cv:.3f}', ha='center', va='bottom', fontweight='bold')

def create_coefficient_of_variation_plot(data, ax):
    """Create comprehensive coefficient of variation comparison."""
    
    metrics = ['mean', 'std', 'delta_power', 'theta_power', 'alpha_power']
    metric_labels = ['Mean Amp', 'Std Dev', 'Delta', 'Theta', 'Alpha']
    
    stages = ['Awake', 'NREM', 'REM']
    
    # Calculate CV for each metric and stage
    cv_matrix = np.zeros((len(stages), len(metrics)))
    
    for i, stage in enumerate(stages):
        for j, metric in enumerate(metrics):
            if stage in data:
                values = []
                for participant, stats in data[stage].items():
                    if metric in stats:
                        values.append(stats[metric])
                
                if values and np.mean(values) > 0:
                    cv_matrix[i, j] = np.std(values) / np.mean(values)
    
    # Create heatmap
    im = ax.imshow(cv_matrix, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(metric_labels)))
    ax.set_xticklabels(metric_labels, rotation=45)
    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels(stages)
    ax.set_title('Coefficient of Variation Heatmap', fontweight='bold')
    
    # Add text annotations
    for i in range(len(stages)):
        for j in range(len(metrics)):
            text = ax.text(j, i, f'{cv_matrix[i, j]:.2f}',
                          ha="center", va="center", color="white" if cv_matrix[i, j] > 0.5 else "black",
                          fontweight='bold')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, shrink=0.8)

def create_participant_summary_statistics(all_stats, output_dir):
    """
    Create comprehensive summary statistics for participants.
    
    Args:
        all_stats (dict): All participant statistics
        output_dir (Path): Output directory
    """
    print("Creating participant summary statistics...")
    
    # Organize data by participant
    summary_data = []
    
    participants = list(set([data['participant_id'] for data in all_stats.values()]))
    
    for participant in participants:
        # Get participant's lab
        participant_lab = None
        participant_data = {}
        
        for key, data in all_stats.items():
            if data['participant_id'] == participant and 'EEG' in data['signal']:
                participant_lab = data['lab']
                stage = data['stage']
                participant_data[stage] = data
        
        # Calculate summary metrics
        if participant_data:
            # Sleep quality indicators
            delta_power_nrem = participant_data.get('NREM', {}).get('delta_power', 0)
            theta_power_rem = participant_data.get('REM', {}).get('theta_power', 0)
            alpha_power_awake = participant_data.get('Awake', {}).get('alpha_power', 0)
            
            # Arousal indicators
            beta_alpha_ratio_awake = 0
            if 'Awake' in participant_data:
                beta = participant_data['Awake'].get('beta_power', 0)
                alpha = participant_data['Awake'].get('alpha_power', 1)
                beta_alpha_ratio_awake = beta / alpha if alpha > 0 else 0
            
            # Sleep depth indicators
            theta_delta_ratio_nrem = 0
            if 'NREM' in participant_data:
                theta = participant_data['NREM'].get('theta_power', 0)
                delta = participant_data['NREM'].get('delta_power', 1)
                theta_delta_ratio_nrem = theta / delta if delta > 0 else 0
            
            summary_data.append({
                'Participant_ID': participant,
                'Laboratory': participant_lab,
                'Delta_Power_NREM': f"{delta_power_nrem:.2e}",
                'Theta_Power_REM': f"{theta_power_rem:.2e}",
                'Alpha_Power_Awake': f"{alpha_power_awake:.2e}",
                'Arousal_Ratio_Awake': f"{beta_alpha_ratio_awake:.3f}",
                'Sleep_Depth_Ratio_NREM': f"{theta_delta_ratio_nrem:.3f}",
                'Sleep_Quality_Score': f"{(delta_power_nrem + theta_power_rem):.2e}"
            })
    
    # Save to CSV
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / 'participant_differences_summary.csv', index=False)
    print(f"Participant summary saved to: {output_dir / 'participant_differences_summary.csv'}")
    
    # Create detailed statistics by metric
    create_detailed_participant_statistics(all_stats, output_dir)

def create_detailed_participant_statistics(all_stats, output_dir):
    """Create detailed participant statistics table."""
    
    detailed_data = []
    
    for key, data in all_stats.items():
        if 'EEG' in data['signal']:
            stats = data
            
            detailed_data.append({
                'Participant_ID': data['participant_id'],
                'Laboratory': data['lab'],
                'Signal_Type': data['signal'],
                'Sleep_Stage': data['stage'],
                'Mean_Amplitude': f"{stats.get('mean', 0):.6f}",
                'Std_Deviation': f"{stats.get('std', 0):.6f}",
                'Variance': f"{stats.get('variance', 0):.8f}",
                'Skewness': f"{stats.get('skewness', 0):.4f}",
                'Delta_Power': f"{stats.get('delta_power', 0):.2e}",
                'Theta_Power': f"{stats.get('theta_power', 0):.2e}",
                'Alpha_Power': f"{stats.get('alpha_power', 0):.2e}",
                'Beta_Power': f"{stats.get('beta_power', 0):.2e}",
                'Gamma_Power': f"{stats.get('gamma_power', 0):.2e}",
                'Total_Power': f"{stats.get('total_power', 0):.2e}",
                'Peak_Frequency': f"{stats.get('peak_frequency', 0):.2f}",
                'Spectral_Centroid': f"{stats.get('spectral_centroid', 0):.2f}"
            })
    
    # Save detailed statistics
    detailed_df = pd.DataFrame(detailed_data)
    detailed_df.to_csv(output_dir / 'participant_differences_detailed.csv', index=False)
    print(f"Detailed participant statistics saved to: {output_dir / 'participant_differences_detailed.csv'}")

if __name__ == "__main__":
    analyze_participant_differences()
