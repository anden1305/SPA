"""
Helper functions for data exploration.
Contains common utilities used across different exploration modules.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Sleep stage mapping
SLEEP_STAGE_MAPPING = {
    1: 'Awake',
    2: 'NREM',
    3: 'REM',
    4: 'Artifact'
}

SLEEP_STAGE_COLORS = {
    'Awake': '#FF6B6B',
    'NREM': '#4ECDC4',
    'REM': '#45B7D1',
    'Artifact': '#FF8C00'  # Orange color for better contrast
}

# Define consistent stage order
STAGE_ORDER = ['Awake', 'NREM', 'REM', 'Artifact']

def load_metadata():
    """Load the complete metadata for all recordings."""
    metadata_path = Path("data/ds006366_processed/metadata.csv")
    return pd.read_csv(metadata_path)

def load_participant_data(participant_id, run_id, data_dir="data/ds006366_processed"):
    """
    Load EEG, EMG, and labels for a specific participant and run.
    
    Args:
        participant_id (str): Participant ID (e.g., 'sub-001')
        run_id (int): Run number
        data_dir (str): Path to processed data directory
    
    Returns:
        dict: Dictionary containing loaded data arrays
    """
    base_path = Path(data_dir) / participant_id / str(run_id)
    
    data = {}
    
    # Load labels (always present)
    data['labels'] = np.load(base_path / 'labels.npy')
    
    # Load available EEG channels
    for eeg_channel in ['EEG1', 'EEG2', 'EEG3', 'EEG4']:
        eeg_path = base_path / f'{eeg_channel}.npy'
        if eeg_path.exists():
            data[eeg_channel] = np.load(eeg_path)
    
    # Load EMG if available
    emg_path = base_path / 'EMG.npy'
    if emg_path.exists():
        data['EMG'] = np.load(emg_path)
    
    return data

def calculate_confidence_interval(data, confidence=0.95):
    """
    Calculate confidence interval for data.
    
    Args:
        data (array-like): Data array
        confidence (float): Confidence level (default 0.95)
    
    Returns:
        tuple: (lower_bound, upper_bound)
    """
    data = np.array(data)
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)  # Standard error of the mean
    t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin_error = t_value * sem
    
    return mean - margin_error, mean + margin_error

def calculate_signal_statistics(signal, labels, sampling_rate=128):
    """
    Calculate comprehensive statistics for signal data grouped by sleep stages.
    
    Args:
        signal (np.array): EEG/EMG signal data
        labels (np.array): Sleep stage labels
        sampling_rate (int): Sampling rate in Hz
    
    Returns:
        dict: Statistics for each sleep stage
    """
    stats_dict = {}
    
    for stage in np.unique(labels):
        stage_mask = labels == stage
        stage_signal = signal[stage_mask]
        
        # Basic statistics
        stage_stats = {
            'mean': np.mean(stage_signal),
            'std': np.std(stage_signal),
            'median': np.median(stage_signal),
            'min': np.min(stage_signal),
            'max': np.max(stage_signal),
            'variance': np.var(stage_signal),
            'skewness': stats.skew(stage_signal),
            'kurtosis': stats.kurtosis(stage_signal),
            'samples': len(stage_signal),
            'duration_minutes': len(stage_signal) / (sampling_rate * 60)
        }
        
        # Confidence intervals
        ci_lower, ci_upper = calculate_confidence_interval(stage_signal)
        stage_stats['ci_lower'] = ci_lower
        stage_stats['ci_upper'] = ci_upper
        
        # Enhanced frequency domain analysis
        freq_stats = calculate_comprehensive_frequency_stats(stage_signal, sampling_rate)
        stage_stats.update(freq_stats)
        
        stats_dict[SLEEP_STAGE_MAPPING[stage]] = stage_stats
    
    return stats_dict

def calculate_comprehensive_frequency_stats(signal, sampling_rate=128):
    """
    Calculate comprehensive frequency domain statistics.
    
    Args:
        signal (np.array): Signal data
        sampling_rate (int): Sampling rate in Hz
    
    Returns:
        dict: Frequency domain statistics
    """
    # Use segments of signal for better frequency resolution
    segment_length = min(len(signal), 60 * sampling_rate)  # Use max 60 seconds
    n_segments = len(signal) // segment_length
    
    all_psds = []
    freqs = None
    
    for i in range(min(n_segments, 10)):  # Analyze up to 10 segments
        start_idx = i * segment_length
        end_idx = start_idx + segment_length
        segment = signal[start_idx:end_idx]
        
        # Calculate power spectral density
        fft = np.fft.fft(segment)
        freqs = np.fft.fftfreq(len(fft), 1/sampling_rate)
        
        # Keep only positive frequencies
        positive_freq_mask = freqs >= 0
        freqs = freqs[positive_freq_mask]
        psd = np.abs(fft[positive_freq_mask])**2
        all_psds.append(psd)
    
    if len(all_psds) == 0:
        return {}
    
    # Average PSD across segments
    avg_psd = np.mean(all_psds, axis=0)
    
    freq_stats = {}
    
    # Traditional frequency bands (absolute power)
    freq_stats['delta_power'] = np.mean(avg_psd[(freqs >= 0.5) & (freqs <= 4)])
    freq_stats['theta_power'] = np.mean(avg_psd[(freqs >= 4) & (freqs <= 8)])
    freq_stats['alpha_power'] = np.mean(avg_psd[(freqs >= 8) & (freqs <= 12)])
    freq_stats['beta_power'] = np.mean(avg_psd[(freqs >= 12) & (freqs <= 30)])
    freq_stats['gamma_power'] = np.mean(avg_psd[(freqs >= 30) & (freqs <= 100)])
    
    # Specific interference frequencies (narrow 50 Hz band per refined spec)
    freq_stats['power_50hz'] = np.mean(avg_psd[(freqs >= 49.5) & (freqs <= 50.5)])
    freq_stats['power_60hz'] = np.mean(avg_psd[(freqs >= 59) & (freqs <= 61)])
    
    # Relative power (normalized by total power)
    total_power = np.sum(avg_psd[(freqs >= 0.5) & (freqs <= 100)])
    if total_power > 0:
        freq_stats['delta_power_rel'] = freq_stats['delta_power'] / total_power
        freq_stats['theta_power_rel'] = freq_stats['theta_power'] / total_power
        freq_stats['alpha_power_rel'] = freq_stats['alpha_power'] / total_power
        freq_stats['beta_power_rel'] = freq_stats['beta_power'] / total_power
        freq_stats['gamma_power_rel'] = freq_stats['gamma_power'] / total_power
    else:
        freq_stats['delta_power_rel'] = 0
        freq_stats['theta_power_rel'] = 0
        freq_stats['alpha_power_rel'] = 0
        freq_stats['beta_power_rel'] = 0
        freq_stats['gamma_power_rel'] = 0
    
    # Additional frequency metrics
    freq_stats['total_power'] = total_power
    freq_stats['peak_frequency'] = freqs[np.argmax(avg_psd)]
    freq_stats['spectral_centroid'] = np.sum(freqs * avg_psd) / np.sum(avg_psd)
    freq_stats['spectral_bandwidth'] = np.sqrt(np.sum(((freqs - freq_stats['spectral_centroid'])**2) * avg_psd) / np.sum(avg_psd))
    
    # Store full PSD for detailed analysis
    freq_stats['psd'] = avg_psd
    freq_stats['frequencies'] = freqs
    
    return freq_stats

def plot_signal_statistics_comparison(stats_data, metric='mean', title_prefix='Signal', save_path=None):
    """
    Create a comparison plot of signal statistics across sleep stages.
    
    Args:
        stats_data (dict): Dictionary of statistics data
        metric (str): Metric to plot
        title_prefix (str): Prefix for plot title
        save_path (str): Path to save the plot
    """
    stages = list(stats_data.keys())
    values = [stats_data[stage][metric] for stage in stages]
    
    if metric in ['mean', 'median']:
        # Include confidence intervals for mean/median
        ci_lower = [stats_data[stage]['ci_lower'] for stage in stages]
        ci_upper = [stats_data[stage]['ci_upper'] for stage in stages]
        errors = [[values[i] - ci_lower[i] for i in range(len(values))],
                  [ci_upper[i] - values[i] for i in range(len(values))]]
    else:
        errors = None
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(stages, values, color=[SLEEP_STAGE_COLORS[stage] for stage in stages],
                   alpha=0.8, edgecolor='black', linewidth=1)
    
    if errors:
        plt.errorbar(stages, values, yerr=errors, fmt='none', color='black', capsize=5)
    
    plt.title(f'{title_prefix} {metric.capitalize()} by Sleep Stage', fontsize=14, fontweight='bold')
    plt.xlabel('Sleep Stage', fontsize=12)
    plt.ylabel(f'{metric.capitalize()}', fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.close()  # Disabled for automated analysis

def create_summary_table(stats_data, metrics=['mean', 'std', 'median'], save_path=None):
    """
    Create a summary table of statistics across sleep stages.
    
    Args:
        stats_data (dict): Dictionary of statistics data
        metrics (list): List of metrics to include
        save_path (str): Path to save the table
    
    Returns:
        pd.DataFrame: Summary table
    """
    summary_data = {}
    
    for stage in stats_data.keys():
        summary_data[stage] = {}
        for metric in metrics:
            if metric in stats_data[stage]:
                summary_data[stage][metric] = stats_data[stage][metric]
    
    df = pd.DataFrame(summary_data).T
    
    if save_path:
        df.to_csv(save_path)
        print(f"Summary table saved to: {save_path}")
    
    return df

def calculate_spectral_confidence_intervals(psds, confidence=0.95):
    """
    Calculate confidence intervals for power spectral density data.
    
    Args:
        psds (list): List of PSD arrays from multiple segments/recordings
        confidence (float): Confidence level (default 0.95)
    
    Returns:
        tuple: (mean_psd, ci_lower, ci_upper)
    """
    if len(psds) == 0:
        return None, None, None
    
    psds_array = np.array(psds)
    mean_psd = np.mean(psds_array, axis=0)
    
    if len(psds) > 1:
        # Calculate confidence intervals for each frequency bin
        n = len(psds)
        sem = stats.sem(psds_array, axis=0)  # Standard error of mean across recordings
        t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
        margin_error = t_value * sem
        
        ci_lower = mean_psd - margin_error
        ci_upper = mean_psd + margin_error
    else:
        ci_lower = mean_psd
        ci_upper = mean_psd
    
    return mean_psd, ci_lower, ci_upper

def downsample_labels(labels, sample_rate=128, target_interval_seconds=4):
    """
    Downsample labels to reduce temporal resolution.
    
    Args:
        labels (np.array): Array of sleep stage labels sampled at high frequency
        sample_rate (int): Original sampling rate in Hz (default: 128)
        target_interval_seconds (float): Target interval between samples in seconds (default: 4)
    
    Returns:
        np.array: Downsampled labels
    """
    # Calculate downsampling factor
    downsample_factor = int(sample_rate * target_interval_seconds)
    
    # Downsample by taking every nth sample
    downsampled_labels = labels[::downsample_factor]
    
    print(f"    Downsampled from {len(labels)} to {len(downsampled_labels)} samples "
          f"(factor: {downsample_factor}, interval: {target_interval_seconds}s)")
    
    return downsampled_labels

def calculate_stage_transitions(labels):
    """
    Calculate transition matrices for sleep stages.
    
    Args:
        labels (np.array): Array of sleep stage labels
    
    Returns:
        dict: Dictionary containing transition matrices and statistics
    """
    from collections import defaultdict
    
    unique_stages = sorted(np.unique(labels))
    stage_names = [SLEEP_STAGE_MAPPING[stage] for stage in unique_stages]
    n_stages = len(unique_stages)
    
    # Initialize transition count matrices
    transition_counts = np.zeros((n_stages, n_stages))
    change_only_counts = np.zeros((n_stages, n_stages))
    
    # Count transitions
    prev_stage = labels[0]
    for i in range(1, len(labels)):
        curr_stage = labels[i]
        
        prev_idx = unique_stages.index(prev_stage)
        curr_idx = unique_stages.index(curr_stage)
        
        # Count all transitions (including staying in same stage)
        transition_counts[prev_idx, curr_idx] += 1
        
        # Count only actual changes
        if prev_stage != curr_stage:
            change_only_counts[prev_idx, curr_idx] += 1
        
        prev_stage = curr_stage
    
    # Calculate transition probabilities
    # All transitions (including self-transitions)
    row_sums = np.sum(transition_counts, axis=1)
    transition_probs = np.zeros_like(transition_counts)
    for i in range(n_stages):
        if row_sums[i] > 0:
            transition_probs[i, :] = transition_counts[i, :] / row_sums[i]
    
    # Change-only transitions (excluding self-transitions)
    change_row_sums = np.sum(change_only_counts, axis=1)
    change_probs = np.zeros_like(change_only_counts)
    for i in range(n_stages):
        if change_row_sums[i] > 0:
            change_probs[i, :] = change_only_counts[i, :] / change_row_sums[i]
    
    return {
        'stage_names': stage_names,
        'unique_stages': unique_stages,
        'transition_counts': transition_counts,
        'transition_probs': transition_probs,
        'change_only_counts': change_only_counts,
        'change_only_probs': change_probs,
        'total_transitions': np.sum(transition_counts),
        'total_changes': np.sum(change_only_counts)
    }

def ensure_output_directory(output_dir="results/data_exploration"):
    """Ensure output directory exists."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return output_dir
