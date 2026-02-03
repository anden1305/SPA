from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style
sns.set_theme(style="whitegrid", context="paper")


def pca_scatter_random_samples(
    datapoints: np.ndarray,                # (N, F)
    labels: np.ndarray,                    # (N,)
    label_names: Sequence[str],            # length C (for labels 0..C-1)
    analysis_name: str,
    seed: int,
    n_samples: int,
    save_path: str,
    *,
    sample_without_replacement: bool = True,
    point_size: float = 18.0,
    alpha: float = 0.75,
    center: bool = True,
    standardize: bool = False,
    return_details: bool = False,
) -> Union[None, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    1) Fits PCA on ALL datapoints (N,F).
    2) Randomly selects n_samples datapoints and projects them into PCA space.
    3) Scatter plot of selected points in (PC1, PC2) colored by their labels.
    4) Legend uses label_names.
    5) Title uses analysis_name.
    6) Saves figure to save_path.

    Notes:
    - PCA is implemented with SVD (no sklearn dependency).
    - If standardize=True, features are z-scored before PCA (after optional centering).
    - Returns (selected_indices, pc_scores_selected, explained_variance_ratio) if return_details=True.
    """
    # -----------------
    # Input validation
    # -----------------
    if not isinstance(datapoints, np.ndarray):
        raise TypeError("datapoints must be a numpy array.")
    if not isinstance(labels, np.ndarray):
        raise TypeError("labels must be a numpy array.")
    if datapoints.ndim != 2:
        raise ValueError(f"datapoints must have shape (N,F). Got {datapoints.shape}.")
    if labels.ndim != 1:
        raise ValueError(f"labels must have shape (N,). Got {labels.shape}.")
    N, F = datapoints.shape
    if labels.shape[0] != N:
        raise ValueError(f"labels length must equal N={N}. Got {labels.shape[0]}.")
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0.")
    if sample_without_replacement and n_samples > N:
        raise ValueError(f"n_samples={n_samples} cannot exceed N={N} without replacement.")
    if len(label_names) == 0:
        raise ValueError("label_names must be non-empty.")
    if not isinstance(analysis_name, str) or not analysis_name.strip():
        raise ValueError("analysis_name must be a non-empty string.")
    if not isinstance(save_path, str) or not save_path.strip():
        raise ValueError("save_path must be a non-empty string.")

    # Ensure labels are integer-like
    if not np.issubdtype(labels.dtype, np.integer):
        # try to safely cast if they look integer-like
        if np.all(np.isfinite(labels)) and np.all(np.equal(labels, np.round(labels))):
            labels = labels.astype(int)
        else:
            raise ValueError("labels must be integer dtype (or safely castable to integers).")

    # Decide number of classes from label_names (assumes labels are 0..C-1)
    C = len(label_names)
    if labels.min() < 0:
        raise ValueError("labels must be >= 0.")
    if labels.max() >= C:
        raise ValueError(
            f"labels contain value {labels.max()} but label_names has length {C}. "
            "Expected labels in [0, C-1]."
        )

    # -----------------
    # PCA fit on ALL data
    # -----------------
    X = datapoints.astype(np.float64, copy=False)

    # Center / standardize
    mu = X.mean(axis=0) if center else np.zeros(F, dtype=np.float64)
    Xc = X - mu

    if standardize:
        sigma = Xc.std(axis=0, ddof=1)
        sigma[sigma == 0.0] = 1.0
        Xc = Xc / sigma

    # SVD: Xc = U S Vt, principal axes are rows of Vt
    # PCs scores for a sample x: (x_centered) @ V, where V = Vt.T
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt.T  # (F, F)
    # explained variance ratio
    # eigenvalues of covariance: (S^2) / (N-1)
    if N > 1:
        eigvals = (S ** 2) / (N - 1)
        explained_variance_ratio = eigvals / eigvals.sum()
    else:
        explained_variance_ratio = np.zeros_like(S)

    # -----------------
    # Sample + project
    # -----------------
    rng = np.random.default_rng(seed)
    if sample_without_replacement:
        idx = rng.choice(N, size=n_samples, replace=False)
    else:
        idx = rng.choice(N, size=n_samples, replace=True)

    X_sel = X[idx]
    X_sel_c = X_sel - mu
    if standardize:
        X_sel_c = X_sel_c / sigma

    # scores in PC space (n_samples, F)
    scores_sel = X_sel_c @ V
    pc1 = scores_sel[:, 0]
    pc2 = scores_sel[:, 1]
    y_sel = labels[idx]

    # -----------------
    # Plot
    # -----------------
    fig, ax = plt.subplots(figsize=(9, 7))

    # Use discrete colormap for C classes
    cmap = plt.get_cmap("tab10" if C <= 10 else "tab20")
    colors = [cmap(i % cmap.N) for i in range(C)]

    # Plot per class to get clean legend entries
    for c in range(C):
        mask = (y_sel == c)
        if not np.any(mask):
            continue
        ax.scatter(
            pc1[mask],
            pc2[mask],
            s=point_size,
            alpha=alpha,
            label=label_names[c],
            color=colors[c],
            edgecolors="none",
        )

    evr1 = explained_variance_ratio[0] if explained_variance_ratio.size > 0 else 0.0
    evr2 = explained_variance_ratio[1] if explained_variance_ratio.size > 1 else 0.0

    ax.set_xlabel(f"PC1 ({evr1*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({evr2*100:.1f}% var)")
    ax.set_title(analysis_name)
    ax.legend(loc="best", frameon=True)
    ax.grid(True, linewidth=0.5, alpha=0.35)

    # -----------------
    # Save
    # -----------------
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    if return_details:
        return idx, scores_sel, explained_variance_ratio
    return None


def plot_substage_frequency_distribution(
    data: np.ndarray,                       # (N, T, C) - data with channels
    labels: np.ndarray,                     # (N, T) - predicted substage labels
    channel_names: Sequence[str],           # channel names (e.g., ["EEG1", "EEG2", "EMG"])
    n_bins: int,                            # number of histogram bins
    plot_title: str,
    save_path: str,
    *,
    figsize: Optional[Tuple[float, float]] = None,
    log_scale: bool = True,
) -> None:
    """
    Creates a grid of frequency distribution plots for each substage and channel.
    
    Args:
        data: (N, T, C) array where N=samples, T=timesteps, C=channels
        labels: (N, T) array of predicted substage labels (0-indexed)
        channel_names: Names of each channel
        n_bins: Number of bins for histograms
        plot_title: Overall plot title
        save_path: Where to save the figure
        figsize: Optional figure size (width, height). Auto-computed if None.
        log_scale: Whether to use log scale for y-axis
    """
    # Input validation
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a numpy array.")
    if not isinstance(labels, np.ndarray):
        raise TypeError("labels must be a numpy array.")
    if data.ndim != 3:
        raise ValueError(f"data must have shape (N,T,C). Got {data.shape}.")
    if labels.ndim != 2:
        raise ValueError(f"labels must have shape (N,T). Got {labels.shape}.")
    
    N, T, C = data.shape
    if labels.shape != (N, T):
        raise ValueError(
            f"labels shape {labels.shape} must match data shape (N,T) = ({N},{T})."
        )
    if len(channel_names) != C:
        raise ValueError(f"channel_names length {len(channel_names)} must equal C={C}.")
    if n_bins <= 0:
        raise ValueError("n_bins must be > 0.")
    
    # Ensure labels are integers
    if not np.issubdtype(labels.dtype, np.integer):
        if np.all(np.isfinite(labels)) and np.all(np.equal(labels, np.round(labels))):
            labels = labels.astype(int)
        else:
            raise ValueError("labels must be integer dtype (or safely castable to integers).")
    
    # Determine number of unique substages
    unique_substages = np.unique(labels)
    n_substages = len(unique_substages)
    
    if labels.min() < 0:
        raise ValueError("labels must be >= 0.")
    
    # Flatten data and labels for easier indexing
    data_flat = data.reshape(-1, C)  # (N*T, C)
    labels_flat = labels.ravel()      # (N*T,)
    
    # Calculate counts and percentages for each substage
    total_points = len(labels_flat)
    substage_counts = {}
    substage_percentages = {}
    for substage in unique_substages:
        count = np.sum(labels_flat == substage)
        substage_counts[substage] = count
        substage_percentages[substage] = (count / total_points) * 100.0
    
    # Auto-compute figure size if not provided
    if figsize is None:
        width = 4 * C
        height = 2.5 * n_substages
        figsize = (width, height)
    
    # Create subplot grid: rows = substages, cols = channels
    fig, axes = plt.subplots(
        nrows=n_substages,
        ncols=C,
        figsize=figsize,
        sharex='col',
        sharey=False,
    )
    
    # Ensure axes is 2D even for single row/column
    if n_substages == 1 and C == 1:
        axes = np.array([[axes]])
    elif n_substages == 1:
        axes = axes.reshape(1, -1)
    elif C == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot each substage × channel combination
    for row_idx, substage in enumerate(unique_substages):
        # Get data points belonging to this substage
        mask = labels_flat == substage
        substage_data = data_flat[mask]  # (n_points, C)
        
        for col_idx, channel_idx in enumerate(range(C)):
            ax = axes[row_idx, col_idx]
            
            # Extract channel values for this substage
            channel_values = substage_data[:, channel_idx]
            
            # Create histogram
            if len(channel_values) > 0:
                ax.hist(
                    channel_values,
                    bins=n_bins,
                    color='#2E86AB',
                    edgecolor='none',
                )
            
            # Set y-axis to log scale if requested
            if log_scale:
                ax.set_yscale('log')
            
            # Add column titles (channel names) only on top row
            if row_idx == 0:
                ax.set_title(channel_names[col_idx], fontsize=12, fontweight='bold')
            
            # Add row labels (substage names + count + percentage) only on leftmost column
            if col_idx == 0:
                count = substage_counts[substage]
                pct = substage_percentages[substage]
                ylabel = f'Substage {substage}\nn={count} ({pct:.1f}%)'
                ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
            
            # Add grid
            ax.grid(True, alpha=0.3, linewidth=0.5)
    
    # Add overall title
    fig.suptitle(plot_title, fontsize=14, fontweight='bold', y=0.995)
    
    # Adjust layout
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_substage_frequency_spectrum(
    data: np.ndarray,                       # (N, T, C) - windowed raw channel data
    labels: np.ndarray,                     # (N, T) - predicted substage labels
    true_labels: np.ndarray,                # (N*T,) - true labels (Awake/NREM/REM)
    channel_names: Sequence[str],           # channel names (e.g., ["EEG1", "EEG2", "EMG"])
    sampling_rate: float,                   # sampling rate in Hz
    freq_ranges: Sequence[Tuple[float, float]],  # frequency ranges per channel [(0.5,30), (0.5,30), (1,60)]
    plot_title: str,
    save_path: str,
    *,
    figsize: Optional[Tuple[float, float]] = None,
    window_size: int = 512,                 # window size for FFT
    min_windows: int = 50,                  # minimum number of windows required for reliable spectrum
    normalize: bool = True,                 # whether to normalize PSD by total power
) -> None:
    """
    Creates a grid of power spectral density plots (rows=substages, cols=channels).
    
    Shows average power spectrum for each substage as bar plots,
    with frequency band annotations and correspondence to true labels.
    
    Args:
        data: (N, T, C) array where N=runs, T=windows, C=channels (window means from raw signals)
        labels: (N, T) array of predicted substage labels (0-indexed)
        true_labels: (N*T,) array of true labels (0=Awake, 1=NREM, 2=REM)
        channel_names: Names of each channel
        sampling_rate: Sampling rate of original signals in Hz
        freq_ranges: List of (min_freq, max_freq) tuples for each channel
        plot_title: Overall plot title
        save_path: Where to save the figure
        figsize: Optional figure size (width, height). Auto-computed if None.
        window_size: Window size for computing FFT
        min_windows: Minimum number of windows required for reliable spectrum estimation
        normalize: If True, normalize PSD by total power (recommended for cross-substage comparison)
    """
    # Input validation
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a numpy array.")
    if not isinstance(labels, np.ndarray):
        raise TypeError("labels must be a numpy array.")
    if data.ndim != 3:
        raise ValueError(f"data must have shape (N,T,C). Got {data.shape}.")
    if labels.ndim != 2:
        raise ValueError(f"labels must have shape (N,T). Got {labels.shape}.")
    
    N, T, C = data.shape
    if labels.shape != (N, T):
        raise ValueError(
            f"labels shape {labels.shape} must match data shape (N,T) = ({N},{T})."
        )
    if len(channel_names) != C:
        raise ValueError(f"channel_names length {len(channel_names)} must equal C={C}.")
    if len(freq_ranges) != C:
        raise ValueError(f"freq_ranges length {len(freq_ranges)} must equal C={C}.")
    
    # Ensure labels are integers
    if not np.issubdtype(labels.dtype, np.integer):
        if np.all(np.isfinite(labels)) and np.all(np.equal(labels, np.round(labels))):
            labels = labels.astype(int)
        else:
            raise ValueError("labels must be integer dtype (or safely castable to integers).")
    
    # Determine number of unique substages
    unique_substages = np.unique(labels)
    n_substages = len(unique_substages)
    
    if labels.min() < 0:
        raise ValueError("labels must be >= 0.")
    
    # Flatten data and labels
    data_flat = data.reshape(-1, C)  # (N*T, C)
    labels_flat = labels.ravel()      # (N*T,)
    
    # Frequency band definitions (for EEG) - matching feature extraction
    bands = {
        'Delta': (0.5, 4),
        'Theta': (6, 9),
        'Alpha': (9, 15),
        'Beta': (15, 30),
    }
    
    # Calculate correspondence between substages and true labels
    substage_to_true = {}
    substage_counts = {}
    for substage in unique_substages:
        mask = labels_flat == substage
        substage_counts[substage] = np.sum(mask)
        if np.any(mask):
            true_label_counts = np.bincount(true_labels[mask].astype(int), minlength=3)
            most_common_true = np.argmax(true_label_counts)
            pct = (true_label_counts[most_common_true] / true_label_counts.sum()) * 100
            substage_to_true[substage] = (most_common_true, pct)
    
    true_label_names = ["Awake", "NREM", "REM"]
    total_points = len(labels_flat)
    
    # Auto-compute figure size
    if figsize is None:
        # 4 columns for regular channels + 1 extra for EMG high-freq
        width = 6 * (C + 1)  # Extra column for EMG high-freq
        height = 3.5 * n_substages
        figsize = (width, height)
    
    # Create subplots grid: rows = substages, cols = channels + 1 (EMG high-freq)
    fig, axes = plt.subplots(nrows=n_substages, ncols=C + 1, figsize=figsize)
    
    # Ensure axes is 2D
    if n_substages == 1 and C + 1 == 1:
        axes = np.array([[axes]])
    elif n_substages == 1:
        axes = axes.reshape(1, -1)
    elif C + 1 == 1:
        axes = axes.reshape(-1, 1)
    
    # Define color palette (same as feature distributions)
    colors = sns.color_palette("husl", n_substages)
    substage_colors = {substage: colors[idx] for idx, substage in enumerate(unique_substages)}
    
    # EMG high-frequency band definition (20-60 Hz for muscle activity detection)
    emg_high_freq_range = (20, 60)
    
    # First pass: compute all spectra to determine per-column y-axis limits
    column_ymax = {}
    
    for col_idx in range(C):
        max_power = 0
        
        for substage in unique_substages:
            mask = labels_flat == substage
            count = substage_counts[substage]
            
            # Skip substages with insufficient data
            if count < min_windows:
                continue
            
            substage_data = data_flat[mask, col_idx]  # (n_points,)
            
            if len(substage_data) >= window_size:
                # Compute FFT
                freqs = np.fft.rfftfreq(window_size, 1.0/sampling_rate)
                
                # Compute power spectra for all windows
                power_spectra = []
                for i in range(0, len(substage_data) - window_size, window_size // 2):
                    window = substage_data[i:i+window_size]
                    window = window * np.hanning(len(window))
                    fft_vals = np.fft.rfft(window)
                    power = np.abs(fft_vals)**2
                    power_spectra.append(power)
                
                if power_spectra:
                    avg_power = np.mean(power_spectra, axis=0)
                    
                    # Normalize by total power in biologically relevant frequency range
                    if normalize:
                        freq_min, freq_max = freq_ranges[col_idx]
                        norm_mask = (freqs >= freq_min) & (freqs <= freq_max)
                        total_power = np.sum(avg_power[norm_mask])
                        if total_power > 0:
                            avg_power = avg_power / total_power
                    
                    # Find max in this frequency range
                    freq_min, freq_max = freq_ranges[col_idx]
                    mask_freq = (freqs >= freq_min) & (freqs <= freq_max)
                    if np.any(mask_freq):
                        # Use 98th percentile to avoid outliers
                        max_power = max(max_power, np.percentile(avg_power[mask_freq], 98))
        
        # Set y-max with 15% padding
        column_ymax[col_idx] = max_power * 1.15 if max_power > 0 else 1.0
    
    # Similar for EMG high-freq column
    column_ymax[C] = 0
    if C > 0 and channel_names[-1].upper() == 'EMG':
        for substage in unique_substages:
            mask = labels_flat == substage
            count = substage_counts[substage]
            
            if count < min_windows:
                continue
            
            substage_data = data_flat[mask, C - 1]  # Last channel is EMG
            
            if len(substage_data) >= window_size:
                freqs = np.fft.rfftfreq(window_size, 1.0/sampling_rate)
                
                power_spectra = []
                for i in range(0, len(substage_data) - window_size, window_size // 2):
                    window = substage_data[i:i+window_size]
                    window = window * np.hanning(len(window))
                    fft_vals = np.fft.rfft(window)
                    power = np.abs(fft_vals)**2
                    power_spectra.append(power)
                
                if power_spectra:
                    avg_power = np.mean(power_spectra, axis=0)
                    
                    # Normalize by total power in EMG high-frequency range (20-60 Hz)
                    if normalize:
                        norm_mask = (freqs >= emg_high_freq_range[0]) & (freqs <= emg_high_freq_range[1])
                        total_power = np.sum(avg_power[norm_mask])
                        if total_power > 0:
                            avg_power = avg_power / total_power
                    
                    mask_freq = (freqs >= emg_high_freq_range[0]) & (freqs <= emg_high_freq_range[1])
                    if np.any(mask_freq):
                        # Use 98th percentile to avoid outliers
                        column_ymax[C] = max(column_ymax[C], np.percentile(avg_power[mask_freq], 98))
        
        column_ymax[C] = column_ymax[C] * 1.15 if column_ymax[C] > 0 else 1.0
    
    # Color for bars
    bar_color = sns.color_palette("deep")[0]
    
    # Plot each substage × channel combination
    for row_idx, substage in enumerate(unique_substages):
        # Get all windows for this substage
        mask = labels_flat == substage
        substage_data = data_flat[mask]  # (n_points, C)
        
        # Get true label correspondence
        true_label, true_pct = substage_to_true.get(substage, (0, 0))
        true_name = true_label_names[true_label]
        count = substage_counts[substage]
        pct_total = (count / total_points) * 100
        
        for col_idx in range(C):
            ax = axes[row_idx, col_idx]
            
            # Check if substage has enough data
            if count < min_windows:
                # Insufficient data - show warning text
                ax.text(
                    0.5, 0.5,
                    f'Insufficient data\n(n={count} < {min_windows})',
                    ha='center', va='center',
                    fontsize=12,
                    color='gray',
                    transform=ax.transAxes
                )
                ax.set_xlim(freq_ranges[col_idx])
                ax.set_ylim(0, column_ymax.get(col_idx, 1))
                ax.grid(True, alpha=0.2)
            else:
                # Extract channel values for this substage
                channel_values = substage_data[:, col_idx]
                
                # Use adaptive window size for small substages
                effective_window_size = min(window_size, len(channel_values))
                
                if effective_window_size >= 32:  # Minimum for meaningful FFT
                    # Zero-pad if needed
                    if len(channel_values) < window_size:
                        padded = np.zeros(window_size)
                        padded[:len(channel_values)] = channel_values
                        channel_values = padded
                    
                    # Compute average power spectrum using overlapping windows
                    n_windows = max(1, len(channel_values) // (window_size // 2) - 1)
                    spectra = []
                    
                    for i in range(max(1, n_windows)):
                        start = i * (window_size // 2)
                        end = start + window_size
                        if end > len(channel_values):
                            break
                        window = channel_values[start:end]
                        
                        # Apply Hann window to reduce spectral leakage
                        window = window * np.hanning(len(window))
                        
                        # Compute FFT
                        fft_vals = np.fft.rfft(window)
                        freqs = np.fft.rfftfreq(len(window), d=1.0/sampling_rate)
                        power = np.abs(fft_vals) ** 2
                        
                        spectra.append(power)
                    
                    if spectra:
                        # Average across windows
                        avg_power = np.mean(spectra, axis=0)
                        
                        # Normalize if requested
                        if normalize:
                            # Get frequency range for normalization
                            min_freq, max_freq = freq_ranges[col_idx]
                            mask_norm = (freqs >= min_freq) & (freqs <= max_freq)
                            total_power = np.sum(avg_power[mask_norm])
                            if total_power > 0:
                                avg_power = avg_power / total_power
                        
                        # Get frequency range for this channel
                        min_freq, max_freq = freq_ranges[col_idx]
                        
                        # Aggregate FFT bins into frequency bins with proper normalization
                        bin_centers = []
                        bin_widths = []
                        power_density = []  # Power per Hz
                        
                        # First bin: 0.5-1 Hz (if min_freq <= 0.5)
                        if min_freq <= 0.5:
                            mask = (freqs >= 0.5) & (freqs < 1.0)
                            if np.any(mask):
                                power = np.mean(avg_power[mask])
                                bin_centers.append(0.75)
                                bin_widths.append(0.5)
                                power_density.append(power / 0.5)  # Normalize by bin width
                        
                        # Remaining bins: 1 Hz wide
                        start_freq = 1.0 if min_freq <= 0.5 else int(np.ceil(min_freq))
                        freq_bins = np.arange(start_freq, int(np.ceil(max_freq)) + 1)
                        
                        for freq_bin in freq_bins:
                            # 1 Hz bins
                            mask = (freqs >= freq_bin) & (freqs < freq_bin + 1)
                            if np.any(mask):
                                power = np.mean(avg_power[mask])
                                bin_centers.append(freq_bin + 0.5)
                                bin_widths.append(1.0)
                                power_density.append(power / 1.0)  # Normalize by bin width
                        
                        bin_centers = np.array(bin_centers)
                        bin_widths = np.array(bin_widths)
                        power_density = np.array(power_density)
                        
                        # Plot as bar chart
                        for center, width, density in zip(bin_centers, bin_widths, power_density):
                            ax.bar(
                                center - width/2,
                                density,
                                width=width,
                                color=substage_colors[substage],
                                edgecolor='none',
                                alpha=0.85,
                                align='edge',
                            )
                
            # Formatting
            ax.set_xlim(freq_ranges[col_idx])
            ax.set_ylim(0, column_ymax.get(col_idx, 1))  # Use column-wide y-axis
            ax.grid(True, alpha=0.3, linewidth=0.5, zorder=1)
            
            # Add frequency band shading for EEG channels (after setting limits)
            if col_idx < 2:  # EEG1, EEG2 only
                x_min, x_max = freq_ranges[col_idx]
                y_min, y_max = ax.get_ylim()
                
                for band_name, (f_min, f_max) in bands.items():
                    # Only show bands that are within the x-axis range
                    if f_min < x_max and f_max > x_min:
                        # Clip to visible range
                        visible_min = max(f_min, x_min)
                        visible_max = min(f_max, x_max)
                        
                        ax.axvspan(visible_min, visible_max, alpha=0.1, color='gray', zorder=0)
                        
                        # Only add label on top row and if band center is visible
                        if row_idx == 0:
                            mid_freq = (visible_min + visible_max) / 2
                            if x_min <= mid_freq <= x_max:
                                ax.text(mid_freq, y_max * 0.95, band_name, 
                                       ha='center', va='top', fontsize=8, alpha=0.6, zorder=10)
            
            # X-axis label only on bottom row
            if row_idx == n_substages - 1:
                ax.set_xlabel('Frequency (Hz)', fontsize=10, fontweight='bold')
            
            # Column titles only on top row
            if row_idx == 0:
                ax.set_title(channel_names[col_idx], fontsize=12, fontweight='bold')
            
            # Row labels on leftmost column
            if col_idx == 0:
                y_label_type = "Relative Power" if normalize else "PSD (μV²/Hz)"
                ylabel = f'Substage {substage}\n{true_name} {true_pct:.0f}%\nn={count} ({pct_total:.1f}%)\n{y_label_type}'
                ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
        
        # EMG High-frequency column (if last channel is EMG)
        if C > 0 and channel_names[-1].upper() == 'EMG':
            ax = axes[row_idx, C]
            
            if count < min_windows:
                # Insufficient data
                ax.text(
                    0.5, 0.5,
                    f'Insufficient data\n(n={count} < {min_windows})',
                    ha='center', va='center',
                    fontsize=12,
                    color='gray',
                    transform=ax.transAxes
                )
                ax.set_xlim(emg_high_freq_range)
                ax.set_ylim(0, column_ymax.get(C, 1))
                ax.grid(True, alpha=0.2)
            else:
                # Extract EMG channel (last channel)
                channel_values = substage_data[:, C - 1]
                
                effective_window_size = min(window_size, len(channel_values))
                
                if effective_window_size >= 32:
                    # Zero-pad if needed
                    if len(channel_values) < window_size:
                        padded = np.zeros(window_size)
                        padded[:len(channel_values)] = channel_values
                        channel_values = padded
                    
                    # Compute average power spectrum
                    n_windows = max(1, len(channel_values) // (window_size // 2) - 1)
                    spectra = []
                    
                    for i in range(max(1, n_windows)):
                        start = i * (window_size // 2)
                        end = start + window_size
                        if end > len(channel_values):
                            break
                        window = channel_values[start:end]
                        window = window * np.hanning(len(window))
                        
                        fft_vals = np.fft.rfft(window)
                        freqs = np.fft.rfftfreq(len(window), d=1.0/sampling_rate)
                        power = np.abs(fft_vals) ** 2
                        spectra.append(power)
                    
                    if spectra:
                        avg_power = np.mean(spectra, axis=0)
                        
                        # Normalize by total power in EMG high-frequency range (20-60 Hz)
                        if normalize:
                            freq_min_norm, freq_max_norm = emg_high_freq_range
                            mask_norm = (freqs >= freq_min_norm) & (freqs <= freq_max_norm)
                            total_power = np.sum(avg_power[mask_norm])
                            if total_power > 0:
                                avg_power = avg_power / total_power
                        
                        # Bin frequencies: 1 Hz bins for high-frequency range
                        bin_centers = []
                        bin_widths = []
                        power_density = []
                        
                        freq_min, freq_max = emg_high_freq_range
                        
                        # 1 Hz bins starting from freq_min
                        for f_low in np.arange(freq_min, freq_max, 1):
                            f_high = f_low + 1
                            mask = (freqs >= f_low) & (freqs < f_high)
                            
                            if np.any(mask):
                                power = np.mean(avg_power[mask])
                                bin_centers.append((f_low + f_high) / 2)
                                bin_widths.append(1.0)
                                power_density.append(power / 1.0)
                        
                        # Plot bars
                        for center, width, density in zip(bin_centers, bin_widths, power_density):
                            ax.bar(
                                center - width/2,
                                density,
                                width=width,
                                color=substage_colors[substage],
                                edgecolor='none',
                                alpha=0.85,
                                align='edge',
                            )
            
            # Formatting
            ax.set_xlim(emg_high_freq_range)
            ax.set_ylim(0, column_ymax.get(C, 1))
            ax.grid(True, alpha=0.3, linewidth=0.5, zorder=1)
            
            # X-axis label only on bottom row
            if row_idx == n_substages - 1:
                ax.set_xlabel('Frequency (Hz)', fontsize=10, fontweight='bold')
            
            # Column title only on top row
            if row_idx == 0:
                ax.set_title('EMG (High-Freq)', fontsize=12, fontweight='bold')
    
    # Add overall title
    fig.suptitle(plot_title, fontsize=14, fontweight='bold', y=0.995)
    
    # Adjust layout
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_substage_feature_distributions(
    data: np.ndarray,                       # (N, T, F) - transformed features
    labels: np.ndarray,                     # (N, T) - predicted substage labels
    true_labels: np.ndarray,                # (N*T,) - true labels (Awake/NREM/REM)
    feature_names: Sequence[str],           # feature names
    plot_title: str,
    save_path: str,
    *,
    figsize: Optional[Tuple[float, float]] = None,
    n_bins: int = 30,
) -> None:
    """
    Creates violin plots showing feature distributions per substage.
    
    Shows distribution of each feature across substages with true label correspondence.
    
    Args:
        data: (N, T, F) array where N=runs, T=windows, F=features
        labels: (N, T) array of predicted substage labels (0-indexed)
        true_labels: (N*T,) array of true labels (0=Awake, 1=NREM, 2=REM)
        feature_names: Names of each feature
        plot_title: Overall plot title
        save_path: Where to save the figure
        figsize: Optional figure size (width, height). Auto-computed if None.
        n_bins: Number of bins for distributions
    """
    # Input validation
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a numpy array.")
    if not isinstance(labels, np.ndarray):
        raise TypeError("labels must be a numpy array.")
    if data.ndim != 3:
        raise ValueError(f"data must have shape (N,T,F). Got {data.shape}.")
    if labels.ndim != 2:
        raise ValueError(f"labels must have shape (N,T). Got {labels.shape}.")
    
    N, T, F = data.shape
    if labels.shape != (N, T):
        raise ValueError(
            f"labels shape {labels.shape} must match data shape (N,T) = ({N},{T})."
        )
    if len(feature_names) != F:
        raise ValueError(f"feature_names length {len(feature_names)} must equal F={F}.")
    
    # Ensure labels are integers
    if not np.issubdtype(labels.dtype, np.integer):
        if np.all(np.isfinite(labels)) and np.all(np.equal(labels, np.round(labels))):
            labels = labels.astype(int)
        else:
            raise ValueError("labels must be integer dtype (or safely castable to integers).")
    
    # Determine number of unique substages
    unique_substages = np.unique(labels)
    n_substages = len(unique_substages)
    
    if labels.min() < 0:
        raise ValueError("labels must be >= 0.")
    
    # Flatten data and labels
    data_flat = data.reshape(-1, F)  # (N*T, F)
    labels_flat = labels.ravel()      # (N*T,)
    
    # Calculate true label correspondence for each substage
    true_label_names = ["Awake", "NREM", "REM"]
    substage_info = {}
    
    for substage in unique_substages:
        mask = labels_flat == substage
        count = np.sum(mask)
        
        # Get percentages for all three true labels
        true_label_counts = np.bincount(true_labels[mask].astype(int), minlength=3)
        percentages = (true_label_counts / true_label_counts.sum()) * 100
        
        substage_info[substage] = {
            'count': count,
            'awake_pct': percentages[0],
            'nrem_pct': percentages[1],
            'rem_pct': percentages[2],
        }
    
    # Auto-compute figure size
    if figsize is None:
        width = 12
        height = 2.5 * F
        figsize = (width, height)
    
    # Create subplots grid: rows = features, cols = 1
    fig, axes = plt.subplots(nrows=F, ncols=1, figsize=figsize)
    
    if F == 1:
        axes = [axes]
    
    # Color palette for substages
    colors = sns.color_palette("husl", n_substages)
    
    # Plot each feature
    for feat_idx in range(F):
        ax = axes[feat_idx]
        
        # Prepare data for violin plot
        plot_data = []
        positions = []
        plot_colors = []
        labels_list = []
        
        for substage_idx, substage in enumerate(unique_substages):
            mask = labels_flat == substage
            feature_values = data_flat[mask, feat_idx]
            
            if len(feature_values) > 0:
                plot_data.append(feature_values)
                positions.append(substage_idx)
                plot_colors.append(colors[substage_idx])
                
                # Create label with true label percentages
                info = substage_info[substage]
                label = (f"S{substage}\n"
                        f"A:{info['awake_pct']:.0f}% "
                        f"N:{info['nrem_pct']:.0f}% "
                        f"R:{info['rem_pct']:.0f}%\n"
                        f"n={info['count']}")
                labels_list.append(label)
        
        # Create violin plot
        parts = ax.violinplot(plot_data, positions=positions, widths=0.7,
                             showmeans=True, showmedians=True)
        
        # Color the violins
        for pc, color in zip(parts['bodies'], plot_colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        # Set x-axis labels
        ax.set_xticks(positions)
        ax.set_xticklabels(labels_list, fontsize=9)
        
        # Set y-axis label
        ax.set_ylabel(feature_names[feat_idx], fontsize=10, fontweight='bold')
        
        # Add grid
        ax.grid(True, alpha=0.3, linewidth=0.5, axis='y')
        
        # Only show x-label on bottom plot
        if feat_idx == F - 1:
            ax.set_xlabel('Substage (A=Awake %, N=NREM %, R=REM %)', fontsize=10, fontweight='bold')
    
    # Add overall title
    fig.suptitle(plot_title, fontsize=14, fontweight='bold', y=0.995)
    
    # Adjust layout
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
