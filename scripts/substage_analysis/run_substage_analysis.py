import argparse
import sys
from pathlib import Path

import numpy as np

# Add repo root to path
repo_root = Path(__file__).parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts.substage_analysis.plot_substages import (
    pca_scatter_random_samples,
    plot_substage_frequency_distribution,
    plot_substage_frequency_spectrum,
    plot_substage_feature_distributions,
)
from scripts.substage_analysis.kmeans import kmeans_predict_labels
from scripts.substage_analysis.transition_matrix import plot_transition_matrix

ANALYSIS_NAME = "Substage Analysis"
SEED = 124
N_SAMPLES = 7500
N_BINS = 20
CHANNEL_NAMES = ["EEG1", "EEG2", "EMG"]
SAMPLING_RATE = 128  # Hz (MSSV dataset sampling rate)
FREQ_RANGES = [(0.5, 30), (0.5, 30), (1, 60)]  # Hz ranges for EEG1, EEG2, EMG (EMG starts at 1 Hz to exclude low-freq noise)

# Feature names from the transform pipeline
FEATURE_NAMES = [
    "EEG1 RMS",
    "EEG1 Theta/Delta",
    "EEG1 Theta/Beta",
    "EEG1 Beta/Delta",
    "EEG1 Theta-Gamma PAC",
    "EEG1 Delta Power (0.5-4 Hz)",
    "EEG1 Theta Power (6-9 Hz)",
    "EEG1 Beta Power (15-30 Hz)",
    "EEG2 RMS",
    "EEG2 Theta/Delta",
    "EEG2 Theta/Beta",
    "EEG2 Beta/Delta",
    "EEG2 Theta-Gamma PAC",
    "EEG2 Delta Power (0.5-4 Hz)",
    "EEG2 Theta Power (6-9 Hz)",
    "EEG2 Beta Power (15-30 Hz)",
    "EMG RMS",
]

def run_substage_analysis(result_path: str) -> None:
    
    print(f"\n{'='*80}")
    print(f"RUNNING SUBSTAGE ANALYSIS")
    print(f"{'='*80}")
    print(f"Result path: {result_path}")
    print(f"{'='*80}\n")
    
    # Extract model type from path (hmm or marhmm)
    result_path_lower = result_path.lower()
    if "marhmm" in result_path_lower:
        model_type = "MARHMM"
    elif "hmm" in result_path_lower:
        model_type = "HMM"
    else:
        model_type = "Model"  # Fallback
    
    # load npz
    data = np.load(result_path)
    datapoints = data["x_latent"]
    # flatten dim 0 and 1
    N, T, F = datapoints.shape
    datapoints = datapoints.reshape(N * T, F)
    labels_true = data["y_true"]
    LABEL_NAMES_TRUE = ["Awake", "NREM", "REM"]
    labels_pred = data["y_hat"]
    LABEL_NAMES_PRED = [f"Label {i}" for i in range(np.max(labels_pred) + 1)]
    
    # Load raw channel data if available
    x_raw = data.get("x_raw")
    if x_raw is None:
        print("⚠️  Warning: x_raw not found in NPZ file. Skipping frequency distribution plots.")
    else:
        print(f"✓ Loaded raw channel data: {x_raw.shape}")
    
    # Reshape labels for frequency plots
    labels_true_2d = labels_true.reshape(N, T)
    labels_pred_2d = labels_pred.reshape(N, T)
    
    labels_kmeans = kmeans_predict_labels(
        datapoints=datapoints,
        c=len(LABEL_NAMES_PRED),
        seed=SEED,
    )
    
    pca_scatter_random_samples(
        datapoints=datapoints,
        labels=labels_true,
        label_names=LABEL_NAMES_TRUE,
        analysis_name=ANALYSIS_NAME,
        seed=SEED,
        n_samples=N_SAMPLES,
        save_path=result_path.replace(".npz", "_pca_scatter_true.png"),
    )
    
    pca_scatter_random_samples(
        datapoints=datapoints,
        labels=labels_pred,
        label_names=LABEL_NAMES_PRED,
        analysis_name=ANALYSIS_NAME,
        seed=SEED,
        n_samples=N_SAMPLES,
        save_path=result_path.replace(".npz", "_pca_scatter_predicted.png"),
    )
    
    pca_scatter_random_samples(
        datapoints=datapoints,
        labels=labels_kmeans,
        label_names=LABEL_NAMES_PRED,
        analysis_name=ANALYSIS_NAME,
        seed=SEED,
        n_samples=N_SAMPLES,
        save_path=result_path.replace(".npz", "_pca_scatter_kmeans.png"),
    )
    
    plot_transition_matrix(
        y=labels_true,
        label_names=LABEL_NAMES_TRUE,
        plot_name="Transition Matrix - True Labels",
        save_path=result_path.replace(".npz", "_transition_matrix_true.png"),
    )
    
    plot_transition_matrix(
        y=labels_pred,
        label_names=LABEL_NAMES_PRED,
        plot_name="Transition Matrix - Predicted Labels",
        save_path=result_path.replace(".npz", "_transition_matrix_predicted.png"),
    )
    
    plot_transition_matrix(
        y=labels_kmeans,
        label_names=LABEL_NAMES_PRED,
        plot_name="Transition Matrix - KMeans Labels",
        save_path=result_path.replace(".npz", "_transition_matrix_kmeans.png"),
    )
    
    # Feature distribution plots
    print("\n🔄 Generating feature distribution plots...")
    
    # Reshape datapoints back to 3D for plotting
    datapoints_3d = datapoints.reshape(N, T, F)
    
    # True labels feature distributions
    plot_substage_feature_distributions(
        data=datapoints_3d,
        labels=labels_true_2d,
        true_labels=labels_true,
        feature_names=FEATURE_NAMES,
        plot_title="Feature Distributions - True Labels",
        save_path=result_path.replace(".npz", "_feature_dist_true.png"),
    )
    print("  ✓ True labels feature distributions saved")
    
    # Predicted substages feature distributions
    plot_substage_feature_distributions(
        data=datapoints_3d,
        labels=labels_pred_2d,
        true_labels=labels_true,
        feature_names=FEATURE_NAMES,
        plot_title="Feature Distributions - Predicted Substages",
        save_path=result_path.replace(".npz", "_feature_dist_predicted.png"),
    )
    print("  ✓ Predicted substages feature distributions saved")
    
    # KMeans substages feature distributions
    labels_kmeans_2d = labels_kmeans.reshape(N, T)
    plot_substage_feature_distributions(
        data=datapoints_3d,
        labels=labels_kmeans_2d,
        true_labels=labels_true,
        feature_names=FEATURE_NAMES,
        plot_title="Feature Distributions - KMeans Substages",
        save_path=result_path.replace(".npz", "_feature_dist_kmeans.png"),
    )
    print("  ✓ KMeans substages feature distributions saved")
    
    # Frequency spectrum plots (if raw data available)
    if x_raw is not None:
        print("\n🔄 Generating frequency spectrum plots...")
        
        # True labels frequency spectrum
        plot_substage_frequency_spectrum(
            data=x_raw,
            labels=labels_true_2d,
            true_labels=labels_true,
            channel_names=CHANNEL_NAMES,
            sampling_rate=SAMPLING_RATE,
            freq_ranges=FREQ_RANGES,
            plot_title=f"{model_type} - Normalized PSD by Sleep Stage",
            save_path=result_path.replace(".npz", "_freq_spectrum_true.png"),
            normalize=True,
        )
        print("  ✓ True labels frequency spectrum saved")
        
        # Predicted substages frequency spectrum
        plot_substage_frequency_spectrum(
            data=x_raw,
            labels=labels_pred_2d,
            true_labels=labels_true,
            channel_names=CHANNEL_NAMES,
            sampling_rate=SAMPLING_RATE,
            freq_ranges=FREQ_RANGES,
            plot_title=f"{model_type} - Normalized PSD by Predicted Substage",
            save_path=result_path.replace(".npz", "_freq_spectrum_predicted.png"),
            normalize=True,
        )
        print("  ✓ Predicted substages frequency spectrum saved")
        
        # KMeans substages frequency spectrum
        plot_substage_frequency_spectrum(
            data=x_raw,
            labels=labels_kmeans_2d,
            true_labels=labels_true,
            channel_names=CHANNEL_NAMES,
            sampling_rate=SAMPLING_RATE,
            freq_ranges=FREQ_RANGES,
            plot_title=f"{model_type} - Normalized PSD by K-Means Substage",
            save_path=result_path.replace(".npz", "_freq_spectrum_kmeans.png"),
            normalize=True,
        )
        print("  ✓ KMeans substages frequency spectrum saved")
    
    print(f"\n{'='*80}")
    print(f"✅ ANALYSIS COMPLETE!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run substage analysis on results.npz file")
    parser.add_argument("result_path", type=str, help="Path to results.npz file")
    args = parser.parse_args()
    
    run_substage_analysis(args.result_path)