from scripts.substage_analysis.plot_substages import pca_scatter_random_samples
import numpy as np
from scripts.substage_analysis.kmeans import kmeans_predict_labels
from scripts.substage_analysis.transition_matrix import plot_transition_matrix
from scripts.substage_analysis.frequency_plot import plot_label_channel_frequency_grid
from scripts.substage_analysis.distribution_plot import plot_label_distribution

RESULT_PATH = "results/substages/gmm/substages_gmm_population_6 [20260122-211745]/plots/results.npz"
ANALYSIS_NAME = "PCA Scatter Example"
SEED = 124
N_SAMPLES = 7500
SAVE_PATH = RESULT_PATH

USE_TRUE_LABELS = True  # Whether to use true labels or predicted labels


def run_substage_analysis(config_path: str) -> None:
    
    RESULT_PATH = config_path
    
    # load npz
    data = np.load(RESULT_PATH)
    datapoints = data["x_latent"]
    # flatten dim 0 and 1
    N, T, F = datapoints.shape
    datapoints = datapoints.reshape(N * T, F)
    labels_true = data["y_true"]
    LABEL_NAMES_TRUE = ["Awake", "NREM", "REM"]
    labels_pred = data["y_hat"]
    LABEL_NAMES_PRED = [f"Substage {i+1}" for i in range(np.max(labels_pred) + 1)]
    raw_datapoints = data["x"]
    N, T, C, F = raw_datapoints.shape
    raw_datapoints = raw_datapoints.reshape(N * T, C, F)
    
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
        save_path=RESULT_PATH.replace(".npz", "_pca_scatter_true.png"),
    )
    
    pca_scatter_random_samples(
        datapoints=datapoints,
        labels=labels_pred,
        label_names=LABEL_NAMES_PRED,
        analysis_name=ANALYSIS_NAME,
        seed=SEED,
        n_samples=N_SAMPLES,
        save_path=RESULT_PATH.replace(".npz", "_pca_scatter_predicted.png"),
    )
    
    pca_scatter_random_samples(
        datapoints=datapoints,
        labels=labels_kmeans,
        label_names=LABEL_NAMES_PRED,
        analysis_name=ANALYSIS_NAME,
        seed=SEED,
        n_samples=N_SAMPLES,
        save_path=RESULT_PATH.replace(".npz", "_pca_scatter_kmeans.png"),
    )
    
    plot_transition_matrix(
        y=labels_true,
        label_names=LABEL_NAMES_TRUE,
        plot_name="Transition Matrix - True Labels",
        save_path=RESULT_PATH.replace(".npz", "_transition_matrix_true.png"),
    )
    
    plot_transition_matrix(
        y=labels_pred,
        label_names=LABEL_NAMES_PRED,
        plot_name="Transition Matrix - Predicted Labels",
        save_path=RESULT_PATH.replace(".npz", "_transition_matrix_predicted.png"),
    )
    
    plot_transition_matrix(
        y=labels_kmeans,
        label_names=LABEL_NAMES_PRED,
        plot_name="Transition Matrix - KMeans Labels",
        save_path=RESULT_PATH.replace(".npz", "_transition_matrix_kmeans.png"),
    )
    
    plot_label_channel_frequency_grid(
        raw_datapoints=raw_datapoints,
        sample_rate=128,
        y_pred=labels_pred,
        label_names=LABEL_NAMES_PRED,
        channel_names=["EEG1", "EEG2", "EMG"],
        channel_freq_ranges=[(0, 20), (0, 20), (5, 60)],
        plot_title="Frequency Plot GMM Predicted Substages",
        save_path=RESULT_PATH.replace(".npz", "_frequency_plot_gmm_predicted.png"),
    )
    
    plot_label_distribution(
        y_pred=labels_pred,
        label_names=LABEL_NAMES_PRED,
        plot_title="Label Distribution - GMM Predicted Substages",
        save_path=RESULT_PATH.replace(".npz", "_label_distribution_gmm_predicted.png"),
    )