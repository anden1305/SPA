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
LABEL_COLORS_TRUE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#ff0000"]  # Awake, NREM, REM, Artifact
LABEL_COLORS_PRED = [
    "#6A3D9A",  # deep violet
    "#1B9E77",  # teal green
    "#D95F02",  # burnt orange (distinct from #ff7f0e)
    "#7570B3",  # soft indigo
    "#E7298A",  # magenta
    "#66A61E",  # olive green (not close to true green)
    "#E6AB02",  # mustard yellow
    "#A6761D",  # warm brown
    "#666666",  # neutral dark gray
    "#8DD3C7",  # light turquoise
    "#BEBADA",  # lavender
    "#FB8072",  # soft coral (not pure red)
    "#80B1D3",  # light steel blue (far from true blue)
    "#FDB462",  # peach
    "#B3B3B3",  # light gray (least important)
]

SAMPLING_RATE = 128  # Hz (MSSV dataset sampling rate)
FREQ_RANGES = [(0.0, 30), (0.0, 30), (5, 60)]

USE_TRUE_LABELS = True  # Whether to use true labels or predicted labels


def remap_labels(y: np.ndarray) -> np.ndarray:

    counts = np.bincount(y)
    ranked_old = np.argsort(-counts)
    ranked_old = ranked_old[counts[ranked_old] > 0]

    mapping = np.empty(counts.shape[0], dtype=int)
    mapping[ranked_old] = np.arange(len(ranked_old))

    y_new = mapping[y]
    
    return y_new


def run_substage_analysis(config_path: str) -> None:
    
    RESULT_PATH = config_path
    
    # load npz
    data = np.load(RESULT_PATH)
    datapoints = data["x_latent"]
    # flatten dim 0 and 1
    N, T, F = datapoints.shape
    datapoints = datapoints.reshape(N * T, F)
    labels_true = data["y_true"]
    LABEL_NAMES_TRUE = ["Awake", "NREM", "REM", "Artifact"][: np.max(labels_true) + 1]
    labels_pred = data["y_hat"]
    labels_pred = remap_labels(labels_pred)
    LABEL_NAMES_PRED = [f"Substage {i+1}" for i in range(np.max(labels_pred) + 1)]
    raw_datapoints = data["x"]
    N, T, C, F = raw_datapoints.shape
    raw_datapoints = raw_datapoints.reshape(N * T, C, F)
    sub_ids = data["sub_ids"]
    
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
        label_colors=LABEL_COLORS_TRUE[: len(LABEL_NAMES_TRUE)],
    )
    
    pca_scatter_random_samples(
        datapoints=datapoints,
        labels=labels_pred,
        label_names=LABEL_NAMES_PRED,
        analysis_name=ANALYSIS_NAME,
        seed=SEED,
        n_samples=N_SAMPLES,
        save_path=RESULT_PATH.replace(".npz", "_pca_scatter_predicted.png"),
        label_colors=LABEL_COLORS_PRED[: len(LABEL_NAMES_PRED)],
    )
    
    pca_scatter_random_samples(
        datapoints=datapoints,
        labels=labels_kmeans,
        label_names=LABEL_NAMES_PRED,
        analysis_name=ANALYSIS_NAME,
        seed=SEED,
        n_samples=N_SAMPLES,
        save_path=RESULT_PATH.replace(".npz", "_pca_scatter_kmeans.png"),
        label_colors=LABEL_COLORS_PRED[: len(LABEL_NAMES_PRED)],
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
        csv_path=RESULT_PATH.replace(".npz", "_transition_matrix_predicted.csv"),
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
        sub_ids=sub_ids,
        label_names=LABEL_NAMES_PRED,
        channel_names=["EEG1", "EEG2", "EMG"],
        channel_freq_ranges=FREQ_RANGES,
        plot_title="Features of GMM Predicted Substages",
        save_path=RESULT_PATH.replace(".npz", "_frequency_plot_gmm_predicted.png"),
        input_scale="linear",
        y_scale="linear",
        y_lim_mode="minmax",
        labels_true=labels_true,
        label_names_true=LABEL_NAMES_TRUE,
        label_colors_true=LABEL_COLORS_TRUE[: len(LABEL_NAMES_TRUE)],
        label_colors=LABEL_COLORS_PRED[: len(LABEL_NAMES_PRED)],
    )