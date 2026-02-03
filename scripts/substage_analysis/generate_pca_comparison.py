"""Generate 4-subplot PCA comparison plots for substage analysis.

Creates horizontal figures comparing K-means, HMM, MARHMM, and true label predictions
in MSSV feature PCA space. Generates one figure per n_stages value (5, 6, 7, 8).

Usage:
    python scripts/substage_analysis/generate_pca_comparison.py --n_stages 5
    python scripts/substage_analysis/generate_pca_comparison.py --n_stages 6 7 8
    python scripts/substage_analysis/generate_pca_comparison.py --all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add repo root to path
repo_root = Path(__file__).parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.config.config import GlobalConfig
from src.data.data_loader_collection import DataLoaderCollection
from scripts.substage_analysis.kmeans import kmeans_predict_labels


# Hardcoded mapping of n_stages to result folder timestamps
HMM_FOLDERS = {
    5: "substages_hmm_mssv_features [20260119-175027]",
    6: "substages_hmm_mssv_features [20260119-175151]",
    7: "substages_hmm_mssv_features [20260119-181445]",
    8: "substages_hmm_mssv_features [20260120-002558]",
}

MARHMM_FOLDERS = {
    5: "substages_marhmm_mssv_features [20260119-204040]",
    6: "substages_marhmm_mssv_features [20260119-204051]",
    7: "substages_marhmm_mssv_features [20260119-221258]",
    8: "substages_marhmm_mssv_features [20260119-235829]",
}


def load_predictions_at_max_nmi(run_dir: Path) -> Tuple[np.ndarray, float, int, bool]:
    """Load predictions from epoch where model achieved maximum NMI.
    
    Args:
        run_dir: Path to run directory (e.g., results/.../1/)
        
    Returns:
        Tuple of (predictions, max_nmi, epoch_at_max_nmi, has_burn_in)
    """
    # Check if MARHMM (which has burn-in)
    config_path = run_dir.parent / "config.json"
    if not config_path.exists():
        config_path = run_dir / "config.json"
    
    has_burn_in = False
    if config_path.exists():
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        model_type = config_dict.get("model", {}).get("type", "").lower()
        has_burn_in = model_type == "marhmm"
    
    # Load validations to find epoch with max NMI
    validations_path = run_dir / "validations.json"
    if not validations_path.exists():
        raise FileNotFoundError(f"No validations.json found at {validations_path}")
    
    with open(validations_path, 'r') as f:
        validations = json.load(f)
    
    # Find epoch with maximum NMI
    max_nmi = -1
    best_epoch = None
    for epoch_str, metrics in validations.items():
        nmi = metrics.get('nmi')
        if nmi is not None and nmi > max_nmi:
            max_nmi = nmi
            best_epoch = int(epoch_str)
    
    if best_epoch is None:
        raise ValueError(f"No valid NMI values found in {validations_path}")
    
    # Load predictions from that epoch
    predictions_path = run_dir / "predictions.json"
    if not predictions_path.exists():
        raise FileNotFoundError(f"No predictions.json found at {predictions_path}")
    
    with open(predictions_path, 'r') as f:
        all_predictions = json.load(f)
    
    predictions = all_predictions.get(str(best_epoch))
    if predictions is None:
        raise ValueError(f"No predictions found for epoch {best_epoch}")
    
    return np.array(predictions, dtype=np.int64), max_nmi, best_epoch, has_burn_in


def load_features_and_labels(result_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load MSSV features and true labels from result directory config.
    
    Args:
        result_dir: Path to result directory (e.g., results/substages/hmm/...)
        
    Returns:
        Tuple of (features, true_labels) as numpy arrays of shape (N, F) and (N,)
    """
    config_path = result_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json found at {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    config = GlobalConfig.model_validate(config_dict)
    
    # Build validation datasets from config
    val_datasets = []
    for ds_config in config.val_datasets:
        if ds_config.type == "mssv":
            from src.data.mssv_dataset import MSSVDataset
            val_datasets.append(MSSVDataset(config=ds_config))
        else:
            raise ValueError(f"Unsupported dataset type: {ds_config.type}")
    
    if not val_datasets:
        raise ValueError("No validation datasets found in config")
    
    # Create data loader
    data_loader = DataLoaderCollection(
        datasets=val_datasets,
        config=config,
        for_validation=True,
        device=torch.device('cpu')
    )
    
    # Get all validation data
    x_val, y_val = data_loader.get_all_data()
    x_val_np = x_val.cpu().numpy()
    y_val_np = y_val.cpu().numpy()
    
    # Reshape from (batches, batch_size, features) to (N, features)
    x_val_np = x_val_np.reshape(-1, x_val_np.shape[-1])
    y_val_np = y_val_np.flatten()
    
    return x_val_np, y_val_np


def fit_pca(X: np.ndarray, n_components: int = 2, center: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit PCA using SVD and return transformation matrix.
    
    Args:
        X: Data matrix of shape (N, F)
        n_components: Number of principal components to keep
        center: Whether to center the data
        
    Returns:
        Tuple of (mean, principal_axes, explained_variance_ratio)
        - mean: Feature means, shape (F,)
        - principal_axes: PC directions, shape (F, n_components)
        - explained_variance_ratio: Variance explained by each PC, shape (n_components,)
    """
    X = X.astype(np.float64, copy=False)
    N, F = X.shape
    
    # Center
    mu = X.mean(axis=0) if center else np.zeros(F, dtype=np.float64)
    Xc = X - mu
    
    # SVD: Xc = U S Vt
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt.T  # (F, F)
    
    # Explained variance ratio
    if N > 1:
        eigenvalues = (S ** 2) / (N - 1)
        explained_variance_ratio = eigenvalues / eigenvalues.sum()
    else:
        explained_variance_ratio = np.ones(len(S)) / len(S)
    
    # Keep only n_components
    V_reduced = V[:, :n_components]
    evr_reduced = explained_variance_ratio[:n_components]
    
    return mu, V_reduced, evr_reduced


def transform_pca(X: np.ndarray, mean: np.ndarray, principal_axes: np.ndarray) -> np.ndarray:
    """Transform data using fitted PCA.
    
    Args:
        X: Data matrix of shape (N, F)
        mean: Feature means from fit_pca
        principal_axes: PC directions from fit_pca
        
    Returns:
        Transformed data of shape (N, n_components)
    """
    X = X.astype(np.float64, copy=False)
    Xc = X - mean
    return Xc @ principal_axes


def stratified_sample(labels: np.ndarray, max_per_class: int, seed: int) -> np.ndarray:
    """Stratified sampling: sample up to max_per_class from each label class.
    
    Args:
        labels: Label array of shape (N,)
        max_per_class: Maximum samples per class
        seed: Random seed
        
    Returns:
        Array of sampled indices
    """
    rng = np.random.default_rng(seed)
    unique_labels = np.unique(labels)
    indices = []
    
    for label in unique_labels:
        class_indices = np.where(labels == label)[0]
        n_available = len(class_indices)
        n_to_sample = min(max_per_class, n_available)
        sampled = rng.choice(class_indices, size=n_to_sample, replace=False)
        indices.append(sampled)
    
    indices = np.concatenate(indices)
    return indices


def generate_pca_comparison(
    n_stages: int,
    results_root: Path = Path("results/substages"),
    output_dir: Path = Path("results/substages/pca_comparison"),
    max_per_class: int = 2000,
    seed: int = 42,
) -> None:
    """Generate 4-subplot PCA comparison for given n_stages.
    
    Args:
        n_stages: Number of states/substages to analyze
        results_root: Root directory containing substage results
        output_dir: Directory to save output figures
        max_per_class: Maximum samples per class for visualization
        seed: Random seed for reproducibility
    """
    print(f"\n{'='*80}")
    print(f"Generating PCA comparison for n_stages = {n_stages}")
    print(f"{'='*80}")
    
    # Get result directories
    if n_stages not in HMM_FOLDERS or n_stages not in MARHMM_FOLDERS:
        raise ValueError(f"No results found for n_stages = {n_stages}")
    
    hmm_dir = results_root / "hmm" / HMM_FOLDERS[n_stages]
    marhmm_dir = results_root / "marhmm" / MARHMM_FOLDERS[n_stages]
    
    if not hmm_dir.exists():
        raise FileNotFoundError(f"HMM directory not found: {hmm_dir}")
    if not marhmm_dir.exists():
        raise FileNotFoundError(f"MARHMM directory not found: {marhmm_dir}")
    
    print(f"HMM directory: {hmm_dir}")
    print(f"MARHMM directory: {marhmm_dir}")
    
    # Load features and true labels (from HMM config - both should be identical)
    print("\nLoading MSSV features and true labels...")
    features, true_labels = load_features_and_labels(hmm_dir)
    print(f"Features shape: {features.shape}")
    print(f"True labels shape: {true_labels.shape}")
    print(f"True label distribution: {np.bincount(true_labels)}")
    
    # Load HMM predictions at max NMI (from run 1)
    print("\nLoading HMM predictions at max NMI...")
    hmm_run_dir = hmm_dir / "1"
    if not hmm_run_dir.exists():
        hmm_run_dir = hmm_dir / "0"
    hmm_preds, hmm_nmi, hmm_epoch, hmm_burn_in = load_predictions_at_max_nmi(hmm_run_dir)
    print(f"HMM max NMI: {hmm_nmi:.4f} at epoch {hmm_epoch}")
    print(f"HMM predictions shape: {hmm_preds.shape}")
    print(f"HMM predicted label distribution: {np.bincount(hmm_preds)}")
    
    # Load MARHMM predictions at max NMI (from run 1)
    print("\nLoading MARHMM predictions at max NMI...")
    marhmm_run_dir = marhmm_dir / "1"
    if not marhmm_run_dir.exists():
        marhmm_run_dir = marhmm_dir / "0"
    marhmm_preds, marhmm_nmi, marhmm_epoch, marhmm_burn_in = load_predictions_at_max_nmi(marhmm_run_dir)
    print(f"MARHMM max NMI: {marhmm_nmi:.4f} at epoch {marhmm_epoch}")
    print(f"MARHMM predictions shape: {marhmm_preds.shape}")
    print(f"MARHMM predicted label distribution: {np.bincount(marhmm_preds)}")
    print(f"MARHMM has burn-in: {marhmm_burn_in}")
    
    # Handle MARHMM burn-in trimming
    if marhmm_burn_in:
        burn_in_length = len(true_labels) - len(marhmm_preds)
        print(f"Trimming {burn_in_length} samples from start to match MARHMM burn-in")
        features_trimmed = features[burn_in_length:]
        true_labels_trimmed = true_labels[burn_in_length:]
        hmm_preds_trimmed = hmm_preds[burn_in_length:]
    else:
        features_trimmed = features
        true_labels_trimmed = true_labels
        hmm_preds_trimmed = hmm_preds
    
    # Verify shapes match
    assert features_trimmed.shape[0] == len(true_labels_trimmed)
    assert len(true_labels_trimmed) == len(hmm_preds_trimmed)
    assert len(hmm_preds_trimmed) == len(marhmm_preds)
    
    # Generate K-means labels on ORIGINAL feature space (not PCA!)
    print(f"\nRunning K-means with {n_stages} clusters on original feature space...")
    kmeans_labels = kmeans_predict_labels(
        datapoints=features_trimmed,  # Use full features, not PCA projection
        c=n_stages,
        seed=seed,
        max_iter=300,
        n_init=10,
    )
    print(f"K-means label distribution: {np.bincount(kmeans_labels)}")
    
    # Fit PCA on all features for visualization
    print("\nFitting PCA on all features for visualization...")
    mean, principal_axes, evr = fit_pca(features_trimmed, n_components=2)
    print(f"Explained variance: PC1={evr[0]*100:.1f}%, PC2={evr[1]*100:.1f}%")
    
    # Transform features to PCA space
    pca_features = transform_pca(features_trimmed, mean, principal_axes)
    print(f"PCA features shape: {pca_features.shape}")
    
    # Stratified sampling for visualization (max per class to keep plots readable)
    print(f"\nSampling up to {max_per_class} points per class for visualization...")
    sample_indices = stratified_sample(true_labels_trimmed, max_per_class, seed)
    print(f"Sampled {len(sample_indices)} points total")
    
    # Extract sampled data
    pca_sampled = pca_features[sample_indices]
    kmeans_sampled = kmeans_labels[sample_indices]
    hmm_sampled = hmm_preds_trimmed[sample_indices]
    marhmm_sampled = marhmm_preds[sample_indices]
    true_sampled = true_labels_trimmed[sample_indices]
    
    # Create 4-subplot figure
    print("\nCreating 4-subplot figure...")
    fig, axes = plt.subplots(1, 4, figsize=(36, 7), sharey=True)
    
    # Subplot titles and data
    subplot_data = [
        ("K-means", kmeans_sampled, n_stages, f"K-means (n={n_stages} clusters)"),
        ("HMM", hmm_sampled, n_stages, f"HMM (NMI={hmm_nmi:.3f}, epoch={hmm_epoch})"),
        ("MARHMM", marhmm_sampled, n_stages, f"MARHMM (NMI={marhmm_nmi:.3f}, epoch={marhmm_epoch})"),
        ("True Labels", true_sampled, 3, "True Labels (N=3 states)"),
    ]
    
    # Color maps
    cmap_methods = plt.get_cmap("tab10" if n_stages <= 10 else "tab20")
    cmap_true = plt.get_cmap("tab10")
    
    for idx, (name, labels, n_classes, title) in enumerate(subplot_data):
        ax = axes[idx]
        
        # Choose colormap
        cmap = cmap_true if name == "True Labels" else cmap_methods
        colors = [cmap(i % cmap.N) for i in range(n_classes)]
        
        # Plot each class separately for clean legend
        for c in range(n_classes):
            mask = labels == c
            if np.any(mask):
                label_name = ["Awake", "NREM", "REM"][c] if name == "True Labels" else f"State {c}"
                ax.scatter(
                    pca_sampled[mask, 0],
                    pca_sampled[mask, 1],
                    c=[colors[c]],
                    s=18,
                    alpha=0.75,
                    label=label_name,
                    edgecolors='none',
                )
        
        ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}% var)")
        if idx == 0:
            ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}% var)")
        ax.set_title(title)
        ax.legend(loc='best', frameon=True)
        ax.grid(True, linewidth=0.5, alpha=0.35)
    
    # Save figure
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"pca_comparison_n{n_stages}.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\n✓ Saved figure to: {output_path}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate PCA comparison plots for substage analysis"
    )
    parser.add_argument(
        "--n_stages",
        type=int,
        nargs="+",
        choices=[5, 6, 7, 8],
        help="Number of stages to analyze (5, 6, 7, or 8)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate plots for all n_stages values (5, 6, 7, 8)",
    )
    parser.add_argument(
        "--results_root",
        type=Path,
        default=Path("results/substages"),
        help="Root directory containing substage results",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/substages/pca_comparison"),
        help="Directory to save output figures",
    )
    parser.add_argument(
        "--max_per_class",
        type=int,
        default=2000,
        help="Maximum points per class to plot (default: 2000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    
    args = parser.parse_args()
    
    # Determine which n_stages values to process
    if args.all:
        n_stages_list = [5, 6, 7, 8]
    elif args.n_stages:
        n_stages_list = args.n_stages
    else:
        parser.error("Must specify either --n_stages or --all")
    
    # Generate plots
    for n_stages in n_stages_list:
        try:
            generate_pca_comparison(
                n_stages=n_stages,
                results_root=args.results_root,
                output_dir=args.output_dir,
                max_per_class=args.max_per_class,
                seed=args.seed,
            )
        except Exception as e:
            print(f"\n✗ Error processing n_stages={n_stages}: {e}\n")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n✓ All plots generated successfully!")


if __name__ == "__main__":
    main()
