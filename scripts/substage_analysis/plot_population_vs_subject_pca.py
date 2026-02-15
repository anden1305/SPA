"""
Plot PCA visualization of expert-driven feature latent space.

Creates a side-by-side comparison:
- Left: Population-level PCA with human labels (all subjects)
- Right: Subject-specific PCA with human labels (subject 39 only)
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Add repo root to path
repo_root = Path(__file__).parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Set seaborn style
sns.set_theme(style="whitegrid", context="paper")


def compute_pca(
    datapoints: np.ndarray,
    center: bool = True,
    standardize: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute PCA using SVD.
    
    Args:
        datapoints: (N, F) array of data points
        center: Whether to center the data
        standardize: Whether to standardize features
        
    Returns:
        scores: (N, F) PCA scores
        explained_variance_ratio: (F,) variance explained by each component
        V: (F, F) principal components matrix
    """
    X = datapoints.astype(np.float64, copy=False)
    N, F = X.shape
    
    # Center / standardize
    mu = X.mean(axis=0) if center else np.zeros(F, dtype=np.float64)
    Xc = X - mu
    
    if standardize:
        sigma = Xc.std(axis=0, ddof=1)
        sigma[sigma == 0.0] = 1.0
        Xc = Xc / sigma
    
    # SVD: Xc = U S Vt
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt.T  # (F, F)
    
    # Explained variance ratio
    if N > 1:
        eigvals = (S ** 2) / (N - 1)
        explained_variance_ratio = eigvals / eigvals.sum()
    else:
        explained_variance_ratio = np.zeros_like(S)
    
    # PCA scores
    scores = Xc @ V
    
    return scores, explained_variance_ratio, V


def plot_pca_scatter(
    ax: plt.Axes,
    pc1: np.ndarray,
    pc2: np.ndarray,
    labels: np.ndarray,
    label_names: list,
    evr1: float,
    evr2: float,
    title: str,
    point_size: float = 18.0,
    alpha: float = 0.75,
) -> None:
    """
    Plot PCA scatter on a given axis.
    
    Args:
        ax: Matplotlib axis to plot on
        pc1: PC1 scores
        pc2: PC2 scores
        labels: Class labels (0-indexed)
        label_names: Names for each class
        evr1: Explained variance ratio for PC1
        evr2: Explained variance ratio for PC2
        title: Plot title
        point_size: Point size for scatter
        alpha: Transparency
    """
    C = len(label_names)
    
    # Use discrete colormap
    cmap = plt.get_cmap("tab10" if C <= 10 else "tab20")
    colors = [cmap(i % cmap.N) for i in range(C)]
    
    # Plot per class to get clean legend entries
    for c in range(C):
        mask = (labels == c)
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
    
    ax.set_xlabel(f"PC1 ({evr1*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({evr2*100:.1f}% var)")
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc="best", frameon=True)
    ax.grid(True, linewidth=0.5, alpha=0.35)


def plot_population_vs_subject_pca(
    population_data: np.ndarray,
    population_labels: np.ndarray,
    subject_data: np.ndarray,
    subject_labels: np.ndarray,
    label_names: list,
    subject_id: str,
    save_path_population: str,
    save_path_subject: str,
    *,
    n_samples_population: Optional[int] = None,
    n_samples_subject: Optional[int] = None,
    seed: int = 42,
    figsize: Tuple[float, float] = (9, 7),
    point_size: float = 18.0,
    alpha: float = 0.75,
    center: bool = True,
    standardize: bool = False,
) -> None:
    """
    Create two separate PCA plots: population-level and subject-specific.
    
    Args:
        population_data: (N_pop, F) population-level feature data
        population_labels: (N_pop,) true labels for population
        subject_data: (N_sub, F) subject-specific feature data
        subject_labels: (N_sub,) true labels for subject
        label_names: Names for each label class
        subject_id: Subject identifier (e.g., "39")
        save_path_population: Path to save the population plot
        save_path_subject: Path to save the subject plot
        n_samples_population: Number of samples to plot for population (None = all)
        n_samples_subject: Number of samples to plot for subject (None = all)
        seed: Random seed for sampling
        figsize: Figure size for each plot (width, height)
        point_size: Point size for scatter
        alpha: Point transparency
        center: Whether to center data before PCA
        standardize: Whether to standardize data before PCA
    """
    # Input validation
    if not isinstance(population_data, np.ndarray):
        raise TypeError("population_data must be a numpy array.")
    if not isinstance(population_labels, np.ndarray):
        raise TypeError("population_labels must be a numpy array.")
    if not isinstance(subject_data, np.ndarray):
        raise TypeError("subject_data must be a numpy array.")
    if not isinstance(subject_labels, np.ndarray):
        raise TypeError("subject_labels must be a numpy array.")
    
    if population_data.ndim != 2:
        raise ValueError(f"population_data must have shape (N,F). Got {population_data.shape}.")
    if subject_data.ndim != 2:
        raise ValueError(f"subject_data must have shape (N,F). Got {subject_data.shape}.")
    if population_labels.ndim != 1:
        raise ValueError(f"population_labels must have shape (N,). Got {population_labels.shape}.")
    if subject_labels.ndim != 1:
        raise ValueError(f"subject_labels must have shape (N,). Got {subject_labels.shape}.")
    
    N_pop, F_pop = population_data.shape
    N_sub, F_sub = subject_data.shape
    
    if population_labels.shape[0] != N_pop:
        raise ValueError(f"population_labels length must equal N_pop={N_pop}. Got {population_labels.shape[0]}.")
    if subject_labels.shape[0] != N_sub:
        raise ValueError(f"subject_labels length must equal N_sub={N_sub}. Got {subject_labels.shape[0]}.")
    if F_pop != F_sub:
        raise ValueError(f"Feature dimensions must match. Got {F_pop} vs {F_sub}.")
    
    print(f"\n{'='*80}")
    print(f"PCA VISUALIZATION: POPULATION vs SUBJECT {subject_id}")
    print(f"{'='*80}")
    print(f"Population data: {population_data.shape}")
    print(f"Subject {subject_id} data: {subject_data.shape}")
    print(f"Feature dimension: {F_pop}")
    print(f"{'='*80}\n")
    
    # Compute PCA on population data
    print("🔄 Computing PCA on population data...")
    pop_scores, pop_evr, _ = compute_pca(population_data, center=center, standardize=standardize)
    
    # Compute PCA on subject data
    print(f"🔄 Computing PCA on subject {subject_id} data...")
    sub_scores, sub_evr, _ = compute_pca(subject_data, center=center, standardize=standardize)
    
    # Sample data if requested
    rng = np.random.default_rng(seed)
    
    if n_samples_population is not None and n_samples_population < N_pop:
        print(f"🔄 Sampling {n_samples_population} points from population...")
        pop_idx = rng.choice(N_pop, size=n_samples_population, replace=False)
        pop_pc1 = pop_scores[pop_idx, 0]
        pop_pc2 = pop_scores[pop_idx, 1]
        pop_labels_plot = population_labels[pop_idx]
    else:
        pop_pc1 = pop_scores[:, 0]
        pop_pc2 = pop_scores[:, 1]
        pop_labels_plot = population_labels
    
    if n_samples_subject is not None and n_samples_subject < N_sub:
        print(f"🔄 Sampling {n_samples_subject} points from subject {subject_id}...")
        sub_idx = rng.choice(N_sub, size=n_samples_subject, replace=False)
        sub_pc1 = sub_scores[sub_idx, 0]
        sub_pc2 = sub_scores[sub_idx, 1]
        sub_labels_plot = subject_labels[sub_idx]
    else:
        sub_pc1 = sub_scores[:, 0]
        sub_pc2 = sub_scores[:, 1]
        sub_labels_plot = subject_labels
    
    # Create population plot
    print("🔄 Creating population plot...")
    fig_pop, ax_pop = plt.subplots(figsize=figsize)
    
    plot_pca_scatter(
        ax=ax_pop,
        pc1=pop_pc1,
        pc2=pop_pc2,
        labels=pop_labels_plot,
        label_names=label_names,
        evr1=pop_evr[0] if pop_evr.size > 0 else 0.0,
        evr2=pop_evr[1] if pop_evr.size > 1 else 0.0,
        title="PCA Visualization of Expert-Driven Feature Latent Space\nPopulation Level (All Subjects)",
        point_size=point_size,
        alpha=alpha,
    )
    
    # Save population plot
    print(f"💾 Saving population plot to {save_path_population}...")
    fig_pop.tight_layout()
    fig_pop.savefig(save_path_population, dpi=300, bbox_inches='tight')
    plt.close(fig_pop)
    
    # Create subject plot
    print(f"🔄 Creating subject {subject_id} plot...")
    fig_sub, ax_sub = plt.subplots(figsize=figsize)
    
    plot_pca_scatter(
        ax=ax_sub,
        pc1=sub_pc1,
        pc2=sub_pc2,
        labels=sub_labels_plot,
        label_names=label_names,
        evr1=sub_evr[0] if sub_evr.size > 0 else 0.0,
        evr2=sub_evr[1] if sub_evr.size > 1 else 0.0,
        title=f"PCA Visualization of Expert-Driven Feature Latent Space\nSubject {subject_id}",
        point_size=point_size,
        alpha=alpha,
    )
    
    # Save subject plot
    print(f"💾 Saving subject plot to {save_path_subject}...")
    fig_sub.tight_layout()
    fig_sub.savefig(save_path_subject, dpi=300, bbox_inches='tight')
    plt.close(fig_sub)
    
    print(f"\n{'='*80}")
    print(f"✅ PLOTS SAVED SUCCESSFULLY!")
    print(f"  Population: {save_path_population}")
    print(f"  Subject {subject_id}: {save_path_subject}")
    print(f"{'='*80}\n")


def load_data_from_result_dir(
    result_dir: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load feature data and labels directly from result directory.
    Caches data as NPZ for faster subsequent loads.
    
    Args:
        result_dir: Path to result directory containing config.json
        
    Returns:
        data: (N*T, F) feature array
        labels: (N*T,) label array
    """
    import json
    import torch
    from pathlib import Path
    
    # Add repo root to path if needed
    repo_root = Path(__file__).parent.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    
    from src.config.config import GlobalConfig
    from src.data.data_loader_collection import DataLoaderCollection
    
    result_dir = Path(result_dir)
    
    # Check for cached NPZ file
    cache_path = result_dir / "latent_space_cache.npz"
    
    try:
        if cache_path.exists():
            print(f"  📦 Loading from cache: {cache_path}")
            data = np.load(cache_path)
            x_flat = data["x_latent"]
            y_flat = data["y_true"]
            print(f"  ✓ Loaded from cache: {x_flat.shape}, {y_flat.shape}")
            return x_flat, y_flat
    except Exception as e:
        print(f"  ⚠️  Cache load failed: {e}")
        print(f"  🔄 Loading from source...")
    
    # Load config
    config_path = result_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json found at {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    config = GlobalConfig.model_validate(config_dict)
    
    # Load validation datasets
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
    
    # Get all validation data (transformed features)
    x_val, y_val = data_loader.get_all_data()
    
    # Handle MARHMM burn-in if needed
    if config.model.type == 'marhmm':
        lags = config.model.params.get('lags', [])
        if lags:
            burn_in = lags[-1]
            x_val = x_val[:, burn_in:]
            y_val = y_val[:, burn_in:]
    
    # Convert to numpy and flatten
    x_np = x_val.cpu().numpy()
    y_np = y_val.cpu().numpy()
    
    # Flatten to (N*T, F)
    if x_np.ndim == 3:
        N, T, F = x_np.shape
        x_flat = x_np.reshape(N * T, F)
    else:
        x_flat = x_np
    
    y_flat = y_np.flatten()
    
    # Try to cache for future use
    try:
        print(f"  💾 Caching latent space to: {cache_path}")
        np.savez(cache_path, x_latent=x_flat, y_true=y_flat)
        print(f"  ✓ Cache saved successfully")
    except Exception as e:
        print(f"  ⚠️  Cache save failed (continuing anyway): {e}")
    
    return x_flat, y_flat


def load_data_from_npz(
    npz_path: str,
    subject_filter: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load feature data and labels from NPZ file.
    
    Args:
        npz_path: Path to NPZ file
        subject_filter: Optional subject ID to filter by (e.g., "39")
        
    Returns:
        data: (N, F) feature array
        labels: (N,) label array
    """
    data = np.load(npz_path)
    
    # Load latent features and true labels
    x_latent = data["x_latent"]  # (N_runs, T_windows, F_features)
    y_true = data["y_true"]       # (N_runs * T_windows,)
    
    # Flatten to (N*T, F)
    N, T, F = x_latent.shape
    x_flat = x_latent.reshape(N * T, F)
    
    # Check if subject information is available
    if "subjects" in data and subject_filter is not None:
        subjects = data["subjects"]  # Should be (N,) or (N*T,)
        
        # Ensure subjects array matches flattened data
        if subjects.shape[0] == N:
            # Repeat for each window
            subjects_flat = np.repeat(subjects, T)
        elif subjects.shape[0] == N * T:
            subjects_flat = subjects
        else:
            raise ValueError(f"Unexpected subjects shape: {subjects.shape}")
        
        # Filter by subject
        mask = subjects_flat == subject_filter
        x_flat = x_flat[mask]
        y_true = y_true[mask]
    
    return x_flat, y_true


def main():
    parser = argparse.ArgumentParser(
        description="Plot population vs subject PCA visualization"
    )
    parser.add_argument(
        "population_path",
        type=str,
        help="Path to population-level result directory or NPZ file (all subjects)",
    )
    parser.add_argument(
        "--subject_path",
        type=str,
        default=None,
        help="Path to subject-specific result directory or NPZ file",
    )
    parser.add_argument(
        "--subject_id",
        type=str,
        default="39",
        help="Subject ID to visualize (default: 39)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the plot (default: auto-generated)",
    )
    parser.add_argument(
        "--n_samples_population",
        type=int,
        default=7500,
        help="Number of samples to plot for population (default: 7500)",
    )
    parser.add_argument(
        "--n_samples_subject",
        type=int,
        default=None,
        help="Number of samples to plot for subject (default: all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=124,
        help="Random seed for sampling (default: 124)",
    )
    
    args = parser.parse_args()
    
    # Load population data (from result dir or NPZ)
    print(f"\n📂 Loading population data from: {args.population_path}")
    if args.population_path.endswith('.npz'):
        population_data, population_labels = load_data_from_npz(args.population_path)
    else:
        population_data, population_labels = load_data_from_result_dir(args.population_path)
    
    # Load subject data
    if args.subject_path is not None:
        print(f"📂 Loading subject {args.subject_id} data from: {args.subject_path}")
        if args.subject_path.endswith('.npz'):
            subject_data, subject_labels = load_data_from_npz(args.subject_path)
        else:
            subject_data, subject_labels = load_data_from_result_dir(args.subject_path)
    else:
        raise ValueError("--subject_path is required")
    
    # Determine output paths
    if args.output is None:
        # Strip .npz if present
        base_path = args.population_path.replace(".npz", "")
        output_path_population = f"{base_path}_pca_population.png"
        output_path_subject = f"{base_path}_pca_subject{args.subject_id}.png"
    else:
        # If custom output specified, create two variations
        output_path_population = args.output.replace(".png", "_population.png")
        output_path_subject = args.output.replace(".png", f"_subject{args.subject_id}.png")
    
    # Label names
    label_names = ["Awake", "NREM", "REM"]
    
    # Create plots
    plot_population_vs_subject_pca(
        population_data=population_data,
        population_labels=population_labels,
        subject_data=subject_data,
        subject_labels=subject_labels,
        label_names=label_names,
        subject_id=args.subject_id,
        save_path_population=output_path_population,
        save_path_subject=output_path_subject,
        n_samples_population=args.n_samples_population,
        n_samples_subject=args.n_samples_subject,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
