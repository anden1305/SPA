"""Generate side-by-side PCA comparison plots for substages sweep results.

This script loads trained models from a substages sweep directory and generates
PCA visualizations comparing true states (N=3) vs predicted states (N=2,3,...,10).

The script uses predictions from the epoch where each model achieved its maximum NMI,
not the final epoch, to show peak performance.

Usage:
    python scripts/generate_substages_pca.py <results_dir> [--n_states 2 3 4 5]
    
Example:
    # Generate PCA for all n_states values found in directory
    python scripts/generate_substages_pca.py results/substages/hmm/mssv
    
    # Generate PCA for specific n_states only
    python scripts/generate_substages_pca.py results/substages/hmm/mssv --n_states 2 5 10
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
import json

# Add repo root to path
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.config.config import GlobalConfig
from src.data.data_loader_collection import DataLoaderCollection
from src.visuals.visualizer import Visualizer
from src.validation.validator import Validator


def load_predictions_at_max_nmi(run_dir: Path):
    """Load predictions from the epoch where model achieved maximum NMI.
    
    Args:
        run_dir: Path to run directory (e.g., results/.../1/)
        
    Returns:
        Tuple of (init_predictions, trained_predictions, max_nmi, epoch_at_max_nmi, has_burn_in)
    """
    # Load config to check if this is MARHMM (which has burn-in)
    config_path = run_dir / "config.json"
    if not config_path.exists():
        config_path = run_dir.parent / "config.json"
    
    has_burn_in = False
    if config_path.exists():
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        model_type = config_dict.get('model', {}).get('type')
        has_burn_in = (model_type == 'marhmm')
    
    # Load validations to find epoch with max NMI
    validations_path = run_dir / "validations.json"
    if not validations_path.exists():
        raise FileNotFoundError(f"Validations not found: {validations_path}")
    
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
        raise ValueError(f"No NMI values found in validations")
    
    # Load predictions from that epoch
    predictions_path = run_dir / "predictions.json"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions not found: {predictions_path}")
    
    with open(predictions_path, 'r') as f:
        all_predictions = json.load(f)
    
    # Get trained predictions (at max NMI)
    trained_predictions = all_predictions.get(str(best_epoch))
    if trained_predictions is None:
        raise ValueError(f"No predictions found for epoch {best_epoch}")
    
    # Get initial predictions (epoch 0 or earliest available)
    init_epoch = 0
    init_predictions = all_predictions.get(str(init_epoch))
    if init_predictions is None:
        # Try to find earliest epoch
        available_epochs = sorted([int(k) for k in all_predictions.keys()])
        if available_epochs:
            init_epoch = available_epochs[0]
            init_predictions = all_predictions[str(init_epoch)]
    
    if init_predictions is None:
        raise ValueError(f"No initial predictions found")
    
    return np.array(init_predictions), np.array(trained_predictions), max_nmi, best_epoch, has_burn_in


def load_model_and_config(run_dir: Path):
    """Load model and config from a run directory.
    
    Handles both structures:
    - New: config.json in parent, model.pth in run_dir (e.g., .../1/)
    - Old: both config.json and model.pth in run_dir (e.g., .../0/)
    
    Args:
        run_dir: Path to run directory (e.g., results/.../1/ or .../0/)
        
    Returns:
        Tuple of (model, config, state_dict)
    """
    # Try config at run_dir level first (old structure)
    config_path = run_dir / "config.json"
    if not config_path.exists():
        # Try parent directory (new structure)
        config_path = run_dir.parent / "config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {run_dir / 'config.json'} or {run_dir.parent / 'config.json'}")
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    config = GlobalConfig.model_validate(config_dict)
    
    # Load model state dict (always in run_dir)
    model_path = run_dir / "model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    state_dict = torch.load(model_path, map_location='cpu')
    
    return config, state_dict


def find_sweep_runs(sweep_dir: Path) -> dict[int, Path]:
    """Find all run directories in sweep results organized by n_states.
    
    The sweep saves each trial in a timestamped directory, with runs inside.
    Handles both structures:
    - New: results/substages/hmm/mssv/substages_hmm_mssv_features [timestamp]/1/
    - Old: results/substages/hmm/substages_hmm_mssv_features/<timestamp>/0/
    
    Args:
        sweep_dir: Root sweep directory
        
    Returns:
        Dict mapping n_states -> run_dir path
    """
    n_states_to_path = {}
    
    # Look for timestamped directories
    if not sweep_dir.exists():
        raise FileNotFoundError(f"Sweep directory not found: {sweep_dir}")
    
    # Scan for all subdirectories
    for entry in sweep_dir.iterdir():
        if not entry.is_dir():
            continue
        
        # Check if config.json exists at this level (new structure)
        config_path = entry / "config.json"
        run_dir = None
        
        if config_path.exists():
            # New structure: config at top level, model in 1/ subdirectory
            # Look for run subdirectory (try 1 first, then 0)
            for run_num in ["1", "0"]:
                potential_run_dir = entry / run_num
                if potential_run_dir.exists() and (potential_run_dir / "model.pth").exists():
                    run_dir = potential_run_dir
                    break
        else:
            # Old structure: <timestamp>/0/ with config inside
            for run_num in ["0", "1"]:
                potential_run_dir = entry / run_num
                potential_config = potential_run_dir / "config.json"
                if potential_config.exists():
                    config_path = potential_config
                    run_dir = potential_run_dir
                    break
        
        if run_dir is None or not config_path.exists():
            continue
        
        # Load config to get n_states
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            n_states = config_dict.get('model', {}).get('n_states')
            if n_states:
                n_states_to_path[n_states] = run_dir
                print(f"Found n_states={n_states} at {run_dir}")
        except Exception as e:
            print(f"Warning: Failed to load config from {config_path}: {e}")
            continue
    
    return n_states_to_path


def generate_pca_plots(sweep_dir: Path, n_states_filter: list[int] = None):
    """Generate PCA comparison plots for substages sweep.
    
    Args:
        sweep_dir: Root directory containing sweep results
        n_states_filter: Optional list of specific n_states values to plot
    """
    print(f"\n{'='*70}")
    print("SUBSTAGES PCA VISUALIZATION GENERATOR")
    print(f"{'='*70}")
    print(f"Sweep Dir: {sweep_dir}")
    print(f"{'='*70}\n")
    
    # Find all sweep runs
    print("📂 Scanning for sweep runs...")
    n_states_to_path = find_sweep_runs(sweep_dir)
    
    if not n_states_to_path:
        print(f"❌ No sweep runs found in {sweep_dir}")
        print("   Expected structure: <sweep_dir>/<timestamp>/0/")
        sys.exit(1)
    
    print(f"✓ Found {len(n_states_to_path)} sweep runs\n")
    
    # Filter if requested
    if n_states_filter:
        n_states_to_path = {k: v for k, v in n_states_to_path.items() if k in n_states_filter}
        print(f"Filtering to n_states={n_states_filter}")
    
    # Load first config to create data loader
    print("🔄 Loading reference config and data...")
    first_run_dir = next(iter(n_states_to_path.values()))
    reference_config, _ = load_model_and_config(first_run_dir)
    
    # Build validation datasets from config
    val_datasets = []
    for ds_config in reference_config.val_datasets:
        if ds_config.type == "synthetic":
            from src.data.synthetic_dataset import SyntheticDataset
            val_datasets.append(SyntheticDataset(config=ds_config))
        elif ds_config.type == "mssv":
            from src.data.mssv_dataset import MSSVDataset
            val_datasets.append(MSSVDataset(config=ds_config))
        else:
            raise ValueError(f"Unknown dataset type: {ds_config.type}")
    
    # Create data loader (shared across all models)
    data_loader = DataLoaderCollection(
        datasets=val_datasets,
        config=reference_config,
        for_validation=True,
        device=torch.device('cpu')
    )
    
    # Get validation data
    x_val, y_val = data_loader.get_all_data()
    x_val_np = x_val.cpu().numpy()
    y_val_np = y_val.cpu().numpy()
    
    # Reshape from (batches, batch_size, features) to (N, features)
    original_shape = x_val_np.shape
    x_val_np = x_val_np.reshape(-1, x_val_np.shape[-1])
    y_val_np = y_val_np.flatten()
    
    print(f"✓ Loaded validation data: {original_shape} → {x_val_np.shape}")
    
    # Load predictions from epoch with max NMI for each n_states
    print("\n🔄 Loading predictions at max NMI epochs...")
    init_predictions_dict = {}
    trained_predictions_dict = {}
    has_burn_in = False
    
    for n_states, run_dir in sorted(n_states_to_path.items()):
        try:
            init_preds, trained_preds, max_nmi, best_epoch, burn_in = load_predictions_at_max_nmi(run_dir)
            init_predictions_dict[n_states] = init_preds
            trained_predictions_dict[n_states] = trained_preds
            has_burn_in = has_burn_in or burn_in
            print(f"✓ n_states={n_states}: max NMI={max_nmi:.4f} at epoch {best_epoch}")
        except Exception as e:
            print(f"⚠️  Failed to load predictions for n_states={n_states}: {e}")
            continue
    
    if not trained_predictions_dict:
        print("❌ No predictions loaded successfully")
        sys.exit(1)
    
    # If MARHMM (has burn-in), trim x_val and y_val to match prediction length
    # All predictions should have same length if from same model type
    if has_burn_in:
        pred_length = len(next(iter(trained_predictions_dict.values())))
        data_length = len(y_val_np)
        burn_in_length = data_length - pred_length
        
        if burn_in_length > 0:
            print(f"\n⚠️  MARHMM detected: trimming first {burn_in_length} samples (burn-in)")
            x_val_np = x_val_np[burn_in_length:]
            y_val_np = y_val_np[burn_in_length:]
            print(f"✓ Adjusted validation data: {x_val_np.shape}")
    
    # Create visualizer (validator not needed for PCA plots, pass None)
    visualizer = Visualizer(data_loader, reference_config, validator=None)
    
    # Generate PCA plots
    output_path = sweep_dir / "pca_plots"
    output_path.mkdir(exist_ok=True)
    
    print(f"\n🎨 Generating tripanel PCA plots (Init → Peak → Truth)...")
    print(f"Output: {output_path}\n")
    
    visualizer._Visualizer__plot_substages_pca_comparison_from_predictions(
        init_predictions_dict=init_predictions_dict,
        trained_predictions_dict=trained_predictions_dict,
        x_val=x_val_np,
        y_true=y_val_np,
        path=output_path,
        n_states_to_show=sorted(trained_predictions_dict.keys())
    )
    
    print(f"\n{'='*70}")
    print("✅ TRIPANEL PCA PLOTS GENERATED SUCCESSFULLY!")
    print(f"{'='*70}")
    print(f"Location: {output_path}")
    print(f"Files: substages_pca_tripanel_n*.png")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate PCA comparison plots for substages sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate PCA for all n_states in sweep
  python scripts/generate_substages_pca.py results/substages/hmm/mssv
  
  # Generate PCA for specific n_states only
  python scripts/generate_substages_pca.py results/substages/hmm/mssv --n_states 2 5 10
"""
    )
    
    parser.add_argument("sweep_dir", type=str,
                       help="Path to sweep results directory")
    parser.add_argument("--n_states", type=int, nargs='+', default=None,
                       help="Specific n_states values to plot (default: all found)")
    
    args = parser.parse_args()
    
    sweep_dir = Path(args.sweep_dir)
    
    generate_pca_plots(sweep_dir, n_states_filter=args.n_states)


if __name__ == "__main__":
    main()
