"""Export trained model predictions to NPZ format for substage analysis.

This script loads a trained HMM/MARHMM model and exports predictions on validation
data in the format expected by the substage analysis scripts.

Usage:
    python scripts/substage_analysis/export_predictions_to_npz.py <result_dir>
    
Example:
    python scripts/substage_analysis/export_predictions_to_npz.py results/substages_analysis/hmm/substages_hmm_mssv_features_5_[timestamp]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Add repo root to path
repo_root = Path(__file__).parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.config.config import GlobalConfig
from src.data.data_loader_collection import DataLoaderCollection


def export_predictions(result_dir: Path, output_name: str = "results.npz"):
    """Export predictions from a trained model to NPZ format.
    
    Args:
        result_dir: Directory containing trained model (e.g., results/.../run_name_[timestamp])
        output_name: Name of output NPZ file (default: results.npz)
    """
    result_dir = Path(result_dir)
    
    print(f"\n{'='*80}")
    print(f"EXPORTING PREDICTIONS TO NPZ")
    print(f"{'='*80}")
    print(f"Result directory: {result_dir}")
    print(f"{'='*80}\n")
    
    # Find run subdirectory (1/ or 0/)
    run_dir = None
    for run_num in ["1", "0"]:
        potential_run_dir = result_dir / run_num
        if potential_run_dir.exists() and (potential_run_dir / "predictions.json").exists():
            run_dir = potential_run_dir
            break
    
    if run_dir is None:
        raise FileNotFoundError(f"No run directory with predictions found in {result_dir}")
    
    print(f"✓ Found run directory: {run_dir}")
    
    # Load config
    config_path = result_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json found at {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    config = GlobalConfig.model_validate(config_dict)
    print(f"✓ Loaded config: {config.model.type}, n_states={config.model.n_states}")
    
    # Check if MARHMM (has burn-in)
    has_burn_in = config.model.type.lower() == "marhmm"
    
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
        raise ValueError(f"No valid NMI values found in validations")
    
    print(f"✓ Best epoch: {best_epoch} (NMI={max_nmi:.4f})")
    
    # Load predictions from best epoch
    predictions_path = run_dir / "predictions.json"
    with open(predictions_path, 'r') as f:
        all_predictions = json.load(f)
    
    y_hat = np.array(all_predictions[str(best_epoch)], dtype=np.int64)
    print(f"✓ Loaded predictions: shape={y_hat.shape}")
    
    # Load validation data
    print("\n🔄 Loading validation data...")
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
    
    # Get all validation data (transformed)
    x_val, y_val = data_loader.get_all_data()
    x_val_np = x_val.cpu().numpy()
    y_val_np = y_val.cpu().numpy()
    
    print(f"✓ Loaded validation data (transformed):")
    print(f"  Features shape (before reshape): {x_val_np.shape}")
    print(f"  Labels shape (before reshape): {y_val_np.shape}")
    
    # Get raw channel data (before transforms)
    print("\n🔄 Loading raw channel data...")
    # We need to window the raw data to match the labels
    # The DataLoader uses window_size and stride to create windows
    window_size = config.dataloader.window_size
    stride = config.dataloader.stride
    
    raw_data_list = []
    for dataset in val_datasets:
        # Access the raw data directly from the dataset's data attribute
        # MSSV dataset stores data as (C, T) where C=channels, T=timesteps
        raw_data = dataset.data  # (C, T)
        
        # Convert to numpy if it's a tensor
        if isinstance(raw_data, torch.Tensor):
            raw_data = raw_data.cpu().numpy()
        
        print(f"  Raw dataset shape: {raw_data.shape}")
        
        # Window the raw data to match the transformed data
        C, T_full = raw_data.shape
        windowed_data = []
        
        # Create windows using same logic as DataLoader
        for start_idx in range(0, T_full - window_size + 1, stride):
            end_idx = start_idx + window_size
            window = raw_data[:, start_idx:end_idx]  # (C, window_size)
            # Take mean across window to get one value per channel per window
            window_mean = window.mean(axis=1)  # (C,)
            windowed_data.append(window_mean)
        
        # Stack windows: (num_windows, C)
        windowed_data = np.stack(windowed_data, axis=0)
        raw_data_list.append(windowed_data)
    
    # Concatenate all datasets along time dimension
    x_raw_all = np.concatenate(raw_data_list, axis=0)  # (total_windows, C)
    print(f"✓ Loaded windowed raw channel data: {x_raw_all.shape}")
    
    # Reshape from (batches, batch_size, features) to (N_runs, T, F)
    # For MSSV with 3 runs, we want (3, T_per_run, F)
    n_runs = len(val_datasets)
    batch_size = config.dataloader.validation_batch_size
    n_features = x_val_np.shape[-1]
    
    # Total timesteps per run
    total_batches = x_val_np.shape[0]
    batches_per_run = total_batches // n_runs
    timesteps_per_run = batches_per_run * batch_size
    
    # Reshape to (n_runs, timesteps_per_run, n_features)
    x_latent = x_val_np.reshape(n_runs, timesteps_per_run, n_features)
    y_true = y_val_np.reshape(n_runs, timesteps_per_run).flatten()
    
    # Reshape raw data to match (trim if needed due to windowing differences)
    n_channels = x_raw_all.shape[-1]
    total_windows_needed = n_runs * timesteps_per_run
    total_windows_available = x_raw_all.shape[0]
    
    if total_windows_available != total_windows_needed:
        print(f"\n⚠️  Window count mismatch: have {total_windows_available}, need {total_windows_needed}")
        if total_windows_available > total_windows_needed:
            print(f"  Trimming raw data to match label length")
            x_raw_all = x_raw_all[:total_windows_needed]
        else:
            print(f"  ERROR: Not enough windows! Cannot proceed with raw data.")
            print(f"  Skipping raw data export.")
            x_raw = None
    
    if x_raw_all is not None and x_raw_all.shape[0] == total_windows_needed:
        x_raw = x_raw_all.reshape(n_runs, timesteps_per_run, n_channels)
        print(f"\n✓ Reshaped data:")
        print(f"  x_latent: {x_latent.shape} (n_runs={n_runs}, timesteps_per_run={timesteps_per_run}, features={n_features})")
        print(f"  x_raw: {x_raw.shape} (n_runs={n_runs}, timesteps_per_run={timesteps_per_run}, channels={n_channels})")
    else:
        x_raw = None
        print(f"\n✓ Reshaped data:")
        print(f"  x_latent: {x_latent.shape} (n_runs={n_runs}, timesteps_per_run={timesteps_per_run}, features={n_features})")
        print(f"  x_raw: None (skipped due to mismatch)")
    
    print(f"  y_true: {y_true.shape}")
    print(f"  y_hat: {y_hat.shape}")
    
    # Handle MARHMM burn-in
    if has_burn_in:
        burn_in_length = len(y_true) - len(y_hat)
        if burn_in_length > 0:
            print(f"\n⚠️  MARHMM burn-in detected: trimming {burn_in_length} samples from start")
            # Trim burn-in from features and true labels
            samples_to_trim_per_run = burn_in_length // n_runs
            x_latent = x_latent[:, samples_to_trim_per_run:, :]
            if x_raw is not None:
                x_raw = x_raw[:, samples_to_trim_per_run:, :]
            y_true = y_true[burn_in_length:]
            print(f"✓ After trimming:")
            print(f"  x_latent: {x_latent.shape}")
            if x_raw is not None:
                print(f"  x_raw: {x_raw.shape}")
            print(f"  y_true: {y_true.shape}")
            print(f"  y_hat: {y_hat.shape}")
    
    # Verify shapes match
    assert len(y_true) == len(y_hat), f"Shape mismatch: y_true={len(y_true)}, y_hat={len(y_hat)}"
    
    # Save to NPZ
    output_dir = result_dir
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / output_name
    
    print(f"\n💾 Saving to: {output_path}")
    if x_raw is not None:
        np.savez(output_path, y_hat=y_hat, y_true=y_true, x_latent=x_latent, x_raw=x_raw)
    else:
        np.savez(output_path, y_hat=y_hat, y_true=y_true, x_latent=x_latent)
    
    print(f"\n{'='*80}")
    print(f"✅ EXPORT COMPLETE!")
    print(f"{'='*80}")
    print(f"Output: {output_path}")
    print(f"Contents:")
    print(f"  y_hat: {y_hat.shape} (predictions)")
    print(f"  y_true: {y_true.shape} (true labels)")
    print(f"  x_latent: {x_latent.shape} (transformed features)")
    if x_raw is not None:
        print(f"  x_raw: {x_raw.shape} (raw channel data)")
    else:
        print(f"  x_raw: None (skipped)")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Export model predictions to NPZ format for substage analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python scripts/substage_analysis/export_predictions_to_npz.py \\
      results/substages_analysis/hmm/substages_hmm_mssv_features_5_[timestamp]
"""
    )
    
    parser.add_argument("result_dir", type=str,
                       help="Path to result directory containing trained model")
    parser.add_argument("--output", "-o", type=str, default="results.npz",
                       help="Name of output NPZ file (default: results.npz)")
    
    args = parser.parse_args()
    
    export_predictions(Path(args.result_dir), args.output)


if __name__ == "__main__":
    main()
