"""Analyze and visualize substages sweep results from W&B.

This script collects results from a completed W&B sweep where n_states 
was varied from 2 to 10, and generates the dual-axis line plot showing
validation log likelihood and NMI convergence.

PCA Visualization:
    The side-by-side PCA plots (true vs predicted states) require loading
    trained models. Since W&B sweeps don't automatically save model checkpoints,
    you need to generate these separately:
    
    1. After sweep completes, the models are saved locally in results/substages/
    2. Use a separate script to load models and generate PCA plots:
       python scripts/generate_substages_pca.py results/substages/hmm/substages_hmm_mssv_features

Usage:
    python scripts/analyze_substages_sweep.py <sweep_id> [--output_path <path>]
    
Example:
    python scripts/analyze_substages_sweep.py abc123def456 --output_path results/substages/analysis
"""

import argparse
import sys
from pathlib import Path
from typing import Dict
import json


def collect_sweep_results(sweep_id: str, project: str = "SPA", entity: str = "dtu_projects") -> tuple[Dict[int, Dict[str, float]], Dict[int, str], str, str]:
    """Collect sweep results from W&B for each n_states value.
    
    Args:
        sweep_id: W&B sweep ID
        project: W&B project name  
        entity: W&B entity/team name
        
    Returns:
        Tuple of:
        - Dict mapping n_states -> {'val_log_likelihood': float, 'nmi': float}
        - Dict mapping n_states -> model_path (for loading models for PCA)
        - Model type (hmm or marhmm)
        - Dataset type (synthetic or mssv)
    """
    try:
        import wandb
    except ImportError:
        print("ERROR: wandb package not installed. Run: pip install wandb")
        sys.exit(1)
    
    print(f"📊 Fetching sweep results from W&B: {entity}/{project}/{sweep_id}")
    
    api = wandb.Api()
    try:
        sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    except Exception as e:
        print(f"ERROR: Failed to fetch sweep: {e}")
        print(f"Make sure you're authenticated (wandb login) and the sweep ID is correct")
        sys.exit(1)
    
    results = {}
    model_paths = {}
    model_type = None
    dataset_type = None
    
    print(f"Found {len(sweep.runs)} runs in sweep")
    
    for run in sweep.runs:
        # Get n_states from config
        # Handle case where config might be a string (JSON) or dict
        config = run.config
        if isinstance(config, str):
            try:
                config = json.loads(config)
            except json.JSONDecodeError:
                print(f"⚠️  Skipping run {run.id} - config is malformed string")
                continue
        
        # W&B wraps config values in {'value': ...} dicts
        # The sweep parameter is stored as 'model.n_states': {'value': X}
        n_states = None
        if "model.n_states" in config:
            n_states_val = config["model.n_states"]
            n_states = n_states_val.get('value') if isinstance(n_states_val, dict) else n_states_val
        elif "model" in config:
            model_val = config["model"]
            if isinstance(model_val, dict):
                model_dict = model_val.get('value', model_val)
                n_states = model_dict.get("n_states") if isinstance(model_dict, dict) else None
            
        if n_states is None:
            print(f"⚠️  Skipping run {run.id} - no n_states in config")
            continue
        
        # Extract model type and dataset type (from first valid run)
        if model_type is None:
            # Get model type
            if "model.type" in config:
                model_type_val = config["model.type"]
                model_type = model_type_val.get('value') if isinstance(model_type_val, dict) else model_type_val
            elif "model" in config:
                model_val = config["model"]
                if isinstance(model_val, dict):
                    model_dict = model_val.get('value', model_val)
                    if isinstance(model_dict, dict):
                        model_type = model_dict.get("type")
            
            # Get dataset type from train_datasets
            if "train_datasets" in config:
                train_ds_val = config["train_datasets"]
                train_ds = train_ds_val.get('value') if isinstance(train_ds_val, dict) else train_ds_val
                if isinstance(train_ds, list) and len(train_ds) > 0:
                    first_ds = train_ds[0]
                    if isinstance(first_ds, dict):
                        ds_type = first_ds.get('type', '')
                        if 'synthetic' in ds_type.lower():
                            dataset_type = 'synthetic'
                        elif 'mssv' in ds_type.lower():
                            dataset_type = 'mssv'
            
            # Fallback: try val_datasets
            if dataset_type is None and "val_datasets" in config:
                val_ds_val = config["val_datasets"]
                val_ds = val_ds_val.get('value') if isinstance(val_ds_val, dict) else val_ds_val
                if isinstance(val_ds, list) and len(val_ds) > 0:
                    first_ds = val_ds[0]
                    if isinstance(first_ds, dict):
                        ds_type = first_ds.get('type', '')
                        if 'synthetic' in ds_type.lower():
                            dataset_type = 'synthetic'
                        elif 'mssv' in ds_type.lower():
                            dataset_type = 'mssv'
        
        # Get best metrics from summary
        # W&B summary can have different access patterns, try multiple approaches
        summary_dict = {}
        
        # Try different ways to access summary
        if hasattr(run.summary, '_json_dict'):
            raw = run.summary._json_dict
            # If it's a string, try to parse it
            if isinstance(raw, str):
                try:
                    summary_dict = json.loads(raw)
                except json.JSONDecodeError:
                    pass
            elif isinstance(raw, dict):
                summary_dict = raw
        
        # Fallback: try direct attribute access for known keys
        if not summary_dict:
            for key in ["val/log_likelihood", "val_log_likelihood", "aggregate/val_log_likelihood",
                       "val/nmi", "nmi", "aggregate/nmi"]:
                try:
                    val = getattr(run.summary, key.replace("/", "_"), None)
                    if val is not None:
                        summary_dict[key] = val
                except:
                    pass
        
        # Try different possible metric names
        # Note: Metrics are logged as val/{key} where key doesn't have val_ prefix
        # Debug: print available summary keys
        print(f"   Summary keys: {list(summary_dict.keys())[:20]}")  # First 20 keys
        
        val_ll = (summary_dict.get("val/log_likelihood") or 
                 summary_dict.get("val/val_log_likelihood") or  # Legacy double-val prefix
                 summary_dict.get("log_likelihood") or
                 summary_dict.get("aggregate/log_likelihood"))
        
        # Get max NMI from run history (across all epochs)
        nmi = None
        try:
            history = run.history(keys=["val/nmi"])
            if not history.empty and "val/nmi" in history.columns:
                nmi = float(history["val/nmi"].max())
        except Exception:
            pass
        
        # Fallback to summary if history fetch failed
        if nmi is None:
            nmi = (summary_dict.get("val/nmi") or
                  summary_dict.get("nmi") or  
                  summary_dict.get("aggregate/nmi"))
        
        print(f"   Found: LL={val_ll}, NMI={nmi} (max across epochs)")
        
        if val_ll is None or nmi is None:
            print(f"⚠️  Skipping run {run.id} - missing metrics (LL={val_ll}, NMI={nmi})")
            continue
        
        # Store result for this n_states (one run per n_states)
        # LL: final validation log likelihood
        # NMI: max NMI achieved during training
        results[n_states] = {
            'val_log_likelihood': float(val_ll),
            'nmi': float(nmi)
        }
        
        # Try to get model artifact path (if saved) - use first encountered for each n_states
        if n_states not in model_paths:
            try:
                # Check if run has model artifacts
                # W&B typically saves model files, we'll try to find model.pth
                # This is optional - if not available, we skip PCA plots
                for artifact in run.logged_artifacts():
                    if 'model' in artifact.name.lower():
                        model_paths[n_states] = f"{run.id}/{artifact.name}"
                        break
            except Exception:
                pass  # Model artifacts not available, skip PCA for this n_states
        
        print(f"✓ n_states={n_states}: LL={val_ll:.2f}, NMI={nmi:.4f}")
    
    # Set defaults if not found
    if model_type is None:
        model_type = "unknown"
    if dataset_type is None:
        dataset_type = "unknown"
    
    print(f"\nDetected: Model={model_type.upper()}, Dataset={dataset_type.upper()}")
    
    return results, model_paths, model_type, dataset_type


def plot_substages_results(results: Dict[int, Dict[str, float]], 
                           model_paths: Dict[int, str],
                           output_path: Path, 
                           sweep_name: str = "substages",
                           generate_pca: bool = False,
                           model_type: str = "unknown",
                           dataset_type: str = "unknown"):
    """Generate the substages sweep visualization.
    
    Args:
        results: Dict mapping n_states -> metrics
        model_paths: Dict mapping n_states -> model artifact path (optional, for PCA)
        output_path: Directory to save plot
        sweep_name: Name to use in title
        generate_pca: Whether to attempt PCA visualization (requires model files)
        model_type: Type of model (hmm/marhmm)
        dataset_type: Type of dataset (synthetic/mssv)
    """
    # Import here to avoid requiring matplotlib for --help
    import sys
    from pathlib import Path as P
    # Add parent directory to path to allow imports
    repo_root = P(__file__).parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    
    from src.config.config import GlobalConfig
    from src.visuals.visualizer import Visualizer
    
    if not results:
        print("❌ No valid results to plot!")
        return
    
    print(f"\n{'='*60}")
    print("SUBSTAGES SWEEP RESULTS")
    print(f"{'='*60}")
    for n_states in sorted(results.keys()):
        r = results[n_states]
        print(f"n_states={n_states:2d}: LL={r['val_log_likelihood']:10.2f}, NMI={r['nmi']:.4f}")
    print(f"{'='*60}\n")
    
    # Create minimal config for visualizer (only needs enough to initialize)
    config_dict = {
        "trainer": {
            "epochs": 1, 
            "learning_rate": 0.001,
            "optimizer": "adam",
            "grad_clip": None,
            "validate_per_epoch": 200,
            "early_stopping": {
                "enabled": False,
                "patience": 10,
                "min_delta": 0.001
            },
            "scheduler": {
                "enabled": False,
                "type": "step",
                "step_size": 100,
                "gamma": 0.1
            }
        },
        "validator": {"nmi": True, "accuracy": False, "cross_nmi": False, "learning_rate": False, "state_distinctness": False, "summary_statistics": False, "log_likelihood": False},
        "visualizer": {"losses": True, "pca_tripanel": False, "confusion_matrix": False, "learning_rate": False, "historic_values": False, "state_distinctness": False, "summary_statistics": False},
        "dataloader": {"batch_size": 32, "window_size": 512, "stride": 512, "num_batches": 100, "shuffle": True, "normalize": True, "transforms": [], "validation_batch_size": 32},
        "train_datasets": [],
        "val_datasets": [],
        "model": {"type": "hmm", "n_states": 3, "covariance_type": "diag", "init_strategy": "random_uniform", "init_noisy": False, "features": False, "params": {}},
        "results_dir": str(output_path.parent),
        "run_name": output_path.name,
        "seed": 42,
        "runs": 1,
        "verbose": False,
        "validate_data": False
    }
    
    config = GlobalConfig.model_validate(config_dict)
    
    # Create visualizer (with dummy data_loader since we only need plotting methods)
    visualizer = Visualizer(data_loader=None, config=config, validator=None)  # type: ignore
    
    # Generate plots
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📈 Generating substages sweep plot...")
    visualizer.visualize_substages_sweep(
        sweep_results=results, 
        path=output_path,
        model_type=model_type.upper(),
        dataset_type=dataset_type.upper()
    )
    
    # Create filename with model and dataset type
    plot_filename_parts = ['substages']
    if model_type and model_type != 'unknown':
        plot_filename_parts.append(model_type.lower())
    if dataset_type and dataset_type != 'unknown':
        plot_filename_parts.append(dataset_type.lower())
    plot_filename_parts.append('metrics.png')
    plot_filename = '_'.join(plot_filename_parts)
    
    print(f"✅ Plot saved to: {output_path / plot_filename}")
    
    # Also save results as JSON for record with same naming convention
    json_filename_parts = ['substages']
    if model_type and model_type != 'unknown':
        json_filename_parts.append(model_type.lower())
    if dataset_type and dataset_type != 'unknown':
        json_filename_parts.append(dataset_type.lower())
    json_filename_parts.append('sweep_results.json')
    json_filename = '_'.join(json_filename_parts)
    
    results_file = output_path / json_filename
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Results saved to: {results_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze substages sweep results from W&B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze sweep and save to default location
  python scripts/analyze_substages_sweep.py abc123def456
  
  # Specify custom output directory
  python scripts/analyze_substages_sweep.py abc123def456 --output_path results/my_analysis
  
  # Use different W&B project
  python scripts/analyze_substages_sweep.py abc123def456 --project MyProject --entity my_team
"""
    )
    
    parser.add_argument("sweep_id", type=str, 
                       help="W&B sweep ID (from sweep URL or bjobs output)")
    parser.add_argument("--output_path", type=str, default=None,
                       help="Directory to save plots (default: results/substages_analysis/<sweep_id>)")
    parser.add_argument("--project", type=str, default="SPA",
                       help="W&B project name (default: SPA)")
    parser.add_argument("--entity", type=str, default="dtu_projects",
                       help="W&B entity/team name (default: dtu_projects)")
    parser.add_argument("--pca", action="store_true",
                       help="Generate PCA comparison plots (requires model checkpoints)")
    
    args = parser.parse_args()
    
    # Determine output path
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = Path("results") / "substages_analysis" / args.sweep_id
    
    print(f"\n{'='*60}")
    print(f"SUBSTAGES SWEEP ANALYSIS")
    print(f"{'='*60}")
    print(f"Sweep ID: {args.sweep_id}")
    print(f"Output:   {output_path}")
    print(f"PCA Plots: {'Yes' if args.pca else 'No'}")
    print(f"{'='*60}\n")
    
    # Collect results from W&B
    results, model_paths, model_type, dataset_type = collect_sweep_results(
        sweep_id=args.sweep_id,
        project=args.project,
        entity=args.entity
    )
    
    if not results:
        print("\n❌ No valid results found in sweep!")
        print("   Make sure:")
        print("   - The sweep has completed")
        print("   - Runs logged val/log_likelihood and val/nmi metrics")
        print("   - You're authenticated with wandb (wandb login)")
        sys.exit(1)
    
    # Generate visualization
    plot_substages_results(
        results=results,
        model_paths=model_paths,
        output_path=output_path,
        sweep_name=args.sweep_id,
        generate_pca=args.pca,
        model_type=model_type,
        dataset_type=dataset_type
    )
    
    print(f"\n{'='*60}")
    print("✅ ANALYSIS COMPLETE!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
