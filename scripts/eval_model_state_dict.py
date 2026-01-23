"""Evaluate model checkpoints at different training stages on held-out subject."""

import torch
import json
from pathlib import Path
import numpy as np
from typing import Dict, List

from src.config.config import GlobalConfig
from src.data.data_loader_collection import DataLoaderCollection
from src.data.mssv_dataset import MSSVDataset
from src.models.hmm import HMM
from src.models.marhmm import MARHMM
from src.helpers.nmi import calculate_nmi
from src.helpers.accuracy import accuracy
from src.helpers.align_labels import align_labels_hungarian


def load_model_at_epoch(
    model_class,
    config: GlobalConfig,
    data_loader: DataLoaderCollection,
    device: torch.device,
    checkpoint_path: Path
) -> torch.nn.Module:
    """Load model with state dict from a specific checkpoint."""
    model = model_class(
        data_loader=data_loader,
        config=config,
        device=device
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def evaluate_model(
    model,
    data_loader: DataLoaderCollection,
    config: GlobalConfig
) -> Dict[str, float]:
    """Evaluate model and return metrics."""
    x, y = data_loader.get_all_data()
    
    # Handle MARHMM burn-in if needed
    if hasattr(model, 'max_lag') and model.max_lag > 0:
        x_eval = x[:, model.max_lag:, :]
        y_eval = y[:, model.max_lag:]
    else:
        x_eval = x
        y_eval = y
    
    with torch.no_grad():
        # Get predictions
        preds = model.predict(x_eval)
        
        # Compute forward pass for log-likelihood
        nll = model.forward(x_eval)
        log_likelihood = -nll.item()
        
        # Convert to numpy for metric calculation
        y_np = y_eval.detach().cpu().numpy().flatten()
        preds_np = preds.detach().cpu().numpy().flatten()
        
        # Compute metrics
        nmi = calculate_nmi(preds_np, y_np)
        aligned_preds = align_labels_hungarian(y_np, preds_np)
        acc = accuracy(aligned_preds, y_np)
    
    return {
        'nmi': nmi,
        'accuracy': acc,
        'log_likelihood': log_likelihood,
        'nll': nll.item()
    }


def find_historic_checkpoints(run_dir: Path) -> List[Dict]:
    """Find all saved checkpoints from historic values.
    
    You'll need to save model checkpoints during training at validation epochs.
    For now, we assume you have model.pth at the final epoch.
    """
    checkpoints = []
    
    # Final checkpoint
    final_ckpt = run_dir / "model.pth"
    if final_ckpt.exists():
        checkpoints.append({
            'path': final_ckpt,
            'epoch': 'final'
        })
    
    # TODO: Add logic to find intermediate checkpoints if you save them
    # For example, if you save model_epoch_100.pth, model_epoch_200.pth, etc.
    
    return checkpoints


def main(
    results_dir: str,
    run_name: str,
    run_number: int = 1,
    test_subject: str = "sub-058",
    test_runs: List[int] = [1, 2]
):
    """Main evaluation function."""
    
    # Load config from the training run
    run_path = Path(results_dir) / run_name
    config_path = run_path / "config.json"
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    config = GlobalConfig(**config_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create test dataset(s)
    test_datasets = []
    for run in test_runs:
        from src.config.config import DatasetConfig
        ds_config = DatasetConfig(
            type="mssv",
            id=test_subject,
            run=run,
            remove_artifact=True
        )
        test_datasets.append(MSSVDataset(config=ds_config))
    
    # Create data loader for test data
    test_loader = DataLoaderCollection(
        datasets=test_datasets,
        config=config,
        for_validation=True,
        device=device
    )
    
    # Determine model class
    model_class = HMM if config.model.type == "hmm" else MARHMM
    
    # Find checkpoints
    run_dir = run_path / str(run_number)
    checkpoints = find_historic_checkpoints(run_dir)
    
    # Evaluate each checkpoint
    results = []
    for ckpt in checkpoints:
        print(f"\nEvaluating checkpoint: {ckpt['epoch']}")
        
        model = load_model_at_epoch(
            model_class=model_class,
            config=config,
            data_loader=test_loader,  # Use test_loader for initialization
            device=device,
            checkpoint_path=ckpt['path']
        )
        
        metrics = evaluate_model(model, test_loader, config)
        
        result = {
            'epoch': ckpt['epoch'],
            'checkpoint': str(ckpt['path']),
            **metrics
        }
        results.append(result)
        
        print(f"  NMI: {metrics['nmi']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Log-likelihood: {metrics['log_likelihood']:.4f}")
    
    # Save results
    output_path = run_dir / "generalization_to_sub058.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    return results


if __name__ == "__main__":
    # Example usage
    results = main(
        results_dir="results/training",
        run_name="your_run_name [YYYYmmdd-HHMMSS]",
        run_number=1,
        test_subject="sub-058",
        test_runs=[1, 2]
    )