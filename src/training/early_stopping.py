"""
Adaptive Early Stopping for Training.
Automatically switches between supervised and unsupervised metrics.
"""

import numpy as np
import torch
from typing import Dict, Optional


class EarlyStopping:
    """
    Smart early stopping that adapts to available metrics:
    - Supervised: Uses NMI/accuracy when labels available
    - Unsupervised: Uses parameter convergence when no labels
    
    Works with existing validation schedule (validate_per_epoch).
    """
    
    def __init__(self, patience: int = 20, min_delta: float = 0.001, restore_best: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        
        # State tracking
        self.best_score = None
        self.best_weights = None
        self.patience_counter = 0
        self.should_stop = False
        self.mode = None  # Will be 'supervised' or 'unsupervised'
        
        # For unsupervised mode
        self.prev_params = None
        self.validation_count = 0  # Track how many validations we've done
        
    def __call__(self, validations: Dict, model: torch.nn.Module, epoch: int) -> bool:
        """Check if training should stop. Returns True to stop."""
        
        self.validation_count += 1
        
        # Auto-detect mode on first call
        if self.mode is None:
            self.mode = self._detect_mode(validations)
            print(f"🎯 Early stopping mode: {self.mode}")
        
        # Get score based on mode
        score = self._get_score(validations, model)
        if score is None:
            return False
        
        # Check for improvement
        if self._is_improvement(score):
            self.best_score = score
            self.patience_counter = 0
            if self.restore_best:
                self.best_weights = {k: v.clone() for k, v in model.named_parameters()}
        else:
            self.patience_counter += 1
        
        # Check if we should stop (patience is in terms of validation steps, not epochs)
        if self.patience_counter >= self.patience:
            self.should_stop = True
            if self.restore_best and self.best_weights:
                for name, param in model.named_parameters():
                    if name in self.best_weights:
                        param.data.copy_(self.best_weights[name])
                print(f"🔄 Restored best weights")
        
        return self.should_stop
    
    def _detect_mode(self, validations: Dict) -> str:
        """Detect if we have supervised metrics available."""
        supervised_metrics = ["nmi", "accuracy"]
        has_supervised = any(metric in validations for metric in supervised_metrics)
        return "supervised" if has_supervised else "unsupervised"
    
    def _get_score(self, validations: Dict, model: torch.nn.Module) -> Optional[float]:
        """Get score based on current mode."""
        
        if self.mode == "supervised":
            return self._get_supervised_score(validations)
        else:
            return self._get_unsupervised_score(model)
    
    def _get_supervised_score(self, validations: Dict) -> Optional[float]:
        """Get best available supervised metric."""
        # Try NMI first, then accuracy
        for metric in ["nmi", "accuracy"]:
            if metric in validations and isinstance(validations[metric], (int, float)):
                return float(validations[metric])
        return None
    
    def _get_unsupervised_score(self, model: torch.nn.Module) -> float:
        """Calculate parameter convergence score (higher = more converged)."""
        # Get current parameters
        current_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                current_params[name] = param.detach().cpu().numpy()
        
        # First epoch - save params and return high score
        if self.prev_params is None:
            self.prev_params = current_params
            return 1.0
        
        # Calculate parameter change
        total_change = 0.0
        total_norm = 0.0
        
        for name in current_params:
            if name in self.prev_params:
                diff = current_params[name] - self.prev_params[name]
                change = np.linalg.norm(diff)
                norm = np.linalg.norm(current_params[name])
                
                total_change += change
                total_norm += norm
        
        # Update for next comparison
        self.prev_params = current_params
        
        # Convert to convergence score (higher = better)
        if total_norm > 0:
            relative_change = total_change / total_norm
            convergence_score = 1.0 / (1.0 + relative_change)  # Higher when change is small
            return convergence_score
        else:
            return 1.0
    
    def _is_improvement(self, score: float) -> bool:
        """Check if current score is better than best."""
        if self.best_score is None:
            return True
        return score > self.best_score + self.min_delta


def create_early_stopper(config_trainer, global_verbose: bool) -> Optional[EarlyStopping]:
    """Factory to create an EarlyStopping instance or None.
    Patience now interpreted directly in *epochs* (trainer is responsible for how often
    the check is invoked). This decouples early stopping frequency from validation
    frequency. The trainer may call the stopper every epoch, every N epochs, or after
    validation; the stopper just counts calls that produce a score.
    """
    if not getattr(config_trainer, 'early_stopping', False):
        return None
    return EarlyStopping(
        patience=getattr(config_trainer, 'patience', 20),
        min_delta=getattr(config_trainer, 'min_delta', 0.001),
    )