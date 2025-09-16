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
        # EMA tracking
        self.ema_score = None
        self.ema_alpha = 0.3  # Fixed smoothing factor (not exposed in config)
        self.warmup_validations = 2  # Number of validation points before early stopping decisions

        # LR scheduling state
        self.lr_reduced = False  # Whether we've already reduced LR once
        self.stop_after_lr = False  # Trigger stop after second plateau
        self.min_lr = 1e-5
        self.lr_factor = 0.5
        
    def __call__(self, validations: Dict, model: torch.nn.Module, epoch: int, optimizer: Optional[torch.optim.Optimizer] = None) -> bool:
        """Check if training should stop. Returns True to stop.

        Parameters
        ----------
        validations : Dict
            Validation metrics for this epoch (may be empty for unsupervised fallback).
        model : torch.nn.Module
            Model being trained.
        epoch : int
            Current epoch index.
        optimizer : torch.optim.Optimizer, optional
            Optimizer for potential LR scheduling.
        """
        
        self.validation_count += 1
        
        # Auto-detect mode on first call
        if self.mode is None:
            self.mode = self._detect_mode(validations)
            print(f"🎯 Early stopping mode: {self.mode}")
        
        # Get score based on mode
        score = self._get_score(validations, model)
        if score is None:
            return False
        
        # Update EMA score
        score_for_eval = self._update_ema(score)

        # During warmup just collect statistics
        if self.validation_count <= self.warmup_validations:
            self.best_score = score_for_eval if self.best_score is None else max(self.best_score, score_for_eval)
            if self.restore_best:
                self.best_weights = {k: v.clone() for k, v in model.named_parameters()}
            return False

        # Check for improvement using EMA-adjusted score
        if self._is_improvement(score_for_eval):
            self.best_score = score
            self.patience_counter = 0
            if self.restore_best:
                self.best_weights = {k: v.clone() for k, v in model.named_parameters()}
        else:
            self.patience_counter += 1
        
        # Plateau handling
        if self.patience_counter >= self.patience:
            # First plateau: try reducing LR if possible
            if optimizer is not None and not self.lr_reduced and self._maybe_reduce_lr(optimizer):
                self.lr_reduced = True
                self.patience_counter = 0
                print("⚙️  EarlyStopping: Plateau detected. Reduced learning rate and reset patience.")
            else:
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
        # Relative threshold if min_delta < 1, absolute otherwise
        if self.min_delta < 1.0:
            rel_improvement = (score - self.best_score) / (abs(self.best_score) + 1e-8)
            return rel_improvement > self.min_delta
        else:
            return score > self.best_score + self.min_delta

    def _update_ema(self, score: float) -> float:
        if self.ema_score is None:
            self.ema_score = score
        else:
            self.ema_score = self.ema_alpha * score + (1 - self.ema_alpha) * self.ema_score
        return self.ema_score

    def _maybe_reduce_lr(self, optimizer: torch.optim.Optimizer) -> bool:
        """Reduce learning rate if above minimum. Returns True if reduced."""
        reduced = False
        for param_group in optimizer.param_groups:
            old_lr = param_group.get('lr', None)
            if old_lr is None:
                continue
            if old_lr <= self.min_lr * 1.01:  # effectively at floor
                return False
            new_lr = max(self.min_lr, old_lr * self.lr_factor)
            if new_lr < old_lr:
                param_group['lr'] = new_lr
                reduced = True
                print(f"⚖️  Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
        return reduced


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