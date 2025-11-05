"""
Early stopping based solely on training loss.
Supports a single LR-on-plateau reduction and best-weight restoration.
"""

import numpy as np
import torch
from typing import Dict, Optional


class EarlyStopping:
    """Early stopping driven by training loss only.

    Features:
        - Relative-only improvement threshold (min_delta interpreted as fractional gain)
        - Single learning rate reduction on first plateau, hard stop on second
    """

    min_lr = 1e-5
    lr_factor = 0.5

    def __init__(self, patience: int = 20, min_delta: float = 0.001):
        self.patience = patience              # epochs without sufficient relative improvement
        self.min_delta = min_delta            # required relative improvement (e.g. 0.001 = +0.1%)
        self.restore_best = True              # always restore best

    # Internal state
        self._best_loss: Optional[float] = None
        self._best_weights = None
        self._patience_used = 0
        self._lr_reduced = False
        self._should_stop = False

    # Public API --------------------------------------------------------------------
    def step(self, train_loss: float, model: torch.nn.Module, epoch: int, optimizer: Optional[torch.optim.Optimizer] = None) -> bool:
        """Update early stopping from the current training loss.

        Returns True if training should stop, else False.
        """
        if train_loss is None or np.isnan(train_loss) or np.isinf(train_loss):
            return False  # ignore invalid values

        if self._best_loss is None or self._is_improvement_loss(train_loss):
            self._update_best(train_loss, model)
            self._patience_used = 0
        else:
            self._patience_used += 1

        if self._patience_used >= self.patience:
            if optimizer is not None and not self._lr_reduced:
                reduced, changes = self._apply_lr_reduction(optimizer)
                if reduced:
                    self._lr_reduced = True
                    self._patience_used = 0
                    for i, (old_lr, new_lr) in enumerate(changes):
                        self._log(f"LR reduced (group {i}): {old_lr:.6g} -> {new_lr:.6g} (x{new_lr/old_lr:.3f})")
                    self._log("Patience reset after LR reduction")
                    return False
            self._should_stop = True
            self._restore_best(model)
            self._log_stop(optimizer)

        return self._should_stop

    # Back-compat: allow callable-style with dict or float ---------------------------
    def __call__(self, arg, model: torch.nn.Module, epoch: int, optimizer: Optional[torch.optim.Optimizer] = None) -> bool:
        """Compatibility wrapper to support existing call-sites.

        Accepts either a dict with key 'train_loss' or a raw float loss.
        """
        if isinstance(arg, dict):
            val = arg.get('train_loss')
            if not isinstance(val, (int, float)):
                return False
            return self.step(float(val), model, epoch, optimizer)
        elif isinstance(arg, (int, float)):
            return self.step(float(arg), model, epoch, optimizer)
        else:
            return False

    # ---- Improvement & Tracking -----------------------------------------------------
    def _is_improvement_loss(self, loss: float) -> bool:
        """Return True if loss decreased by more than min_delta (relative)."""
        if self._best_loss is None:
            return True
        rel_impr = (self._best_loss - loss) / (abs(self._best_loss) + 1e-8)
        return rel_impr > self.min_delta

    def _update_best(self, loss: float, model: torch.nn.Module) -> None:
        if self._best_loss is None or loss < self._best_loss:
            self._best_loss = loss
            self._best_weights = {k: v.clone() for k, v in model.named_parameters()}

    # ---- Learning Rate Handling -----------------------------------------------------
    def _apply_lr_reduction(self, optimizer: torch.optim.Optimizer) -> tuple[bool, list[tuple[float, float]]]:
        changes: list[tuple[float, float]] = []
        for param_group in optimizer.param_groups:
            old_lr = param_group.get('lr')
            if old_lr is None or old_lr <= self.min_lr * 1.01:
                continue
            new_lr = max(self.min_lr, old_lr * self.lr_factor)
            if new_lr < old_lr:
                param_group['lr'] = new_lr
                changes.append((old_lr, new_lr))
        return (len(changes) > 0, changes)

    # ---- Restoring & Logging --------------------------------------------------------
    def _restore_best(self, model: torch.nn.Module) -> None:
        if not self._best_weights:
            return
        for name, param in model.named_parameters():
            if name in self._best_weights:
                param.data.copy_(self._best_weights[name])
        self._log("Restored best weights")

    def _log_stop(self, optimizer: Optional[torch.optim.Optimizer]) -> None:
        lr = None
        if optimizer and optimizer.param_groups:
            lr = optimizer.param_groups[0].get('lr')
        if lr is not None:
            self._log(f"Stopping (lr={lr:.6g}, best_loss={self._best_loss:.6f})")
        else:
            self._log(f"Stopping (best_loss={self._best_loss:.6f})")

    def _log(self, msg: str) -> None:
        print(f"[EarlyStopping] {msg}")


# Factory ---------------------------------------------------------------------
def create_early_stopper(config_trainer, verbose: bool = True) -> Optional[EarlyStopping]:
    early_stopping_cfg = getattr(config_trainer, 'early_stopping', None)
    if early_stopping_cfg is None:
        return None
    if hasattr(early_stopping_cfg, 'enabled') and not early_stopping_cfg.enabled:
        return None
    patience = getattr(early_stopping_cfg, 'patience', 20)
    min_delta = getattr(early_stopping_cfg, 'min_delta', 0.002)
    stopper = EarlyStopping(patience=patience, min_delta=min_delta)
    if verbose:
        print(f"[EarlyStopping] Enabled (patience={patience} epochs, rel_min_delta={min_delta})")
    return stopper