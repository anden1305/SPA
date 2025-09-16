"""
Adaptive Early Stopping for Training.
Automatically switches between supervised and unsupervised metrics.
"""

import numpy as np
import torch
from typing import Dict, Optional


class EarlyStopping:
    """Adaptive early stopping with metric auto-selection, EMA smoothing and LR-on-plateau.

    Modes:
      - supervised: prefer NMI, fallback to accuracy
      - unsupervised: parameter convergence proxy
    Features:
      - EMA smoothing of score
      - Warmup window (no stopping during first N validations)
      - Relative or absolute improvement threshold (min_delta < 1 => relative)
      - Single learning rate reduction on first plateau, stop on second
    """

    _EMA_ALPHA = 0.3
    _WARMUP_VALIDATIONS = 5
    _MIN_LR = 1e-5
    _LR_FACTOR = 0.5

    def __init__(self, patience: int = 20, min_delta: float = 0.001, restore_best: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best

        # Internal state
        self._mode: Optional[str] = None
        self._best_score: Optional[float] = None
        self._best_weights = None
        self._patience_used = 0
        self._validations_seen = 0
        self._ema: Optional[float] = None
        self._prev_params: Optional[Dict[str, np.ndarray]] = None
        self._lr_reduced = False
        self._should_stop = False

    def __call__(self, validations: Dict, model: torch.nn.Module, epoch: int, optimizer: Optional[torch.optim.Optimizer] = None) -> bool:
        """Update state from a validation event and decide whether to stop."""
        self._validations_seen += 1

        if self._mode is None:
            self._mode = self._infer_mode(validations)
            self._log(f"Mode: {self._mode}")

        score_raw = self._select_score(validations, model)
        if score_raw is None:
            return False

        score_smoothed = self._update_ema(score_raw)

        if self._validations_seen <= self._WARMUP_VALIDATIONS:
            self._update_best(score_smoothed, model)
            return False

        if self._is_improvement(score_smoothed):
            self._update_best(score_raw, model)
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

    # ---- Metric Selection & Scoring -------------------------------------------------
    def _infer_mode(self, validations: Dict) -> str:
        if any(m in validations for m in ("nmi", "accuracy")):
            return "supervised"
        return "unsupervised"

    def _select_score(self, validations: Dict, model: torch.nn.Module) -> Optional[float]:
        return self._score_supervised(validations) if self._mode == "supervised" else self._score_unsupervised(model)

    def _score_supervised(self, validations: Dict) -> Optional[float]:
        for metric in ("nmi", "accuracy"):
            val = validations.get(metric)
            if isinstance(val, (int, float)):
                return float(val)
        return None

    def _score_unsupervised(self, model: torch.nn.Module) -> float:
        current_params: Dict[str, np.ndarray] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                current_params[name] = param.detach().cpu().numpy()
        if self._prev_params is None:
            self._prev_params = current_params
            return 1.0
        total_change = 0.0
        total_norm = 0.0
        for name in current_params:
            if name in self._prev_params:
                diff = current_params[name] - self._prev_params[name]
                total_change += np.linalg.norm(diff)
                total_norm += np.linalg.norm(current_params[name])
        self._prev_params = current_params
        if total_norm > 0:
            rel_change = total_change / total_norm
            return 1.0 / (1.0 + rel_change)
        return 1.0

    # ---- Improvement & Tracking -----------------------------------------------------
    def _is_improvement(self, score: float) -> bool:
        if self._best_score is None:
            return True
        if self.min_delta < 1.0:
            rel = (score - self._best_score) / (abs(self._best_score) + 1e-8)
            return rel > self.min_delta
        return score > self._best_score + self.min_delta

    def _update_best(self, score: float, model: torch.nn.Module) -> None:
        if self._best_score is None or score > self._best_score:
            self._best_score = score
            if self.restore_best:
                self._best_weights = {k: v.clone() for k, v in model.named_parameters()}

    def _update_ema(self, score: float) -> float:
        self._ema = score if self._ema is None else (self._EMA_ALPHA * score + (1 - self._EMA_ALPHA) * self._ema)
        return self._ema

    # ---- Learning Rate Handling -----------------------------------------------------
    def _apply_lr_reduction(self, optimizer: torch.optim.Optimizer) -> tuple[bool, list[tuple[float, float]]]:
        changes: list[tuple[float, float]] = []
        for param_group in optimizer.param_groups:
            old_lr = param_group.get('lr')
            if old_lr is None or old_lr <= self._MIN_LR * 1.01:
                continue
            new_lr = max(self._MIN_LR, old_lr * self._LR_FACTOR)
            if new_lr < old_lr:
                param_group['lr'] = new_lr
                changes.append((old_lr, new_lr))
        return (len(changes) > 0, changes)

    # ---- Restoring & Logging --------------------------------------------------------
    def _restore_best(self, model: torch.nn.Module) -> None:
        if not (self.restore_best and self._best_weights):
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
            self._log(f"Stopping (lr={lr:.6g}, best_score={self._best_score:.6f})")
        else:
            self._log(f"Stopping (best_score={self._best_score:.6f})")

    def _log(self, msg: str) -> None:
        print(f"[EarlyStopping] {msg}")


# Factory ---------------------------------------------------------------------
def create_early_stopper(config_trainer, verbose: bool = True) -> Optional[EarlyStopping]:
    """Factory returning an EarlyStopping instance or None.

    Expects config_trainer to expose:
      - early_stopping (bool)
      - patience (int)
      - min_delta (float)
      - restore_best (optional, default True if missing)
    """
    if not getattr(config_trainer, 'early_stopping', False):
        return None
    patience = getattr(config_trainer, 'patience', 20)
    min_delta = getattr(config_trainer, 'min_delta', 0.001)
    restore_best = getattr(config_trainer, 'restore_best', True)
    stopper = EarlyStopping(patience=patience, min_delta=min_delta, restore_best=restore_best)
    if verbose:
        print(f"[EarlyStopping] Enabled (patience={patience}, min_delta={min_delta}, restore_best={restore_best})")
    return stopper