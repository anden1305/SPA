from __future__ import annotations
from typing import Dict, Optional
import torch

class ValidationController:
    """Encapsulates validation scheduling and early stopping decisions.

    Responsibilities:
    - Decide when to run validation based on validate_per_epoch.
    - Invoke validator and fetch metrics.
    - Invoke early stopping (if enabled) immediately after validation.
    - Return a tuple (ran_validation, should_stop).
    """

    def __init__(self, *, validator, early_stopper, config_trainer, global_verbose: bool):
        self.validator = validator
        self.early_stopper = early_stopper
        self.config_trainer = config_trainer
        self.verbose = global_verbose

    def step(self, epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> bool:
        """Run scheduled validation + early stopping.

        Returns
        -------
        bool
            True if training should stop, else False.
        """
        ran_validation = False
        should_stop = False
        validations: Dict = {}

        # Decide if we validate this epoch
        if self.config_trainer.validate_per_epoch > 0 and (epoch + 1) % self.config_trainer.validate_per_epoch == 0:
            self.validator.validate_epoch(epoch)
            ran_validation = True
            validations = self.validator.validations.get(epoch, {})

        # Early stopping only after validation
        if self.early_stopper is not None and ran_validation:
            if not validations and self.verbose:
                print(f"[early-stopping] Epoch {epoch + 1}: no validation metrics; using unsupervised criterion.")
            should_stop = self.early_stopper(validations, model, epoch, optimizer=optimizer)
            if should_stop and self.verbose:
                print(f"🛑 Early stopping at epoch {epoch + 1}")

        return should_stop
