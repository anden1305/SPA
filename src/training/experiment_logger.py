from __future__ import annotations

from typing import Any, Dict, Optional
import os

class ExperimentLogger:
    """Tiny logger wrapper to keep trainer.py uncluttered.

    Methods are no-ops if wandb is unavailable or disabled.
    Reuses an existing wandb run if launched by a sweep.
    """

    def __init__(self, *, enabled: bool = True, project: str | None = None, entity: str | None = None, group: str | None = None, tags: list[str] | None = None):
        self.enabled = enabled
        self.project = project
        self.entity = entity
        self.group = group
        self.tags = tags
        try:
            import wandb  # type: ignore
        except Exception:
            self._wb = None
        else:
            self._wb = wandb
        self._started_here = False

    def start(self, *, run_name: str, config: Dict[str, Any]):
        if not self.enabled or self._wb is None:
            return
        if getattr(self._wb, 'run', None) is not None:
            # Reuse run created by a sweep; ensure full config is present
            try:
                self._wb.config.update(config, allow_val_change=True)
            except Exception:
                pass
            return
        self._wb.init(
            project=self.project,
            entity=self.entity,
            name=run_name,
            group=self.group,
            tags=self.tags,
            config=config,
        )
        self._started_here = True

    def log_epoch(self, *, step: int, train_total_loss: float, train_reg_loss: float, lr: Optional[float] = None, val_metrics: Optional[Dict[str, float]] = None, epoch: Optional[int] = None, run_seed: Optional[int] = None):
        """Log per-epoch metrics to Weights & Biases.

        Note: We intentionally do NOT pass an explicit `step` to wandb.log.
        During sweeps we reuse a single wandb run while running multiple
        internal orchestrator runs (cfg.runs > 1). Epoch indices restart at 0
        for each internal run, which can cause non‑monotonic step warnings if
        we forward the epoch number. Letting W&B auto-increment ensures steps
        stay monotonic across internal runs.
        """
        if not self.enabled or self._wb is None or getattr(self._wb, 'run', None) is None:
            return
        payload: Dict[str, Any] = {
            'train/total_loss': float(train_total_loss),
            'train/reg_loss': float(train_reg_loss),
        }
        # Derive and include pure data loss for convenience
        try:
            payload['train/data_loss'] = float(train_total_loss - train_reg_loss)
        except Exception:
            pass
        if epoch is not None:
            payload['epoch'] = int(epoch)
        if run_seed is not None:
            payload['run/seed'] = int(run_seed)
        if lr is not None:
            payload['train/lr'] = float(lr)
        if isinstance(val_metrics, dict):
            for k, v in val_metrics.items():
                try:
                    float_v = float(v)
                except Exception as e:
                    print(f"Warning: skipping non-float val metric '{k}': {v} ({e})")
                    continue
                if k.startswith('train_'):
                    # Remove 'train_' prefix and add to train/ namespace
                    payload[f'train/{k[6:]}'] = float(v)
                else:
                    payload[f'val/{k}'] = float(v)
        # Don't pass step to avoid non-monotonic warnings across internal runs
        self._wb.log(payload)

    def update_summary(self, metrics: Dict[str, float | int | None]) -> None:
        """Write scalar run summaries (e.g. best checkpoint epoch, final prior NMI)."""
        if not self.enabled or self._wb is None or getattr(self._wb, "run", None) is None:
            return
        for key, value in metrics.items():
            if value is None:
                continue
            try:
                self._wb.summary[key] = float(value) if isinstance(value, (int, float)) else value
            except Exception:
                pass

    def finish(self):
        if not self.enabled or self._wb is None:
            return
        if self._started_here and getattr(self._wb, 'run', None) is not None:
            self._wb.finish()


def create_logger(global_config) -> ExperimentLogger:
    # Enabled by config.wandb.enabled if available, else no-op
    wb_cfg = getattr(global_config, 'wandb', None)
    enabled = getattr(wb_cfg, 'enabled', True) if wb_cfg is not None else False
    group = getattr(wb_cfg, 'group', None) if wb_cfg is not None else None
    tags = getattr(wb_cfg, 'tags', None) if wb_cfg is not None else None
    # Allow environment to override defaults to match HPC/team settings
    project = os.getenv('WANDB_PROJECT', 'SPA')
    entity = os.getenv('WANDB_ENTITY', 'dtu_projects')
    return ExperimentLogger(
        enabled=enabled,
        project=project,
        entity=entity,
        group=group,
        tags=tags,
    )
