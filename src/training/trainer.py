from src.models.cvae_mar_hmm import CVAEMARHMM
from src.data.data_loader_collection import DataLoaderCollection
from src.models.base_model import BaseModel
from src.validation.validator import Validator
from src.config.config import GlobalConfig
from src.training.early_stopping import create_early_stopper
from src.training.experiment_logger import create_logger
from src.helpers.profiling import write_cprofile_outputs
import cProfile
from pathlib import Path
from torch.optim import Adam, SGD, RMSprop
import torch

class Trainer:
    
    def __init__(self,
                 data_loader: DataLoaderCollection,
                 model: BaseModel,
                 config: GlobalConfig,
                 validator: Validator):
        self.data_loader = data_loader
        self.model = model
        self.global_config = config
        self.config = self.global_config.trainer
        self.losses: dict[int, float] = {}
        self.regularization_losses: dict[int, float] = {}
        self.current_epoch = 0
        self.validator = validator
        self.early_stopping = create_early_stopper(self.config, self.global_config.verbose)
        self._best_kmeans_nmi = -1.0
        self._best_kmeans_epoch: int | None = None
        self._best_checkpoint_score = float("-inf")
        self._best_checkpoint_score_epoch: int | None = None
        self._experiment_logger = None
        self.results_run_subdir: str | None = None

    def _checkpoint_dir(self) -> Path:
        base = Path(self.global_config.results_dir) / self.global_config.run_name
        if self.results_run_subdir is not None:
            out = base / self.results_run_subdir / "checkpoints"
        else:
            out = base
        out.mkdir(parents=True, exist_ok=True)
        return out
    
    def __init_optimizer(self):
        if self.config.optimizer == "adam":
            self.optimizer = Adam(self.model.parameters(), lr=self.config.learning_rate)
        elif self.config.optimizer == "sgd":
            self.optimizer = SGD(self.model.parameters(), lr=self.config.learning_rate)
        elif self.config.optimizer == "rmsprop":
            self.optimizer = RMSprop(self.model.parameters(), lr=self.config.learning_rate)
        elif self.config.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
    
    def __init_scheduler(self):
        if not self.config.scheduler.enabled:
            self.scheduler = None
            return
        elif self.config.scheduler.type == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config.scheduler.step_size, gamma=self.config.scheduler.gamma)
        elif self.config.scheduler.type == "exponential":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.config.scheduler.gamma)
        else:
            raise ValueError(f"Unsupported scheduler: {self.config.scheduler}")
    
    def __init_training(self):
        self.__init_optimizer()
        self.__init_scheduler()
        self.epoch_losses: list[float] = []
        self.epoch_regularization_losses: list[float] = []
        self.losses: dict[int, float] = {}
        self.regularization_losses: dict[int, float] = {}
        self.model.prepare_for_training()
        self.current_epoch = 0
    
    def train(self):
        if self.global_config.verbose:
            self.__print_training_start()
        logger = create_logger(self.global_config)
        self._experiment_logger = logger
        logger.start(
            run_name=self.global_config.run_name,
            config=self.global_config.model_dump(),
        )
        self.__init_training()
        self._best_kmeans_nmi = -1.0
        self._best_kmeans_epoch = None
        self._best_checkpoint_score = float("-inf")
        self._best_checkpoint_score_epoch = None
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            skipped_batches = 0
            for x, _, sub_ids in self.data_loader:
                self.optimizer.zero_grad()
                if isinstance(self.model, CVAEMARHMM):
                    loss, reg_loss = self.model.forward(x, sub_ids, epoch)
                else:
                    loss = self.model.forward(x)
                    reg_loss = self.model.regularization_loss()
                total_loss = loss + reg_loss
                if total_loss.dim() != 0 or loss.dim() != 0 or reg_loss.dim() != 0:
                    raise RuntimeError(
                        f"Expected scalar losses at epoch={epoch}; "
                        f"got loss.shape={tuple(loss.shape)}, reg_loss.shape={tuple(reg_loss.shape)}"
                    )
                if not torch.isfinite(total_loss):
                    skipped_batches += 1
                    if self.global_config.verbose and skipped_batches <= 3:
                        recon_val = float(loss.detach().cpu()) if torch.isfinite(loss) else float("nan")
                        reg_val = float(reg_loss.detach().cpu()) if torch.isfinite(reg_loss) else float("nan")
                        print(
                            f"WARNING: non-finite loss at epoch={epoch}, skipping batch "
                            f"(recon={recon_val}, reg={reg_val})"
                        )
                    continue
                total_loss.backward()
                if self.config.grad_clip is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip
                    )
                    if not torch.isfinite(grad_norm):
                        skipped_batches += 1
                        self.optimizer.zero_grad()
                        if self.global_config.verbose and skipped_batches <= 3:
                            print(f"WARNING: non-finite grad norm at epoch={epoch}, skipping step")
                        continue
                self.optimizer.step()
                self.epoch_losses.append(total_loss.item())
                self.epoch_regularization_losses.append(reg_loss.item())
            if skipped_batches and self.global_config.verbose:
                print(f"Epoch {epoch + 1}: skipped {skipped_batches} non-finite batch(es)")
            if self.scheduler:
                    self.scheduler.step()
            if self.epoch_losses:
                self.losses[epoch] = sum(self.epoch_losses) / len(self.epoch_losses)
                self.regularization_losses[epoch] = sum(self.epoch_regularization_losses) / len(
                    self.epoch_regularization_losses
                )
            elif epoch > 0 and (epoch - 1) in self.losses:
                self.losses[epoch] = self.losses[epoch - 1]
                self.regularization_losses[epoch] = self.regularization_losses[epoch - 1]
            else:
                self.losses[epoch] = float("nan")
                self.regularization_losses[epoch] = float("nan")
            if self.config.validate_per_epoch > 0 and (epoch + 1) % self.config.validate_per_epoch == 0:
                if isinstance(self.model, CVAEMARHMM) and self.model.training_pipeline == 'cvae':
                    self.validator.validate_cvae_epoch(epoch)
                    self._maybe_save_best_kmeans_checkpoint(epoch)
                    self._maybe_save_best_checkpoint_score(epoch)
                else:
                    self.validator.validate_epoch(epoch, self.optimizer)
            lr = self.optimizer.param_groups[0].get('lr')
            val = self.validator.validations.get(epoch, {})
            logger.log_epoch(
                step=epoch,
                train_total_loss=self.losses[epoch],
                train_reg_loss=self.regularization_losses[epoch],
                lr=lr,
                val_metrics=val,
                epoch=epoch + 1,
                run_seed=int(self.global_config.seed),
            )
            if self.early_stopping is not None and self.early_stopping.step(self.losses[epoch], self.model, epoch, optimizer=self.optimizer):
                break
            self.epoch_losses.clear()
            self.epoch_regularization_losses.clear()
            if self.global_config.verbose:
                self.__print_epoch()
        self._log_wandb_training_summary()
        if self._experiment_logger is not None:
            defer = (
                isinstance(self.model, CVAEMARHMM)
                and self.model.training_pipeline == "cvae"
            )
            if not defer:
                self.finalize_wandb()
    
    def train_profiled(self, out_dir: str | Path, *, basename: str = "train", sort: str = "cumulative") -> None:
        prof = cProfile.Profile()
        prof.enable()
        try:
            self.train()
        finally:
            prof.disable()
            write_cprofile_outputs(prof, out_dir, basename=basename, sort=sort)
    
    def get_losses(self) -> dict[int, float]:
        assert self.losses, "Training has not been run yet."
        return self.losses
    
    def _maybe_save_best_kmeans_checkpoint(self, epoch: int) -> None:
        """Save CVAE weights when latent KMeans NMI improves (pre-collapse checkpoint)."""
        metrics = self.validator.validations.get(epoch, {})
        nmi = metrics.get("cvae_latent_kmeans_nmi")
        if nmi is None or nmi <= self._best_kmeans_nmi:
            return
        self._best_kmeans_nmi = float(nmi)
        self._best_kmeans_epoch = epoch
        path = self._checkpoint_dir() / "cvae_best_kmeans_nmi.pth"
        torch.save(self.model.state_dict(), path)
        if self.global_config.verbose:
            print(
                f"Saved best KMeans-NMI checkpoint (NMI={nmi:.4f}, epoch={epoch + 1}) to {path}",
                flush=True,
            )

    def get_best_kmeans_checkpoint_path(self) -> Path | None:
        path = self._checkpoint_dir() / "cvae_best_kmeans_nmi.pth"
        return path if path.exists() else None

    def _checkpoint_score_warmup_epoch(self) -> int:
        frac = self.config.checkpoint_score.warmup_frac
        return int(frac * self.config.epochs)

    def _maybe_save_best_checkpoint_score(self, epoch: int) -> None:
        if not self.config.checkpoint_score.enabled:
            return
        if epoch < self._checkpoint_score_warmup_epoch():
            return
        metrics = self.validator.validations.get(epoch, {})
        score = metrics.get("checkpoint_score")
        if score is None or score <= self._best_checkpoint_score:
            return
        self._best_checkpoint_score = float(score)
        self._best_checkpoint_score_epoch = epoch
        path = self._checkpoint_dir() / "cvae_best_checkpoint_score.pth"
        torch.save(self.model.state_dict(), path)
        if self.global_config.verbose:
            print(
                f"Saved best checkpoint-score S={score:.4f} (epoch={epoch + 1}) to {path}",
                flush=True,
            )

    def get_best_checkpoint_score_path(self) -> Path | None:
        path = self._checkpoint_dir() / "cvae_best_checkpoint_score.pth"
        return path if path.exists() else None

    def _log_wandb_training_summary(self) -> None:
        logger = self._experiment_logger
        if logger is None:
            return
        summary: dict[str, float | int] = {}
        if self._best_kmeans_epoch is not None:
            summary["val/best_kmeans_nmi"] = self._best_kmeans_nmi
            summary["val/best_kmeans_nmi_epoch"] = self._best_kmeans_epoch + 1
        if self._best_checkpoint_score_epoch is not None:
            summary["val/best_checkpoint_score"] = self._best_checkpoint_score
            summary["val/best_checkpoint_score_epoch"] = self._best_checkpoint_score_epoch + 1
        if summary:
            logger.update_summary(summary)

    def finalize_wandb(self, extra_summary: dict[str, float | int] | None = None) -> None:
        logger = self._experiment_logger
        if logger is None:
            return
        if extra_summary:
            logger.update_summary(extra_summary)
        logger.finish()
        self._experiment_logger = None

    def reset(self):
        self.current_epoch = 0
        self._best_kmeans_nmi = -1.0
        self._best_kmeans_epoch = None
        self._best_checkpoint_score = float("-inf")
        self._best_checkpoint_score_epoch = None
        if self.config.early_stopping:
            self.early_stopping = create_early_stopper(self.config, self.global_config.verbose)
        else:
            self.early_stopping = None
            
    def __print_training_start(self):
        print("\n" + "=" * 60)
        print(f"🚀 Starting Training ({self.config.epochs} Epochs)".center(60, "="))
        print("=" * 60 + "\n")

    def __print_epoch(self):
        epoch_str = f"Epoch {self.current_epoch + 1}/{self.config.epochs}"
        total_loss_str = f"Total Loss: {self.losses[self.current_epoch]:.7f}"
        loss_str = f"Loss: {self.losses[self.current_epoch]-self.regularization_losses[self.current_epoch]:.7f}"
        reg_loss_str = f"Reg Loss: {self.regularization_losses[self.current_epoch]:.7f}" if self.regularization_losses else "Reg Loss: 0.0000"
        
        print(f"{epoch_str:<15} | {total_loss_str} | {loss_str} | {reg_loss_str}", flush=True)

    def __str__(self) -> str:
        return f"Trainer(config={self.config})"