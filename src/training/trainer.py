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
        logger.start(
            run_name=self.global_config.run_name,
            config=self.global_config.model_dump(),
        )
        self.__init_training()
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            for x, _, sub_ids in self.data_loader:
                self.optimizer.zero_grad()
                if isinstance(self.model, CVAEMARHMM):
                    loss, reg_loss = self.model.forward(x, sub_ids, epoch)
                else:
                    loss = self.model.forward(x)
                    reg_loss = self.model.regularization_loss()
                loss = loss + reg_loss
                loss.backward()
                if self.config.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()
                self.epoch_losses.append(loss.item())
                self.epoch_regularization_losses.append(reg_loss.item())
            if self.scheduler:
                    self.scheduler.step()
            self.losses[epoch] = sum(self.epoch_losses) / len(self.epoch_losses)
            self.regularization_losses[epoch] = sum(self.epoch_regularization_losses) / len(self.epoch_regularization_losses)
            if self.config.validate_per_epoch > 0 and (epoch + 1) % self.config.validate_per_epoch == 0:
                if isinstance(self.model, CVAEMARHMM) and self.model.training_pipeline == 'cvae':
                    self.validator.validate_cvae_epoch(epoch)
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
        logger.finish()
    
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
    
    def reset(self):
        self.current_epoch = 0
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