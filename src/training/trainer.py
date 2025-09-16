import json
from src.data.data_loader import DataLoader
from src.models.base_model import BaseModel
from src.validation.validator import Validator
from src.config.config import GlobalConfig, TrainerConfig
from src.training.early_stopping import create_early_stopper
from src.training.validation_controller import ValidationController
from torch.optim import Adam, SGD, RMSprop
import torch

class Trainer:
    
    def __init__(self,
                 data_loader: DataLoader,
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
        self.validation_controller = ValidationController(validator=self.validator, early_stopper=self.early_stopping, config_trainer=self.config, global_verbose=self.global_config.verbose,)
    
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
        self.__init_training()
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            for x, _ in self.data_loader:
                self.optimizer.zero_grad()
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
            # Validation + early stopping via controller (prints internally if stopping)
            should_stop = self.validation_controller.step(epoch, self.model, self.optimizer)
            if should_stop:
                break
            self.epoch_losses.clear()
            self.epoch_regularization_losses.clear()
            if self.global_config.verbose:
                self.__print_epoch()
    
    def get_losses(self) -> dict[int, float]:
        assert self.losses, "Training has not been run yet."
        return self.losses
    
    def reset(self):
        self.current_epoch = 0
        if self.config.early_stopping:
            self.early_stopping = create_early_stopper(self.config, self.global_config.verbose)
        else:
            self.early_stopping = None
        self.validation_controller = ValidationController(validator=self.validator, early_stopper=self.early_stopping, config_trainer=self.config, global_verbose=self.global_config.verbose)
            
    def __print_training_start(self):
        print("\n" + "=" * 60)
        print(f"🚀 Starting Training ({self.config.epochs} Epochs)".center(60, "="))
        print("=" * 60 + "\n")

    def __print_epoch(self):
        epoch_str = f"Epoch {self.current_epoch + 1}/{self.config.epochs}"
        total_loss_str = f"Total Loss: {self.losses[self.current_epoch]:.4f}"
        loss_str = f"Loss: {self.losses[self.current_epoch]-self.regularization_losses[self.current_epoch]:.4f}"
        reg_loss_str = f"Reg Loss: {self.regularization_losses[self.current_epoch]:.4f}" if self.regularization_losses else "Reg Loss: 0.0000"
        
        print(f"{epoch_str:<15} | {total_loss_str} | {loss_str} | {reg_loss_str}", flush=True)

    def __str__(self) -> str:
        return f"Trainer(config={self.config})"