
import json
from src.data.data_loader import DataLoader
from src.models.base_model import MLModel
from src.config.config import GlobalConfig, TrainerConfig
from torch.optim import Adam, SGD, RMSprop
import torch

class Trainer:
    
    def __init__(self,
                 data_loader: DataLoader,
                 model: MLModel,
                 config: GlobalConfig):
        self.data_loader = data_loader
        self.model = model
        self.global_config = config
        self.config = self.global_config.trainer
        self.losses: dict[int, float] = {}
        self.current_epoch = 0

    def __init_optimizer(self):
        if self.config.optimizer == "adam":
            self.optimizer = Adam(self.model.parameters(), lr=self.config.learning_rate)
        elif self.config.optimizer == "sgd":
            self.optimizer = SGD(self.model.parameters(), lr=self.config.learning_rate)
        elif self.config.optimizer == "rmsprop":
            self.optimizer = RMSprop(self.model.parameters(), lr=self.config.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")

    def __init_training(self):
        self.__init_optimizer()
        self.epoch_losses: list[float] = []
        self.losses: dict[int, float] = {}
        self.model.prepare_for_training()
        self.current_epoch = 0
    
    def train(self):
        if self.global_config.verbose:
            self.__print_training_start()
        self.__init_training()
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            for x, y in self.data_loader:
                self.optimizer.zero_grad()
                logp = self.model.forward(x)
                loss = -logp.mean()
                loss.backward()
                if self.config.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()
                self.epoch_losses.append(loss.item())
            self.losses[epoch] = sum(self.epoch_losses) / len(self.epoch_losses)
            self.epoch_losses.clear()
            if self.global_config.verbose:
                self.__print_epoch()
    
    def get_losses(self) -> dict[int, float]:
        assert self.losses, "Training has not been run yet."
        return self.losses
    
    def reset(self):
        self.current_epoch = 0
    
    def __print_training_start(self):
        print("\n" + "=" * 60)
        print(f"🚀 Starting Training ({self.config.epochs} Epochs)".center(60, "="))
        print("=" * 60 + "\n")

    def __print_epoch(self):
        epoch_str = f"Epoch {self.current_epoch + 1}/{self.config.epochs}"
        loss_str = f"Loss: {self.losses[self.current_epoch]:.4f}"
        print(f"{epoch_str:<15} | {loss_str}", flush=True)

    def __str__(self) -> str:
        return f"Trainer(config={self.config})"