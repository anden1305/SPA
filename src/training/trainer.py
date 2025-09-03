
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

    def __init_training(self):
        self.__init_optimizer()
        self.epoch_losses: list[float] = []
        self.losses: dict[int, float] = {}
        self.model.prepare_for_training()
        self.current_epoch = 0
    
    def train(self):
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
            if self.global_config.verbose:
                print(f"Epoch {epoch + 1} / {self.config.epochs} loss: {sum(self.epoch_losses) / len(self.epoch_losses):.4f}")
                # print(f"Dimension of x: {x.shape}")
                # print(f"Dimension of y: {y.shape}")
            self.losses[epoch] = sum(self.epoch_losses) / len(self.epoch_losses)
            self.epoch_losses.clear()
    
    def get_losses(self) -> dict[int, float]:
        assert self.losses, "Training has not been run yet."
        return self.losses

    def __str__(self) -> str:
        return f"Trainer(config={self.config})"