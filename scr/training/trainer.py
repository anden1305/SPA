
from scr.data.data_loader import DataLoader
from scr.models.base_model import MLModel
from scr.config.config import GlobalConfig, TrainerConfig
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
        self.losses: list[float] = []
        x, _ = next(iter(self.data_loader))
        self.model.prepare_for_training(data=x)
    
    def train(self):
        self.__init_training()
        if self.global_config.verbose:
            print(f"Starting training for {self.model} with {self.config.epochs} epochs on {self.data_loader}...")
        for epoch in range(self.config.epochs):
            if self.global_config.verbose:
                print(f"Epoch {epoch + 1}/{self.config.epochs}")
            for x, _ in self.data_loader:
                self.model.train()
                self.optimizer.zero_grad()
                logp = self.model.forward(x)
                loss = -logp.mean()
                loss.backward()
                if self.config.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()
                self.epoch_losses.append(loss.item())
            if self.global_config.verbose:
                print(f"Epoch {epoch + 1} loss: {sum(self.epoch_losses) / len(self.epoch_losses):.4f}")
            self.losses.append(sum(self.epoch_losses) / len(self.epoch_losses))
            self.epoch_losses.clear()

    def __str__(self) -> str:
        return f"Trainer(config={self.config})"