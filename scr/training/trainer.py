
from scr.data.data_loader import DataLoader
from scr.models.base_model import BaseModel
from scr.training.config import TrainingConfig
from torch.optim import Adam, SGD, RMSprop
import torch

class Trainer:
    
    def __init__(self,
                 data_loader: DataLoader,
                 model: BaseModel,
                 config: TrainingConfig):
        self.data_loader = data_loader
        self.model = model
        self.config = config
        self.__init_optimizer()
        
    def __init_optimizer(self):
        if self.config.optimizer == "adam":
            self.optimizer = Adam(self.model.parameters(), lr=self.config.learning_rate)
        elif self.config.optimizer == "sgd":
            self.optimizer = SGD(self.model.parameters(), lr=self.config.learning_rate)
        elif self.config.optimizer == "rmsprop":
            self.optimizer = RMSprop(self.model.parameters(), lr=self.config.learning_rate)

    def __init_training(self):
        self.losses: list[float] = []

    def train(self):
        if self.config.verbose:
            print(f"Starting training for {self.model} with {self.config.epochs} epochs on {self.data_loader}...")
        for epoch in range(self.config.epochs):
            if self.config.verbose:
                print(f"Epoch {epoch + 1}/{self.config.epochs}")
            for i, (x, y) in enumerate(self.data_loader):
                if self.config.verbose:
                    print(f"Epoch {epoch + 1}/{self.config.epochs} | Batch {i + 1}/{len(self.data_loader)}")
                self.model.train()
                self.optimizer.zero_grad()
                logp = self.model.forward(x)
                loss = -logp.mean()
                if self.config.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()
                self.losses.append(loss.item())
                
                

        # model.train()
		# optimizer.zero_grad()
		# logp = model(batch_x)
		# loss = -logp.mean()
		# loss.backward()
		# if args.grad_clip is not None:
		# 	torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
		# optimizer.step()
		# print(f"Epoch {epoch}/{args.epochs} | negNLL={loss.item():.4f}")
		# losses.append(loss.item())