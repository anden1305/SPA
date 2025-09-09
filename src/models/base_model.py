
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from src.config.config import GlobalConfig
from src.data.base_dataset import BaseDataset
from src.data.data_loader import DataLoader
from src.orchestrator.train_details import TrainDetails


class BaseModel(nn.Module, ABC):
    """Common base class for models.

    Currently only enforces that subclasses implement `forward`.
    A concrete `__init__` is provided so calling `super().__init__()` is safe.
    """

    def __init__(self, data_loader: DataLoader, config: GlobalConfig, device: torch.device):
        self.global_config = config
        self.config = self.global_config.model
        self.dataset = data_loader.dataset
        self.data_loader = data_loader
        self.device = device
        self.__extract_model_parameters()
        super().__init__()
    
    def __extract_model_parameters(self):
        self.num_states = self.dataset.get_num_states()
        self.num_features = self.data_loader.get_feature_dim()

    ### TRAINING ###
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:  # pragma: no cover - interface
        raise NotImplementedError('This method has to be implemented.')
    
    @abstractmethod
    def prepare_for_training(self):
        raise NotImplementedError('This method has to be implemented.')
    
    @abstractmethod
    def reset(self):
        raise NotImplementedError('This method has to be implemented.')

    ### INFERENCE ###
    @abstractmethod
    def prepare_for_inference(self):
        raise NotImplementedError('This method has to be implemented.')
    
    @abstractmethod
    def predict(self, x: Tensor) -> Tensor:
        raise NotImplementedError('This method has to be implemented.')
    
    ### SAVE MODEL ###
    def save_info(self, train_details: TrainDetails):
        """Save model parameters to a file."""
        torch.save(self.state_dict(), train_details.get_path() / "model.pth")

    ### STRING REPRESENTATION ###
    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError('This method has to be implemented.')
    
    ### UTILITY METHODS ###
    @torch.no_grad()
    def clear_param_grads(self) -> None:
        for p in self.parameters():
            if p.grad is not None:
                p.grad.zero_()