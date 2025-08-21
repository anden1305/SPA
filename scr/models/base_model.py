
from abc import ABC, abstractmethod

import numpy as np
import torch.nn as nn
from torch import Tensor

from scr.config.config import GlobalConfig
from scr.data.base_dataset import BaseDataset
from scr.data.data_loader import DataLoader


class MLModel(nn.Module, ABC):
    """Common base class for models.

    Currently only enforces that subclasses implement `forward`.
    A concrete `__init__` is provided so calling `super().__init__()` is safe.
    """

    def __init__(self, data_loader: DataLoader, config: GlobalConfig):
        self.global_config = config
        self.config = self.global_config.model
        self.dataset = data_loader.dataset
        self.data_loader = data_loader
        self.__extract_model_parameters()
        super().__init__()
    
    def __extract_model_parameters(self):
        self.num_states = self.dataset.get_num_states()
        self.num_features = self.data_loader.get_feature_dim()

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:  # pragma: no cover - interface
        raise NotImplementedError('This method has to be implemented.')
    
    @abstractmethod
    def prepare_for_training(self):
        raise NotImplementedError('This method has to be implemented.')
    
    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError('This method has to be implemented.')