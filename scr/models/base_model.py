
from abc import ABC, abstractmethod

import numpy as np
import torch.nn as nn
from torch import Tensor


class BaseModel(nn.Module, ABC):
    """Documentation
    
    Abstract class for all models that are needed for this codebase.
    """
    
    @abstractmethod
    def __init__(self):
        super().__init__()
        raise NotImplementedError('This method has to be implemented.')
    
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError('This method has to be implemented.')