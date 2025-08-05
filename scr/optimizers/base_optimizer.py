
from abc import ABC, abstractmethod
from typing import Any, Iterable

import numpy as np
import torch.nn as nn
from torch import Tensor


class BaseOptimizer(ABC):
    """Documentation
    
    Abstract class for all optimizers that are needed for this codebase.
    """
    
    @abstractmethod
    def __init__(self, params: Iterable[Any], hyper_params: dict):
        raise NotImplementedError('This method has to be implemented.')
    
    @abstractmethod
    def step(self) -> None:
        raise NotImplementedError('This method has to be implemented.')
    
    @abstractmethod
    def zero_grad(self) -> None:
        raise NotImplementedError('This method has to be implemented.')
    
    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        raise NotImplementedError('This method has to be implemented.')
    
    @abstractmethod
    def load_state_dict(self, state: dict[str, Any]) -> None:
        raise NotImplementedError('This method has to be implemented.')