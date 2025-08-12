
from abc import ABC, abstractmethod

import numpy as np
import torch.nn as nn
from torch import Tensor
import torch


class BaseLoss(ABC):
    """Documentation
    
    Abstract class for all loss functions that are needed for this codebase.
    """
    
    @abstractmethod
    def __init__(self):
        raise NotImplementedError('This method has to be implemented.')
    
    @abstractmethod
    def __call__(self, out: Tensor, target: Tensor) -> Tensor:
        raise NotImplementedError('This method has to be implemented.')