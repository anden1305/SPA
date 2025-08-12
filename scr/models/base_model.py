
from abc import ABC, abstractmethod

import numpy as np
import torch.nn as nn
from torch import Tensor


class BaseModel(nn.Module, ABC):
    """Common base class for models.

    Currently only enforces that subclasses implement `forward`.
    A concrete `__init__` is provided so calling `super().__init__()` is safe.
    """

    def __init__(self) -> None:  # not abstract; allow safe super() chain
        super().__init__()

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:  # pragma: no cover - interface
        raise NotImplementedError('This method has to be implemented.')