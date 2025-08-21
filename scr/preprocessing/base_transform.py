
from abc import ABC, abstractmethod
import numpy as np


class BaseTransform(ABC):
    """Documentation

    Abstract class for all preprocessing methods that is needed for this codebase.
    """
    
    def __init__(self, params: dict):
        self.params = params
    
    @abstractmethod
    def __call__(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Documentation
        Processes the data. Input and output should be of shape (N, T, C), 
        where N is samples, T is timestamps and C is channels.
        """
        raise NotImplementedError('This method has to be implemented.')