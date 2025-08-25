
from abc import ABC, abstractmethod
import numpy as np

from scr.config.config import TransformsConfig


class BaseTransform(ABC):
    """Documentation

    Abstract class for all preprocessing methods that is needed for this codebase.
    """
    
    def __init__(self, config: TransformsConfig):
        self.config = config
        self.validate_config(self.config)
    
    @abstractmethod
    def validate_config(self, config: TransformsConfig):
        raise NotImplementedError('This method has to be implemented.')
    
    @abstractmethod
    def __call__(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Documentation
        Processes the data. Input and output should be of shape (N, T, C), 
        where N is samples, T is timestamps and C is channels.
        """
        raise NotImplementedError('This method has to be implemented.')
    
    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError('This method has to be implemented.')
    
    def __repr__(self):
        return self.__str__()