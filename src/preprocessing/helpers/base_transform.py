
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np

from src.config.config import TransformsConfig

class PreprocessingStage(Enum):
    PREPROCESSING = "preprocessing"
    POSTPROCESSING = "postprocessing"

class BaseTransform(ABC):
    """Documentation

    Abstract class for all preprocessing methods that is needed for this codebase.
    """
    
    def __init__(self, config: TransformsConfig, stage: str):
        self.config = config
        self.stage: PreprocessingStage = PreprocessingStage[stage.upper()]
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
    
    @abstractmethod
    def get_short_name(self) -> str:
        raise NotImplementedError('This method has to be implemented.')

    def get_stage(self) -> str:
        return self.stage.value

    def __repr__(self):
        return self.__str__()