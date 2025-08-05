
from abc import ABC, abstractmethod
from typing import Iterator
import numpy as np
from scr.data.base_dataset import BaseDataset
from scr.preprocessing.base_transform import BaseTransform


class BaseLoader(ABC, Iterator):
    """Documentation

    Abstract class for all data loaders that are needed for this codebase.
    """
    
    @abstractmethod
    def __init__(self, 
                 dataset: BaseDataset, 
                 batch_size: int, 
                 shuffle: bool,
                 transforms: list[BaseTransform]):
        raise NotImplementedError('This method has to be implemented.')
    
    @abstractmethod
    def __len__(self) -> int:
        """Number of batches per epoch."""
        raise NotImplementedError('This method has to be implemented.')
    
    @abstractmethod
    def __iter__(self) -> "BaseLoader":
        """Handles every start of new epoch logic (shuffle etc.)."""
        raise NotImplementedError('This method has to be implemented.')
    
    @abstractmethod
    def __next__(self) -> np.ndarray:
        """Loads, preprocesses and returns the next batch of data."""
        raise NotImplementedError('This method has to be implemented.')

