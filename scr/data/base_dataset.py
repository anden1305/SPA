
from abc import ABC, abstractmethod


class BaseDataset(ABC):
    """Documentation
    
    Abstract class for all datasets that are needed for this codebase.
    """
    
    @abstractmethod
    def __init__(self):
        raise NotImplementedError('This method has to be implemented.')
    
    @abstractmethod
    def __len__(self):
        """Number of samples in dataset."""
        raise NotImplementedError('This method has to be implemented.')
    
    @abstractmethod
    def __getitem__(self):
        """Loads and returns the next samples in the dataset."""
        """Returns data as (x, y), where x is of shape (C, T) and y is of shape (T)"""
        raise NotImplementedError('This method has to be implemented.')
    