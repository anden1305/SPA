
from jinja2 import BaseLoader
import numpy as np
from scr.data.base_dataset import BaseDataset
from scr.preprocessing.base_transform import BaseTransform


class SyntheticLoader(BaseLoader):
    """Documentation

    Abstract class for all data loaders that are needed for this codebase.
    """
    
    def __init__(self, 
                 dataset: BaseDataset, 
                 batch_size: int,
                 shuffle: bool,
                 transforms: list[BaseTransform]):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transforms = transforms
    
    def __len__(self) -> int:
        return len(self.dataset) // self.batch_size

    def __iter__(self) -> "SyntheticLoader":
        """Handles every start of new epoch logic (shuffle etc.)."""
        self._size = len(self.dataset)
        self._indices = list(range(self._size))
        self._cursor = 0
        return self
    
    def __apply_transforms(self, sample):
        """Apply all transforms in order to a single sample."""
        for t in self.transforms:
            sample = t(sample)
        return sample
    
    def __next__(self) -> np.ndarray:
        """Loads, transforms and returns the next batch of data."""
        remaining = self._size - self._cursor
        if remaining < self.batch_size:
            raise StopIteration

        start = self._cursor
        end = start + self.batch_size
        batch_idx = self._indices[start:end]

        # Fetch samples
        samples = self.dataset[batch_idx]
        
        # Apply transforms sample-wise
        if self.transforms:
            samples = self.__apply_transforms(samples)

        # Advance cursor
        self._cursor = end
        return samples