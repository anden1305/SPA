
from typing import Iterator

import numpy as np
from scr.data.base_dataset import BaseDataset
from scr.config.config import DataLoaderConfig, GlobalConfig, TransformsConfig
from scr.preprocessing.base_transform import BaseTransform
import torch
from scr.preprocessing.collapse_dimensions import CollapseDimensions
from scr.preprocessing.fft import FFT

class DataLoader(Iterator):
    """Documentation

    Abstract class for all data loaders that are needed for this codebase.
    """
    
    def __init__(self, 
                 dataset: BaseDataset, 
                 config: GlobalConfig,
                 device: torch.device = torch.device("cpu")):
        self.global_config = config
        self.config = self.global_config.dataloader
        self.dataset = dataset
        self.batch_size = self.config.batch_size
        self.__init_transforms(transform_configs=self.config.transforms)
        self.device = device
        self.verbose = self.global_config.verbose
    
    def __init_transforms(self, transform_configs: list[TransformsConfig]):
        self.transforms: list[BaseTransform] = []
        for config in transform_configs:
            match config.type.lower():
                case "fft":
                    self.transforms.append(FFT(config=config))
                case "collapse_dimensions":
                    self.transforms.append(CollapseDimensions(config=config))
                case _:
                    raise ValueError(f"Unknown transform: {config.type}.")
    
    def __len__(self) -> int:
        """Number of batches per epoch."""
        return len(self.dataset) // self.batch_size
    
    def __apply_transforms(self, x: np.ndarray, y: np.ndarray):
        """Apply all transforms in order to a single sample."""
        for t in self.transforms:
            x, y = t(x, y)
        return x, y
    
    def __iter__(self) -> "DataLoader":
        """Handles every start of new epoch logic (shuffle etc.)."""
        self._size = len(self.dataset)
        self._indices = list(range(self._size))
        self._cursor = 0
        return self
    
    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Loads, transforms and returns the next batch of data."""
        remaining = self._size - self._cursor
        if remaining < self.batch_size:
            raise StopIteration
        start = self._cursor
        end = start + self.batch_size
        batch_idx = self._indices[start:end]
        x, y = self.dataset[batch_idx]
        if self.transforms:
            x, y = self.__apply_transforms(x, y)
        self._cursor = end
        return torch.from_numpy(x).to(self.device), torch.from_numpy(y).to(self.device)

    def get_feature_dim(self) -> int:
        x, y = self.dataset[:self.batch_size]
        if self.transforms:
            x, y = self.__apply_transforms(x, y)
        return x.shape[-1]

    def __str__(self) -> str:
        return f"DataLoader(transform={self.transforms})"