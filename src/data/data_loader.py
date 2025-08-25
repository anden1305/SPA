
from typing import Iterator

import numpy as np
from src.data.base_dataset import BaseDataset
from src.config.config import GlobalConfig, TransformsConfig
from src.preprocessing.base_transform import BaseTransform
import torch
from src.preprocessing.fft import FFT
from src.preprocessing.reshape import Reshape

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
        self.seed = self.global_config.seed
        self.shuffle = self.config.shuffle
        self.__init_transforms(transform_configs=self.config.transforms)
        self.device = device
        self.verbose = self.global_config.verbose
        self._epoch = 0
    
    def __init_transforms(self, transform_configs: list[TransformsConfig]):
        self.transforms: list[BaseTransform] = []
        for config in transform_configs:
            match config.type.lower():
                case "fft":
                    self.transforms.append(FFT(config=config))
                case "reshape":
                    self.transforms.append(Reshape(config=config))
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
        self._num_batches = self._size // self.batch_size
        batch_starts = [i * self.batch_size for i in range(self._num_batches)]
        if self.shuffle:
            rs = np.random.RandomState(self.seed + self._epoch)
            batch_starts = rs.permutation(batch_starts).tolist()
        self._batch_starts = batch_starts
        self._batch_cursor = 0
        self._epoch += 1
        return self
    
    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Loads, transforms and returns the next batch of data."""
        if self._batch_cursor >= getattr(self, "_num_batches", 0):
            raise StopIteration
        start = self._batch_starts[self._batch_cursor]
        end = start + self.batch_size
        batch_idx = list(range(start, end))
        x, y = self.dataset[batch_idx]
        if self.transforms:
            x, y = self.__apply_transforms(x, y)
        self._batch_cursor += 1
        x = x[np.newaxis, :]
        return torch.from_numpy(x).to(self.device), torch.from_numpy(y).to(self.device)

    def get_feature_dim(self) -> int:
        x, y = self.dataset[:self.batch_size]
        if self.transforms:
            x, y = self.__apply_transforms(x, y)
        return x.shape[-1]
    
    def get_all_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        xs, ys = [], []
        for xb, yb in self:
            xs.append(xb)
            ys.append(yb)
        x = torch.concat(xs, dim=0)
        y = torch.concat(ys, dim=0)
        return x,y

    def __str__(self) -> str:
        return f"DataLoader(transform={self.transforms})"