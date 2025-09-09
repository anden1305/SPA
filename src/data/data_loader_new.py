
from typing import Iterator

import numpy as np
from src.data.base_dataset import BaseDataset
from src.config.config import GlobalConfig, TransformsConfig
from src.preprocessing.base_transform import BaseTransform
import torch
from src.preprocessing.batch_raw import BatchRaw
from src.preprocessing.fft import FFT
from src.preprocessing.high_pass_filter import HighPassFilter
from src.preprocessing.percentile_clipping import PercentileClipping
from src.preprocessing.reshape import Reshape

class DataLoaderNew(Iterator):
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
        self.normalize = self.config.normalize
        self.__init_transforms(transform_configs=self.config.transforms)
        self.device = device
        self.verbose = self.global_config.verbose
        self._epoch = 0
        self.__process_data()
    
    def __process_data(self):
        # get data
        x, y = self.dataset[:]
        x, y = self.__apply_transforms(x, y)
        # normalize data
        if self.normalize:
            x = (x - x.mean(axis=(0,1))) / x.std(axis=(0,1))
        # set batch size
        if self.batch_size is None:
            batch_size = x.shape[0]
        else:
            batch_size = self.batch_size
        num_full_batches = x.shape[0] // batch_size
        if num_full_batches == 0:
            raise ValueError(f"Dataset size {x.shape[0]} is smaller than batch size {batch_size}.")
        # trim data to have full batches only
        x = x[:num_full_batches * batch_size]
        y = y[:num_full_batches * batch_size * x.shape[1]]
        # reshape data so that each batch is a continuous segment of the original data
        x = x.reshape((num_full_batches, batch_size*x.shape[1], x.shape[2]))
        y = y.reshape((num_full_batches, y.shape[0] // num_full_batches))
        # make data contiguous in memory
        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        # store data
        self.data = (x, y)
        self.__print_data_info()

    def __init_transforms(self, transform_configs: list[TransformsConfig]):
        self.transforms: list[BaseTransform] = []
        for config in transform_configs:
            match config.type.lower():
                case "fft":
                    self.transforms.append(FFT(config=config))
                case "reshape":
                    self.transforms.append(Reshape(config=config))
                case "percentile_clipping":
                    self.transforms.append(PercentileClipping(config=config))
                case "high_pass_filter":
                    self.transforms.append(HighPassFilter(config=config))
                case "batch_raw":
                    self.transforms.append(BatchRaw(config=config))
                case _:
                    raise ValueError(f"Unknown transform: {config.type}.")
    
    def __len__(self) -> int:
        """Number of batches per epoch."""
        return self.data[0].shape[0]
    
    def __apply_transforms(self, x: np.ndarray, y: np.ndarray):
        """Apply all transforms in order to a single sample."""
        for t in self.transforms:
            x, y = t(x, y)
        return x, y
    
    def __iter__(self) -> "DataLoaderNew":
        """Handles every start of new epoch logic (shuffle etc.)."""
        self._size = self.data[0].shape[0]
        self._num_batches = self._size
        batch_order = [i for i in range(self._num_batches)]
        if self.shuffle:
            rs = np.random.RandomState(self.seed + self._epoch)
            batch_order = rs.permutation(batch_order).tolist()
        self._batch_order = batch_order
        self._batch_cursor = 0
        self._epoch += 1
        return self
    
    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Loads, transforms and returns the next batch of data."""
        if self._batch_cursor >= getattr(self, "_num_batches", 0):
            raise StopIteration
        x = self.data[0][self._batch_order[self._batch_cursor]]
        y = self.data[1][self._batch_order[self._batch_cursor]]
        x = np.expand_dims(x, axis=0)
        self._batch_cursor += 1
        return torch.from_numpy(x).to(self.device), torch.from_numpy(y).to(self.device)

    def get_feature_dim(self) -> int:
        return self.data[0].shape[2]
    
    def get_all_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.data
        y = y.reshape(-1)
        x = x.reshape(-1, x.shape[2])
        x = np.expand_dims(x, axis=0)
        y = np.expand_dims(y, axis=0)
        return torch.from_numpy(x).to(self.device), torch.from_numpy(y).to(self.device)

    def __print_data_info(self):
        print("\n" + "=" * 60)
        print("📊 Data Info".center(60, "="))
        print("=" * 60)
        print(f"Dataset: {self.dataset}")
        print(f"Number of samples: {self.data[0].shape[0] * self.data[0].shape[1]}")
        print(f"Number of batches per epoch: {self.__len__()}")
        print(f"Batch size: {self.batch_size}")
        print(f"Feature dimension: {self.get_feature_dim()}")
        print(f"Shuffle each epoch: {self.shuffle}")
        print(f"Normalize data: {self.normalize}")
        print(f"Device: {self.device}")
        print(f"Final data shape: {self.data[0].shape}")
        if self.transforms:
            print("Transforms:")
            for t in self.transforms:
                print(f" - {t}")
        else:
            print("No transforms applied.")
        print("=" * 60 + "\n")

    def __str__(self) -> str:
        return f"DataLoader(transform={self.transforms})"