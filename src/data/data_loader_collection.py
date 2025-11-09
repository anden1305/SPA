
import numpy as np
import torch
from src.config.config import GlobalConfig
from src.data.base_dataset import BaseDataset
from src.data.data_loader import DataLoader
from src.preprocessing.helpers.base_transform import BaseTransform
from concurrent.futures import ThreadPoolExecutor

class DataLoaderCollection:
    """
    A collection of DataLoader instances for training and validation datasets.
    """
    
    def __init__(self, 
                 datasets: list[BaseDataset], 
                 config: GlobalConfig,
                 device: torch.device = torch.device("cpu")):
        self.global_config = config
        self.config = self.global_config.dataloader
        self.datasets = datasets
        # self.data_loaders = [DataLoader(dataset=ds, config=self.global_config, device=device) for ds in self.datasets]
        self.__build_data_loaders_in_parallel(device)
        self.__validate_data()
        self.device = device
        self.shuffle = self.config.shuffle
        self.random_seed = self.global_config.seed
        self.x, self.y = self.prepare_data()
        self.batches_per_next = self.config.num_batches
    
    def prepare_data(self) -> tuple[np.ndarray, np.ndarray]:
        x = []
        y = []
        for dl in self.data_loaders:
            x_dl, y_dl = dl.get_data()
            x.append(x_dl)
            y.append(y_dl)
        x_all = np.concatenate(x, axis=0)
        y_all = np.concatenate(y, axis=0)
        return x_all, y_all
    
    def __build_data_loaders_in_parallel(self, device: torch.device):
        with ThreadPoolExecutor(max_workers=20) as ex:
            self.data_loaders = list(
                ex.map(lambda ds: DataLoader(dataset=ds, config=self.global_config, device=device),
                        self.datasets)
            )
    
    def __validate_data(self):
        num_states = set([ds.get_num_states() for ds in self.datasets])
        assert len(num_states) == 1, "All datasets must have the same number of states."
        self.num_states = num_states.pop()
        feature_dims = set([dl.get_feature_dim() for dl in self.data_loaders])
        assert len(feature_dims) == 1, "All data loaders must have the same feature dimension."
        self.feature_dim = feature_dims.pop()
        self.state_names = self.datasets[0].get_state_names()
    
    def get_transforms(self) -> list[BaseTransform]:
        return self.data_loaders[0].transforms
    
    def get_num_states(self) -> int:
        return self.num_states
    
    def get_feature_dim(self) -> int:
        return self.feature_dim
    
    def get_state_names(self) -> list[str]:
        return self.state_names

    def get_feature_names(self) -> list[str]:
        return self.data_loaders[0].get_feature_names()

    def get_all_data(self):
        xs = []
        ys = []
        for dl in self.data_loaders:
            x, y = dl.get_all_data()
            xs.append(x)
            ys.append(y)
        x_all = torch.cat(xs, dim=1)
        y_all = torch.cat(ys, dim=1)
        
        return x_all, y_all
    
    def has_features_enabled(self):
        return self.data_loaders[0].has_features_enabled()
    
    def __iter__(self) -> "DataLoaderCollection":
        # Reset cursor and (optionally) shuffle order for a new pass
        self._num_batches = int(self.x.shape[0])
        self._cursor = 0
        # Track epochs to vary shuffle across iterations if desired
        if not hasattr(self, "_epoch"):
            self._epoch = 0
        # Build index order using numpy (self.x/self.y are numpy arrays)
        if self.shuffle:
            # Deterministic RNG based on seed + epoch when seed is provided
            if self.random_seed is not None:
                rng = np.random.default_rng(int(self.random_seed) + int(self._epoch))
            else:
                rng = np.random.default_rng()
            self._order = rng.permutation(self._num_batches)
        else:
            self._order = np.arange(self._num_batches)
        self._epoch += 1
        return self
    
    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Stop when we've consumed all batches
        if not hasattr(self, "_cursor"):
            # Support calling next() without an explicit iter() first
            _ = iter(self)
        if self._cursor >= self._num_batches:
            raise StopIteration
        # Compute slice of indices for this step
        start = self._cursor
        end = min(start + int(self.batches_per_next), self._num_batches)
        idx = self._order[start:end]
        # Gather with numpy indexing; convert to torch tensors only at return time
        x_np = self.x[idx]
        y_np = self.y[idx]
        self._cursor = end
        x_t = torch.as_tensor(x_np, device=self.device)
        y_t = torch.as_tensor(y_np, device=self.device)
        return x_t, y_t
    
    def __str__(self) -> str:
        return (f"DataLoaderCollection(\n"
                f"  num_datasets={len(self.datasets)},\n"
                f"  num_states={self.num_states},\n"
                f"  feature_dim={self.feature_dim},\n"
                f"  state_names={self.state_names}\n"
                f")")