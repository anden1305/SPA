
import copy
import re
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
                 for_validation: bool = False,
                 device: torch.device = torch.device("cpu")):
        self.global_config = copy.deepcopy(config)
        self.config = copy.deepcopy(self.global_config.dataloader)
        if for_validation:
            self.global_config.dataloader.batch_size = self.global_config.dataloader.validation_batch_size
            self.config.batch_size = self.global_config.dataloader.validation_batch_size
        self.datasets = datasets
        # self.data_loaders = [DataLoader(dataset=ds, config=self.global_config, device=device) for ds in self.datasets]
        self.__build_data_loaders_in_parallel(device)
        self.__validate_data()
        self.device = device
        self.shuffle = self.config.shuffle
        self.random_seed = self.global_config.seed
        self.subject_map = self.get_subject_map()
        self.pre_load_to_device = True
        self.x, self.y, self.sub_ids = self.prepare_data()
        self.batches_per_next = self.config.num_batches
        
    def prepare_data(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = []
        y = []
        sub_ids = []
        for dl in self.data_loaders:
            x_dl, y_dl = dl.get_data()
            x.append(x_dl)
            y.append(y_dl)
            
            # Create sub_ids array matching x_dl shape (except feature dim)
            sub_id = self.subject_map[dl.dataset.get_id()]
            # x_dl shape: (Batch, Time, Feature) -> sub_ids shape: (Batch, Time, 1)
            sub_id_arr = np.full((x_dl.shape[0], x_dl.shape[1]), sub_id, dtype=np.int64)
            sub_ids.append(sub_id_arr)
        
        x_all = np.concatenate(x, axis=0)
        y_all = np.concatenate(y, axis=0)
        sub_ids_all = np.concatenate(sub_ids, axis=0)
        
        # normalize
        if self.global_config.cvae.normalize_global:
            mean = np.mean(x_all, axis=(0,1), keepdims=True)
            std = np.std(x_all, axis=(0,1), keepdims=True) + 1e-8
            x_all = (x_all - mean) / std
        
        # save as torch
        x_all = torch.from_numpy(x_all)
        y_all = torch.from_numpy(y_all)
        sub_ids_all = torch.from_numpy(sub_ids_all)
        
        # move to device and pin memory if cuda
        if self.device.type == "cuda" and self.pre_load_to_device:
            x_all = x_all.pin_memory()
            y_all = y_all.pin_memory()
            sub_ids_all = sub_ids_all.pin_memory()
            x_all = x_all.to(self.device, non_blocking=True)
            y_all = y_all.to(self.device, non_blocking=True)
            sub_ids_all = sub_ids_all.to(self.device, non_blocking=True)
        return x_all, y_all, sub_ids_all
    
    def __build_data_loaders_in_parallel(self, device: torch.device):
        with ThreadPoolExecutor(max_workers=20) as ex:
            self.data_loaders = list(
                ex.map(lambda ds: DataLoader(dataset=ds, config=self.global_config, device=device),
                        self.datasets)
            )
    
    def __validate_data(self):
        num_states = set([ds.get_num_states() for ds in self.datasets])
        # assert len(num_states) == 1, "All datasets must have the same number of states."
        # self.num_states = num_states.pop()
        self.num_states = max(num_states)
        feature_dims = set([dl.get_feature_dim() for dl in self.data_loaders])
        assert len(feature_dims) == 1, "All data loaders must have the same feature dimension."
        self.feature_dim = feature_dims.pop()
        self.state_names = self.datasets[0].get_state_names()
        assert len(self.state_names) == self.num_states, "Number of state names must match number of states."
     
    def get_transforms(self) -> list[BaseTransform]:
        return self.data_loaders[0].transforms
    
    def get_num_states(self) -> int:
        return self.num_states
    
    def get_feature_dim(self) -> int:
        return self.feature_dim
    
    def get_vae_dims(self) -> tuple[int, int, int]:
        data_dim = self.data_loaders[0].data[0].shape
        channels = data_dim[-2]
        features = data_dim[-1]
        sequence_length = data_dim[-3]
        return channels, features, sequence_length, self.num_states
    
    def get_state_names(self) -> list[str]:
        return self.state_names

    def get_feature_names(self) -> list[str]:
        return self.data_loaders[0].get_feature_names()

    def get_all_data(self):
        if self.device.type == "cuda" and not self.pre_load_to_device:
            return self.x.to(self.device, non_blocking=True), self.y.to(self.device, non_blocking=True), self.sub_ids.to(self.device, non_blocking=True)
        return self.x, self.y, self.sub_ids
    
    def has_features_enabled(self):
        return self.data_loaders[0].has_features_enabled()
    
    def get_num_subjects(self) -> int:
        return 92
    
    def get_subject_map(self) -> list[int]:
        # Returns list mapping subject ID to index
        subject_ids = set([ds.get_id() for ds in self.datasets])
        subject_to_id = {subject_id: int(re.search(r'\d+', subject_id).group()) for subject_id in subject_ids}
        return subject_to_id
    
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
        self._cursor = end
        if self.device.type == "cuda" and not self.pre_load_to_device:
            return (self.x[idx].to(self.device, non_blocking=True),
                    self.y[idx].to(self.device, non_blocking=True),
                    self.sub_ids[idx].to(self.device, non_blocking=True))
        return self.x[idx], self.y[idx], self.sub_ids[idx]
    
    def __str__(self) -> str:
        return (f"DataLoaderCollection(\n"
                f"  num_datasets={len(self.datasets)},\n"
                f"  num_states={self.num_states},\n"
                f"  feature_dim={self.feature_dim},\n"
                f"  state_names={self.state_names}\n"
                f")")