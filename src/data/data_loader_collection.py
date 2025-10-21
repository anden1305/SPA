
import torch
from src.config.config import GlobalConfig
from src.data.base_dataset import BaseDataset
from src.data.data_loader import DataLoader
from src.preprocessing.base_transform import BaseTransform


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
        self.data_loaders = [DataLoader(dataset=ds, config=self.global_config, device=device) for ds in self.datasets]
        self.__validate_data()
        self.device = device
        self.shuffle = self.config.shuffle
        self.random_seed = self.global_config.seed
        self._epoch = 0
        self._current_loader = 0
        self._current_batch = 0
        
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
    
    def __iter__(self) -> "DataLoaderCollection":
        ## shuffle data loaders at the start of each epoch if required
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.random_seed + self._epoch)
            indices = torch.randperm(len(self.data_loaders), generator=generator).tolist()
            self.data_loaders = [iter(self.data_loaders[i]) for i in indices]
        self._epoch += 1
        self._current_loader = 0
        self._current_batch = 0
        return self
    
    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the next batch from the current DataLoader. Moves to the next DataLoader when the current one is exhausted."""
        if self._current_loader >= len(self.data_loaders):
            raise StopIteration
        current_loader = self.data_loaders[self._current_loader]
        try:
            batch = next(current_loader)
            self._current_batch += 1
            return batch
        except StopIteration:
            self._current_loader += 1
            self._current_batch = 0
            return self.__next__()

    def __str__(self) -> str:
        return (f"DataLoaderCollection(\n"
                f"  num_datasets={len(self.datasets)},\n"
                f"  num_states={self.num_states},\n"
                f"  feature_dim={self.feature_dim},\n"
                f"  state_names={self.state_names}\n"
                f")")