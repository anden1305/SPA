
from abc import ABC, abstractmethod
import numpy as np
import yaml

from src.config.config import DatasetConfig, GlobalConfig

class BaseDataset(ABC):
    """Documentation
    
    Abstract class for all datasets that are needed for this codebase.
    """
    
    def __init__(self,
                 config: DatasetConfig):
        self.config = config
        self.id = self.config.id
        self.config = self.load_config()
        self.data = self.load_data()
        self.labels = self.load_labels()
        self.validate_config()
        self.validate_data()
        self.validate_labels()
    
    ###### BASE METHODS ######
    
    def validate_config(self):
        if not isinstance(self.config, dict):
            raise ValueError('Config must be a dictionary')
        assert 'name' in self.config, "Dataset name not found in config."
        assert 'n_channels' in self.config, "Number of channels not found in config."
        assert 'n_timesteps' in self.config, "Number of timesteps not found in config."
        assert 'sampling_rate' in self.config, "Sampling rate not found in config."
        assert 'epoch_length' in self.config, "Epoch length not found in config."
        assert 'n_stages' in self.config, "Number of stages not found in config."
        assert 'stage_names' in self.config, "Stage names not found in config."
    
    def validate_data(self):
        assert self.data.ndim == 2, f"Data must be a 2D array with shape (C, T), but got shape {self.data.ndim}."
        assert self.data.shape[0] == self.config['n_channels'], "Data channels do not match config."
        assert self.data.shape[1] == self.config['n_timesteps'], "Data timesteps do not match config."
    
    def validate_labels(self):
        assert hasattr(self, 'labels'), "Labels not found in dataset."
        assert self.labels.ndim == 1, "Labels must be a 1D array."
        assert self.labels.shape[0] == self.config['n_timesteps'], f"Labels length {self.labels.shape[0]} does not match number of timesteps {self.config['n_timesteps']}."
        assert len(np.unique(self.labels)) == self.config['n_stages'], f"Number of unique ({np.unique(self.labels)}) labels does not match number of stages ({self.config['n_stages']}) for {self}"
    
    def get_config(self):
        return self.config
    
    def save_config(self, path: str):
        with open(f"{path}/config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(
                self.config,
                f,
                sort_keys=False,
                allow_unicode=True,
                indent=2,
            )
    
    def __len__(self):
        return len(self.data[-1])

    def __getitem__(self, idx):
        return self.data[:, idx], self.labels[idx]
    
    def get_num_states(self):
        return self.config['n_stages']

    ###### ABSTRACT METHODS ######

    @abstractmethod
    def load_data(self) -> np.ndarray:
        raise NotImplementedError('This method has to be implemented.')
    
    @abstractmethod
    def load_labels(self) -> np.ndarray:
        raise NotImplementedError('This method has to be implemented.')
    
    @abstractmethod
    def load_config(self) -> dict:
        raise NotImplementedError('This method has to be implemented.')

    @abstractmethod
    def __str__(self):
        """Returns a string representation of the dataset."""
        raise NotImplementedError('This method has to be implemented.')
    
    def get_state_names(self):
        return self.config.get('stage_names', None)