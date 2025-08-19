
from scr.data.base_dataset import BaseDataset
import yaml
import numpy as np

class SyntheticDataset(BaseDataset):
    """Documentation
    
    Abstract class for all datasets that are needed for this codebase.
    """
    
    def __init__(self, path: str):
        self.path = path
        self.__initialize()
    
    def __initialize(self):
        self.__load_config()
        self.data: np.ndarray = np.load(self.path + "/eeg.npy")
        if self.data.ndim == 1:
            self.data = self.data[np.newaxis, :]
        self.raw_labels: np.ndarray = np.load(self.path + "/labels.npy")
        self.labels: np.ndarray = np.repeat(self.raw_labels, self.sampling_rate * self.epoch_length, axis=0)

    def __load_config(self):
        with open(self.path + "/config_copy.yml", 'r') as file:
            self.config = yaml.safe_load(file)
        assert self.config is not None, "Config file is empty or not found."
        assert 'name' in self.config, "Dataset name not found in config."
        assert 'sampling_rate_hz' in self.config, "Sampling rate not found in config."
        assert 'epoch_length_s' in self.config, "Epoch length not found in config."
        assert 'n_stages' in self.config, "Number of stages not found in config."
        assert 'stage_names' in self.config, "Stage names not found in config."
        self.name = self.config['name']
        self.sampling_rate = self.config['sampling_rate_hz']
        self.epoch_length = self.config['epoch_length_s']
        self.n_stages = self.config['n_stages']
        self.stage_names = self.config['stage_names']

    def __len__(self):
        """Number of samples in dataset (T)."""
        return len(self.data[-1])
    
    def __getitem__(self, idx):
        """Loads and returns the next samples in the dataset."""
        """Returns data as (x, y), where x is of shape (C, T) and y is of shape (T)"""
        if isinstance(idx, (int, np.integer)):
            if idx < 0 or idx >= len(self):
                raise IndexError("index out of range")
        x = self.data[:, idx]
        y = self.labels[idx] if self.labels is not None else None
        return (x, y)