
from src.config.config import DatasetConfig, GlobalConfig
from src.data.base_dataset import BaseDataset
import yaml
import numpy as np

class SyntheticDataset(BaseDataset):
    """Documentation
    
    Dataset class for the synthetic EEG sleep data.
    """
    
    BASE_PATH = 'data/synthetic_data'
    
    def __init__(self, config: DatasetConfig):
        self.data_path = f'{self.BASE_PATH}/{config.id}'
        super().__init__(config=config)

    def load_data(self):
        data = np.load(f'{self.data_path}/eeg.npy')
        data = data[np.newaxis, :]
        return data

    def load_labels(self):
        epoch_labels = np.load(f'{self.data_path}/labels.npy')
        labels: np.ndarray = np.repeat(epoch_labels, self.config['sampling_rate'] * self.config['epoch_length'], axis=0)
        return labels
    
    def load_config(self):
        config = {}
        with open(f'{self.data_path}/config_copy.yml', 'r') as file:
            synthetic_config = yaml.safe_load(file)
        config['name'] = synthetic_config['name']
        config['n_channels'] = 1
        config['channels'] = ['EEG']
        config['n_timesteps'] = synthetic_config['sampling_rate_hz'] * synthetic_config['epoch_length_s'] * synthetic_config['n_epochs']
        config['sampling_rate'] = synthetic_config['sampling_rate_hz']
        config['epoch_length'] = synthetic_config['epoch_length_s']
        config['n_stages'] = synthetic_config['n_stages']
        config['stage_names'] = synthetic_config['stage_names']
        return config
    
    def __str__(self):
        return f"Synthetic(id={self.config['name']})"