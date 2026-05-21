
from src.config.config import DatasetConfig, GlobalConfig
from src.data.base_dataset import BaseDataset
import yaml
import numpy as np
import pandas as pd

class MSSVDataset(BaseDataset):
    """Documentation
    
    Dataset class for the MSSV dataset.
    """
    
    BASE_PATH = 'data/ds006366_processed'
    
    def __init__(self, 
                 config: DatasetConfig):
        self.run = config.run if config.run is not None else 1
        self.data_path = f'{self.BASE_PATH}/{config.id}/{self.run}'
        super().__init__(config=config)
        if self.dataset_config.remove_artifact:
            self.data, self.labels = self.remove_artifact(self.data, self.labels)
    
    def load_data(self):
        data = None
        for signal in self.config['signals']:
            signal_data = np.load(f'{self.data_path}/{signal}.npy')
            if data is None:
                data = signal_data[np.newaxis, :]
            else:
                data = np.concatenate((data, signal_data[np.newaxis, :]), axis=0)
        return data

    def load_labels(self):
        labels = np.load(f'{self.data_path}/labels.npy')
        ordered_unique_labels = np.unique(labels) - 1
        self.config['stage_names'] = [self.config['stage_names'][i] for i in ordered_unique_labels]
        self.config['n_stages'] = len(self.config['stage_names'])
        labels = labels - min(labels)
        return labels
    
    def remove_artifact(self, data: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not 'Artifact' in self.config['stage_names']:
            return data, labels
        artifact_index = self.config['stage_names'].index('Artifact')
        mask = labels != artifact_index
        data = data[:, mask]
        labels = labels[mask]
        self.config['stage_names'].remove('Artifact')
        self.config['n_stages'] -= 1
        return data, labels

    def load_config(self):
        config = {}
        metadata_all = pd.read_csv(f'{self.BASE_PATH}/metadata.csv')
        assert not metadata_all[(metadata_all['participant_id'] == self.id) & (metadata_all['run'] == self.run)].empty, f"Metadata for participant {self.id} and run {self.run} not found"
        metadata = metadata_all[(metadata_all['participant_id'] == self.id) & (metadata_all['run'] == self.run)].iloc[0]
        config['name'] = metadata['participant_id']
        config['lab'] = metadata['lab']
        config['n_timesteps'] = metadata['samples']
        config['sampling_rate'] = metadata['fs']
        config['epoch_length'] = 4
        config['n_stages'] = 4 if not config['lab'] in ('lab_2', 'lab_4') else 3
        config['stage_names'] = ['Awake', 'NREM', 'REM', 'Artifact'] if not config['lab'] in ('lab_2', 'lab_4') else ['Awake', 'NREM', 'REM']
        available = []
        if metadata['EEG1']:
            available.append('EEG1')
        if metadata['EEG2']:
            available.append('EEG2')
        if metadata['EEG3']:
            available.append('EEG3')
        if metadata['EEG4']:
            available.append('EEG4')
        if metadata['EMG']:
            available.append('EMG')
        if self.dataset_config.signals is not None:
            signals = [s for s in self.dataset_config.signals if s in available]
            if not signals:
                raise ValueError(
                    f"No requested signals {self.dataset_config.signals} available for "
                    f"{self.id} run {self.run} (available: {available})"
                )
        else:
            signals = available
        config['n_channels'] = len(signals)
        config['signals'] = signals
        config['run'] = self.run
        config['channels'] = [signal[:3] for signal in signals]
        return config
        
    def __str__(self):
        return f"MSSV(id={self.config['name']}, run={self.run})"