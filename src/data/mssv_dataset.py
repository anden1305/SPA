
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
        labels = labels - min(labels)
        return labels

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
        signals = []
        if metadata['EEG1']:
            signals.append('EEG1')
        if metadata['EEG2']:
            signals.append('EEG2')
        if metadata['EEG3']:
            signals.append('EEG3')
        if metadata['EEG4']:
            signals.append('EEG4')
        if metadata['EMG']:
            signals.append('EMG')
        config['n_channels'] = len(signals)
        config['signals'] = signals
        config['run'] = self.run
        config['channels'] = [signal[:3] for signal in signals]
        return config
        
    def __str__(self):
        return f"MSSV(id={self.config['name']}, run={self.run})"