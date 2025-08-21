
from scr.config.config import GlobalConfig
from scr.data.base_dataset import BaseDataset
import yaml
import numpy as np
import pandas as pd

class MSSVDataset(BaseDataset):
    """Documentation
    
    Dataset class for the MSSV dataset.
    """
    
    BASE_PATH = 'data/ds006366_processed'

    def __init__(self, 
                 config: GlobalConfig):
        self.run = config.dataset.run
        self.data_path = f'{self.BASE_PATH}/{config.dataset.id}/{config.dataset.run}'
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
        return labels

    def load_config(self):
        config = {}
        metadata_all = pd.read_csv(f'{self.BASE_PATH}/metadata.csv')
        metadata = metadata_all[(metadata_all['participant_id'] == self.id) & (metadata_all['run'] == self.run)].iloc[0]
        config['name'] = metadata['participant_id']
        config['lab'] = metadata['lab']
        config['n_timesteps'] = metadata['samples']
        config['sampling_rate'] = metadata['fs']
        config['epoch_length'] = 4
        config['n_stages'] = 4 if config['lab'] != 'lab_2' else 3
        config['stage_names'] = ['Awake', 'NREM', 'REM', 'Artifact'] if config['lab'] != 'lab_2' else ['Awake', 'NREM', 'REM']
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
        return config
        
    def __str__(self):
        return f"SyntheticDataset(id={self.config['name']})"