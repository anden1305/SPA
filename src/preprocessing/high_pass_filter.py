import numpy as np
from scipy.signal import butter, filtfilt

from src.config.config import TransformsConfig
from src.data.base_dataset import BaseDataset
from src.preprocessing.base_transform import BaseTransform


class HighPassFilter(BaseTransform):

    def __init__(self, config: TransformsConfig):
        super().__init__(config, stage="preprocessing")
        nyquist = 0.5 * self.config.params['sampling_rate']
        self.low = self.config.params['low_cutoff'] / nyquist
        self.high = self.config.params['high_cutoff'] / nyquist

    def validate_config(self, _: TransformsConfig):
        assert 'sampling_rate' in self.config.params, "HighPassFilter transform requires 'sampling_rate' parameter."
        assert 'low_cutoff' in self.config.params, "HighPassFilter transform requires 'low_cutoff' parameter."
        assert 'high_cutoff' in self.config.params, "HighPassFilter transform requires 'high_cutoff' parameter."
        assert isinstance(self.config.params['sampling_rate'], (int, float)) and self.config.params['sampling_rate'] > 0, "Sampling rate must be a positive number."
        assert isinstance(self.config.params['low_cutoff'], (int, float)) and 0 < self.config.params['low_cutoff'] < self.config.params['sampling_rate'] / 2, "Low cutoff must be a positive number less than Nyquist frequency."
        assert isinstance(self.config.params['high_cutoff'], (int, float)) and 0 < self.config.params['high_cutoff'] < self.config.params['sampling_rate'] / 2, "High cutoff must be a positive number less than Nyquist frequency."
        assert self.config.params['low_cutoff'] < self.config.params['high_cutoff'], "Low cutoff must be less than high cutoff."
    
    def __filter(self, x: np.ndarray, y: np.ndarray):
        b, a = butter(1, [self.low, self.high], btype='band')
        x = filtfilt(b, a, x, axis=0)
        return x, y
    
    def __call__(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Documentation
        input of shape (T, C) and output should be of shape (T, C).
        """
        return self.__filter(x, y)

    def __str__(self) -> str:
        return f"HighPassFilter(low={self.low}, high={self.high})"