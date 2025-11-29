import numpy as np
from scipy.signal import iirnotch, filtfilt

from src.config.config import TransformsConfig
from src.preprocessing.helpers.base_transform import BaseTransform


class NotchFilter(BaseTransform):
    
    def __init__(self, config: TransformsConfig, sampling_rate: int):
        self.sampling_rate = sampling_rate
        super().__init__(config, stage="preprocessing")
        start_freq = self.config.params['start_freq']
        end_freq = self.config.params['end_freq']
        self.notch_freq = (start_freq + end_freq) / 2
        bandwidth = end_freq - start_freq
        self.Q = self.notch_freq / bandwidth
    
    def validate_config(self, _: TransformsConfig):
        assert type(self.sampling_rate) in [int, float] and self.sampling_rate > 0, "sampling_rate must be a positive number."
        assert 'start_freq' in self.config.params, "NotchFilter transform requires 'start_freq' parameter."
        assert 'end_freq' in self.config.params, "NotchFilter transform requires 'end_freq' parameter."
        assert isinstance(self.config.params['start_freq'], (int, float)) and 0 < self.config.params['start_freq'] < self.sampling_rate / 2, "Start frequency must be a positive number less than Nyquist frequency."
        assert isinstance(self.config.params['end_freq'], (int, float)) and 0 < self.config.params['end_freq'] < self.sampling_rate / 2, "End frequency must be a positive number less than Nyquist frequency."
        assert self.config.params['start_freq'] < self.config.params['end_freq'], "Start frequency must be less than end frequency."
    
    def __filter(self, x: np.ndarray, y: np.ndarray):
        b, a = iirnotch(self.notch_freq, self.Q, fs=self.sampling_rate)
        x = filtfilt(b, a, x, axis=0)
        return x, y
    
    def __call__(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Documentation
        input of shape (T, C) and output should be of shape (T, C).
        """
        return self.__filter(x, y)
    
    def get_short_name(self):
        return "NotchFilter"
    
    def __str__(self) -> str:
        return f"NotchFilter(start={self.config.params['start_freq']}, end={self.config.params['end_freq']})"