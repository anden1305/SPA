import numpy as np
from scipy.signal import butter, filtfilt

from src.config.config import TransformsConfig
from src.data.base_dataset import BaseDataset
from src.preprocessing.base_transform import BaseTransform


class HighPassFilter(BaseTransform):

    def __init__(self, config: TransformsConfig):
        super().__init__(config)

    def validate_config(self, _: TransformsConfig):
        nyquist = 0.5 * self.config.params['sampling_rate']
        self.low = self.config.params['low_cutoff'] / nyquist
        self.high = self.config.params['high_cutoff'] / nyquist
        
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
        return f"PercentileClipping(window_size={self.window_size})"