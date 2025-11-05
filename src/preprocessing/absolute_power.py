
import numpy as np

from src.config.config import TransformsConfig
from src.data.base_dataset import BaseDataset
from src.preprocessing.helpers.base_transform import BaseTransform
from src.preprocessing.helpers.fft import FFT
from numpy.lib.stride_tricks import sliding_window_view

class AbsolutePower(BaseTransform):
    
    def __init__(self, config: TransformsConfig, window_size: int, stride: int, sampling_rate: int):
        self.window_size = window_size
        self.stride = stride
        self.sampling_rate = sampling_rate
        super().__init__(config, stage="postprocessing")
        self.fft = FFT(
            window_size=window_size,
            stride=stride,
            sampling_rate=sampling_rate,
            feature="log_band_power",
            bands=[[0.5, 45.0]],
        )

    def validate_config(self, _: TransformsConfig):
        assert isinstance(self.window_size, int) and self.window_size > 0, "window_size must be a positive integer."
        assert isinstance(self.stride, int) and 0 < self.stride <= self.window_size, "stride must be a positive integer less than or equal to window_size."
        assert isinstance(self.sampling_rate, (int, float)) and self.sampling_rate > 0, "sampling_rate must be a positive number."
    
    def __call__(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        band_features, labels = self.fft(x, y)
        return band_features, labels
    
    def get_short_name(self):
        return "AbsolutePower"
    
    def __str__(self) -> str:
        return f"AbsolutePower()"