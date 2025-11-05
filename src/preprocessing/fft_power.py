
import numpy as np

from src.config.config import TransformsConfig
from src.data.base_dataset import BaseDataset
from src.preprocessing.helpers.base_transform import BaseTransform
from src.preprocessing.helpers.fft import FFT


class FFTPower(BaseTransform):
    """Documentation

    Abstract class for all preprocessing methods that is needed for this codebase.
    """

    def __init__(self, config: TransformsConfig, window_size: int, stride: int, sampling_rate: int):
        self.window_size: int = window_size
        self.stride: int = stride
        self.sampling_rate = sampling_rate
        super().__init__(config, stage="postprocessing")
        self.fft = FFT(
            window_size=window_size,
            stride=stride,
            sampling_rate=sampling_rate,
            feature='power',
        )
    
    def validate_config(self, _: TransformsConfig):
        assert type(self.window_size) == int and self.window_size > 0, "window_size must be a positive integer."
        assert type(self.stride) == int and self.stride > 0 and self.stride <= self.window_size, "stride must be a positive integer less than or equal to window_size."
        assert type(self.sampling_rate) in [int, float] and self.sampling_rate > 0, "sampling_rate must be a positive number."

    def __call__(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.fft(x, y)
    
    def get_short_name(self):
        return "Power"

    def __str__(self) -> str:
        return f"FFTPower(window_size={self.window_size}, stride={self.stride}, sampling_rate={self.sampling_rate})"