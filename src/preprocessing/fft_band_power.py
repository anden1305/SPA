
import numpy as np

from src.config.config import TransformsConfig
from src.data.base_dataset import BaseDataset
from src.preprocessing.base_transform import BaseTransform
from src.preprocessing.fft import FFT


class FFTBandPower(BaseTransform):
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
            feature='band_power',
            bands=self.bands,
        )
    
    def validate_config(self, config: TransformsConfig):
        assert type(self.window_size) == int and self.window_size > 0, "window_size must be a positive integer."
        assert type(self.stride) == int and self.stride > 0 and self.stride <= self.window_size, "stride must be a positive integer less than or equal to window_size."
        assert type(self.sampling_rate) in [int, float] and self.sampling_rate > 0, "sampling_rate must be a positive number."
        assert 'bands' in config.params, "Frequency bands must be specified in params for band_power feature."
        assert isinstance(config.params['bands'], list) and all(isinstance(band, list) and len(band) == 2 for band in config.params['bands']), "Bands must be a list of [low, high] pairs."
        self.bands = config.params['bands']
    
    def __call__(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.fft(x, y)

    def __str__(self) -> str:
        return f"FFTBandPower(window_size={self.window_size}, stride={self.stride}, sampling_rate={self.sampling_rate})"