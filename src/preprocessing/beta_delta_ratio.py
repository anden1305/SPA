
import numpy as np

from src.config.config import TransformsConfig
from src.preprocessing.helpers.base_transform import BaseTransform
from src.preprocessing.helpers.fft import FFT


class BetaDeltaRatio(BaseTransform):
    """Compute beta/delta power ratio per channel window.

    Input is (T, C) data and (T,) labels. Output contains a single feature per
    channel and window: beta_power / delta_power.
    """
    
    _DELTA_TARGET = (0.5, 4.0)
    _DELTA_INDEX = 0
    _BETA_TARGET = (15.0, 40.0)
    _BETA_INDEX = 1

    def __init__(self, config: TransformsConfig, window_size: int, stride: int, sampling_rate: int):
        self.window_size: int = window_size
        self.stride: int = stride
        self.sampling_rate = sampling_rate
        self.eps: np.float32 = np.float32(config.params.get("eps", 1e-12))
        super().__init__(config, stage="postprocessing")
        self.fft = FFT(
            window_size=window_size,
            stride=stride,
            sampling_rate=sampling_rate,
            feature="log_band_power",
            bands=[self._DELTA_TARGET, self._BETA_TARGET],
        )
        

    def validate_config(self, _: TransformsConfig):
        assert isinstance(self.window_size, int) and self.window_size > 0, "window_size must be a positive integer."
        assert isinstance(self.stride, int) and 0 < self.stride <= self.window_size, "stride must be a positive integer less than or equal to window_size."
        assert isinstance(self.sampling_rate, (int, float)) and self.sampling_rate > 0, "sampling_rate must be a positive number."

    def __call__(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        num_channels = x.shape[1]
        band_features, labels = self.fft(x, y)

        # reshape to (B, 1, C, num_bands)
        band_features = band_features.reshape(band_features.shape[0], 1, num_channels, 2)
        beta_power = band_features[..., self._BETA_INDEX]
        delta_power = band_features[..., self._DELTA_INDEX]
        ratio = beta_power - delta_power # beta_power / (delta_power + self.eps)

        # ensure output is (B, 1, C)
        ratio = ratio.reshape(ratio.shape[0], 1, num_channels)
        return ratio.astype(np.float32, copy=False), labels
    
    def get_short_name(self):
        return "BetaDeltaRatio"

    def __str__(self) -> str:
        return (
            "BetaDeltaRatio(window_size="
            f"{self.window_size}, stride={self.stride}, sampling_rate={self.sampling_rate})"
        )