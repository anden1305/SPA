

import numpy as np

from src.config.config import TransformsConfig
from src.preprocessing.helpers.base_transform import BaseTransform
from src.preprocessing.helpers.fft import FFT


class FFTRelativeBandPower(BaseTransform):
    """
    Relative band power per window & channel:
        rel_k = P_k / (P_broadband + eps)
    where P_broadband is power in 0.5–45 Hz for the SAME channel and window.

    Output shape: (N_windows, C * num_bands) to match FFT helper conventions.
    """

    def __init__(self, config: TransformsConfig, window_size: int, stride: int, sampling_rate: int):
        self.window_size: int = window_size
        self.stride: int = stride
        self.sampling_rate = sampling_rate
        self.eps: float = float(config.params.get("eps", 1e-12))

        super().__init__(config, stage="postprocessing")
        self.validate_config(config)

        # Broadband (denominator)
        self.fft_full = FFT(
            window_size=window_size,
            stride=stride,
            sampling_rate=sampling_rate,
            feature="log_band_power",
            bands=[[0.5, 30.0]],
        )
        # Target bands (numerator)
        self.fft = FFT(
            window_size=window_size,
            stride=stride,
            sampling_rate=sampling_rate,
            feature="log_band_power",
            bands=self.bands,
        )
    
    def validate_config(self, config: TransformsConfig):
        assert isinstance(self.window_size, int) and self.window_size > 0, "window_size must be a positive integer."
        assert isinstance(self.stride, int) and 0 < self.stride <= self.window_size, "stride must be a positive integer ≤ window_size."
        assert isinstance(self.sampling_rate, (int, float)) and self.sampling_rate > 0, "sampling_rate must be a positive number."
        assert "bands" in config.params, "Frequency bands must be specified in params."
        bands = config.params["bands"]
        assert isinstance(bands, list) and all(isinstance(b, list) and len(b) == 2 for b in bands), "Bands must be a list of [low, high] pairs."
        assert len(bands) > 0, "At least one frequency band must be specified."
        self.bands = bands

    def __call__(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        x: (T, C), y: (T,)
        returns:
            rel_flat: (N_windows, C * num_bands) float32
            labels:   (N_windows,)
        """
        num_channels = x.shape[1]
        n_bands = len(self.bands)

        # Absolute powers
        total_power_flat, labels = self.fft_full(x, y)           # shape: (N, C*1) == (N, C)
        band_power_flat, _ = self.fft(x, y)                      # shape: (N, C*n_bands)

        # Reshape correctly: (N, C, 1) and (N, C, n_bands)
        total_power = total_power_flat.reshape(total_power_flat.shape[0], num_channels, 1)  # per-channel broadband
        band_power = band_power_flat.reshape(band_power_flat.shape[0], num_channels, n_bands)
        
        # Relative power per channel & band
        rel = band_power - total_power  # band_power / (total_power + self.eps)              # (N, C, n_bands)
        
        # Flatten back to (N, C*n_bands)
        rel_flat = rel.reshape(rel.shape[0], n_bands, num_channels).astype(np.float32, copy=False)
        return rel_flat, labels


    def get_short_name(self):
        return "RelativeBandPower"

    def __str__(self) -> str:
        return f"FFTRelativeBandPower(window_size={self.window_size}, stride={self.stride}, sampling_rate={self.sampling_rate}, bands={self.bands})"
