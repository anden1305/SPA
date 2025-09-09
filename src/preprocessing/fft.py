
import numpy as np

from src.config.config import TransformsConfig
from src.data.base_dataset import BaseDataset
from src.preprocessing.base_transform import BaseTransform


class FFT(BaseTransform):
    """Documentation

    Abstract class for all preprocessing methods that is needed for this codebase.
    """

    def __init__(self, config: TransformsConfig):
        super().__init__(config)
        self.window_size: int = self.config.params['window_size']
        self.feature: str = self.config.params['feature'].lower()
        self.eps: float = float(self.config.params.get('eps', 1e-12))
        self.sampling_rate = self.config.params.get('sampling_rate', 128)
        self.bands = self.config.params.get('bands', None)
    
    def validate_config(self, config: TransformsConfig):
        assert 'window_size' in config.params, "Window size must be specified in params."
        assert isinstance(config.params['window_size'], int) and config.params['window_size'] > 0, "Window size must be a positive integer."
        assert 'feature' in config.params, "Feature type must be specified in params."
        assert config.params['feature'].lower() in ['magnitude', 'power', 'log_power', 'band_power'], "Feature must be one of 'magnitude', 'power', 'log_power', or 'band_power'."
        if config.params['feature'].lower() == 'band_power':
            assert 'sampling_rate' in config.params, "Sampling rate must be specified in params for band_power feature."
            assert isinstance(config.params['sampling_rate'], (int, float))
            assert 'brands' in config.params, "Frequency bands must be specified in params for band_power feature."
            assert isinstance(config.params['bands'], list) and all(isinstance(band, list) and len(band) == 2 for band in config.params['bands']), "Bands must be a list of [low, high] pairs."
        

    def __perform_fft(self, x: np.ndarray) -> np.ndarray:
        return np.fft.rfft(x, axis=-1)

    def __complex_to_real_features(self, Xc: np.ndarray) -> np.ndarray:
        """Convert complex spectrum to a real-valued representation.

        Parameters
        ----------
        Xc : np.ndarray
            Complex-valued rFFT output with shape (..., F).

        Returns
        -------
        np.ndarray
            Real-valued features with the same shape as Xc, dtype float32.
        """
        feat = self.feature.lower()
        mag = np.abs(Xc)
        if feat == 'magnitude':
            return mag.astype(np.float32)
        if feat == 'power':
            return (mag ** 2).astype(np.float32)
        if feat == 'log_power':
            power = mag ** 2
            return np.log(power + self.eps).astype(np.float32)
        if feat == 'band_power':
            # compute band power for standard EEG bands per channel
            if self.sampling_rate is None:
                raise ValueError("sampling_rate must be set in transform params for band_power feature.")
            power = mag ** 2  # shape (..., F)
            freqs = np.fft.rfftfreq(self.window_size, d=1.0 / self.sampling_rate)
            band_powers = []
            for i, (low, high) in enumerate(self.bands):
                if i == len(self.bands) - 1:
                    mask = (freqs >= low) & (freqs <= high)
                else:
                    mask = (freqs >= low) & (freqs < high)
                if not np.any(mask):
                    zeros = np.zeros(power.shape[:-1], dtype=np.float32)
                    band_powers.append(zeros)
                else:
                    bp = power[..., mask].sum(axis=-1)
                    band_powers.append(bp.astype(np.float32))
            return np.stack(band_powers, axis=-1)
        raise ValueError(f"Unknown feature '{self.feature}'. Use 'magnitude', 'power', 'log_power', or 'band_power'.")

    def __reshape_input(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Reshape input assuming x has shape (T, C) and y has length T.

        Produces windows along the time axis of length `window_size` and returns
        an array shaped (n, C, window_size), where n = floor(T / window_size).
        """
        T, C = x.shape
        n = T // self.window_size
        T_trunc = n * self.window_size
        x = x[:T_trunc, :].reshape(n, self.window_size, C).transpose(0, 2, 1)
        y = y[:T_trunc]
        return x, y

    def __collapse_dimensions(self, x: np.ndarray):
        # Collapse channel and frequency dims to one feature dimension and
        # add a singleton time axis so output shape is (B, 1, F_total)
        return x.reshape(x.shape[0], 1, -1)

    def __downsample_by_majority_voting(self, y: np.ndarray) -> np.ndarray:
        n = len(y) // self.window_size
        if n == 0:
            # preserve original dtype when returning an empty array
            return np.array([], dtype=y.dtype)
        Y = y[: n*self.window_size].reshape(n, self.window_size)

        def vote(row):
            vals, counts = np.unique(row, return_counts=True)
            return vals[counts.argmax()]

        return np.apply_along_axis(vote, 1, Y)

    def __call__(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Documentation

        Processes the data.

        Input
        -----
        x : (T, C) array
            Time x Channels
        y : (T,) array

        Output
        ------
        X : ndarray, shape (B, 1, F_total)
            B = number of windows (floor(T / window_size)).
            1 is a singleton time axis per window.
            F_total = channels * features_per_channel (e.g., rFFT bins)
        Y : ndarray, shape (B,)
            Labels downsampled by majority voting per window.
        """
        x, y = self.__reshape_input(x,y)
        Xc = self.__perform_fft(x=x)
        X = self.__complex_to_real_features(Xc)
        X = self.__collapse_dimensions(X)
        Y = self.__downsample_by_majority_voting(y=y)
        return X, Y

    def __str__(self) -> str:
        return f"FFT(window_size={self.window_size}, feature={self.feature})"