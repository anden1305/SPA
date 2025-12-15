
import numpy as np

from src.config.config import TransformsConfig
from src.data.base_dataset import BaseDataset
from src.preprocessing.helpers.base_transform import BaseTransform
from src.preprocessing.helpers.fft import FFT


class VAEPreprocessing(BaseTransform):
    """Documentation

    Abstract class for all preprocessing methods that is needed for this codebase.
    """

    def __init__(self, 
                 config: TransformsConfig, 
                 window_size: int, 
                 stride: int, 
                 sequence_length: int,
                 sampling_rate: int):
        self.window_size: int = window_size
        self.stride: int = stride
        self.sequence_length: int = sequence_length
        self.sampling_rate = sampling_rate
        super().__init__(config, stage="postprocessing")
    
    def validate_config(self, _: TransformsConfig):
        assert type(self.window_size) == int and self.window_size > 0, "window_size must be a positive integer."
        assert type(self.stride) == int and self.stride > 0 and self.stride <= self.window_size, "stride must be a positive integer less than or equal to window_size."
        assert type(self.sampling_rate) in [int, float] and self.sampling_rate > 0, "sampling_rate must be a positive number."
    
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
        mag = np.abs(Xc)
        power = mag ** 2
        return np.log(power + 1e-12).astype(np.float32)
    
    def __reshape_input(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Reshape input assuming x has shape (C, T) and y has length T.

        Produces (possibly overlapping) windows along the time axis of length
        `window_size` stepped by `stride` and returns an array shaped
        (n, C, window_size), where n = floor((T - window_size) / stride) + 1.
        """
        C, T = x.shape

        if self.window_size <= 0:
            raise ValueError("window_size must be a positive integer.")
        if self.stride <= 0:
            raise ValueError("stride must be a positive integer.")
        if y.shape[0] != T:
            raise ValueError(f"y must have length T={T}, got {y.shape[0]}.")
        if self.window_size > T:
            raise ValueError("window_size must be less than or equal to the length of the input sequence.")

        starts = np.arange(0, T - self.window_size + 1, self.stride, dtype=int)
        if starts.size == 0:
            return (
                np.empty((0, C, self.window_size), dtype=x.dtype),
                np.empty((0, self.window_size), dtype=y.dtype),
            )

        # x windows: slice time axis => each is (C, window_size), stack => (n, C, window_size)
        x_windows = np.stack(
            [x[:, start : start + self.window_size] for start in starts], axis=0
        )

        # y windows: each is (window_size,), stack => (n, window_size)
        y_windows = np.stack(
            [y[start : start + self.window_size] for start in starts], axis=0
        )

        return x_windows, y_windows


    def __downsample_by_majority_voting(self, y_windows: np.ndarray) -> np.ndarray:
        if y_windows.size == 0:
            # preserve original dtype when returning an empty array
            return np.array([], dtype=y_windows.dtype)

        def vote(row):
            vals, counts = np.unique(row, return_counts=True)
            return vals[counts.argmax()]

        return np.apply_along_axis(vote, 1, y_windows)
    
    def __apply_seuence_length(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Reshape input into sequences of length `sequence_length`."""        
        n_windows, C, features = x.shape
        n_sequences = n_windows // self.sequence_length
        x = x[:n_sequences * self.sequence_length].reshape(n_sequences, self.sequence_length, C, features)
        y = y[:n_sequences * self.sequence_length].reshape(n_sequences, self.sequence_length)
        return x, y
    
    def __call__(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, y = self.__reshape_input(x, y)
        Xc = self.__perform_fft(x=x)
        X = self.__complex_to_real_features(Xc)
        Y = self.__downsample_by_majority_voting(y_windows=y)
        X, Y = self.__apply_seuence_length(X, Y)
        return X, Y
    
    def get_short_name(self):
        return "VAEPreprocessing"
    
    def __str__(self) -> str:
        return f"VAEPreprocessing(window_size={self.window_size}, stride={self.stride}, sequence_length={self.sequence_length}, sampling_rate={self.sampling_rate})"