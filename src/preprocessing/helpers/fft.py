import numpy as np

class FFT:
    """Documentation

    Abstract class for all preprocessing methods that is needed for this codebase.
    """
    
    def __init__(self, 
                 window_size: int, 
                 stride: int, 
                 sampling_rate: int,
                 feature: str,
                 bands: list[tuple[float, float]] = None):
        self.window_size: int = window_size
        self.stride: int = stride
        self.sampling_rate = sampling_rate   
        self.feature = feature.lower()
        self.bands = bands  
    
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
        if self.feature == 'magnitude':
            return mag.astype(np.float32)
        if self.feature == 'power':
            return (mag ** 2).astype(np.float32)
        if self.feature == 'log_power':
            power = mag ** 2
            return np.log(power + 1e-12).astype(np.float32)
        if self.feature == 'band_power':
            if self.sampling_rate is None:
                raise ValueError("sampling_rate must be set in transform params for band_power feature.")
            power = mag ** 2
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
        if self.feature == 'log_band_power':
            if self.sampling_rate is None:
                raise ValueError("sampling_rate must be set in transform params for log_band_power feature.")
            power = mag ** 2
            freqs = np.fft.rfftfreq(self.window_size, d=1.0 / self.sampling_rate)
            band_logs = []
            eps = 1e-12
            for i, (low, high) in enumerate(self.bands):
                # right-open for all but the last band
                if i == len(self.bands) - 1:
                    mask = (freqs >= low) & (freqs <= high)
                else:
                    mask = (freqs >= low) & (freqs < high)
                if not np.any(mask):
                    band_logs.append(np.full(power.shape[:-1], np.log(eps), dtype=np.float32))
                else:
                    bp = power[..., mask].sum(axis=-1)               # sum over bins in the band
                    band_logs.append(np.log(bp + eps).astype(np.float32))
            return np.stack(band_logs, axis=-1)

        raise ValueError(f"Unknown feature '{self.feature}'. Use 'magnitude', 'power', 'log_power', or 'band_power'.")
    
    def __reshape_input(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Reshape input assuming x has shape (T, C) and y has length T.

        Produces (possibly overlapping) windows along the time axis of length
        `window_size` stepped by `stride` and returns an array shaped
        (n, C, window_size), where n = floor((T - window_size) / stride) + 1.
        """
        T, C = x.shape
        if self.window_size <= 0:
            raise ValueError("window_size must be a positive integer.")
        if self.stride <= 0:
            raise ValueError("stride must be a positive integer.")
        if self.window_size > T:
            return (
                np.empty((0, C, self.window_size), dtype=x.dtype),
                np.empty((0, self.window_size), dtype=y.dtype),
            )

        starts = np.arange(0, T - self.window_size + 1, self.stride, dtype=int)
        if starts.size == 0:
            return (
                np.empty((0, C, self.window_size), dtype=x.dtype),
                np.empty((0, self.window_size), dtype=y.dtype),
            )

        # shape (n, window_size, C)
        x_windows = np.stack(
            [x[start : start + self.window_size, :] for start in starts], axis=0
        )
        # shape (n, window_size)
        y_windows = np.stack(
            [y[start : start + self.window_size] for start in starts], axis=0
        )

        # transpose to (n, C, window_size) for downstream processing
        x_windows = x_windows.transpose(0, 2, 1)
        return x_windows, y_windows

    def __collapse_dimensions(self, x: np.ndarray):
        # Collapse channel and frequency dims to one feature dimension and
        # add a singleton time axis so output shape is (B, 1, F_total)
        return x.reshape(x.shape[0], 1, -1)

    def __downsample_by_majority_voting(self, y_windows: np.ndarray) -> np.ndarray:
        if y_windows.size == 0:
            # preserve original dtype when returning an empty array
            return np.array([], dtype=y_windows.dtype)

        def vote(row):
            vals, counts = np.unique(row, return_counts=True)
            return vals[counts.argmax()]

        return np.apply_along_axis(vote, 1, y_windows)

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
            B = number of windows (floor((T - window_size) / stride) + 1).
            1 is a singleton time axis per window.
            F_total = channels * features_per_channel (e.g., rFFT bins)
        Y : ndarray, shape (B,)
            Labels downsampled by majority voting per window.
        """
        x, y = self.__reshape_input(x, y)
        Xc = self.__perform_fft(x=x)
        X = self.__complex_to_real_features(Xc)
        X = self.__collapse_dimensions(X)
        Y = self.__downsample_by_majority_voting(y_windows=y)
        return X, Y

    def __str__(self) -> str:
        return f"FFT(window_size={self.window_size}, feature={self.feature})"