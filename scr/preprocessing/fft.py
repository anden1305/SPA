
import numpy as np

from scr.data.base_dataset import BaseDataset
from scr.preprocessing.base_transform import BaseTransform


class FFT(BaseTransform):
    """Documentation

    Abstract class for all preprocessing methods that is needed for this codebase.
    """
    
    def __init__(self, params: dict):
        self.params = params
        assert 'window_size' in self.params, "Window size must be specified in params."
        self.window_size: int = self.params['window_size']
        # How to convert complex FFT to real features: 'magnitude' | 'power' | 'log_power'
        self.feature: str = self.params.get('feature', 'log_power')
        # Numerical stability for log power
        self.eps: float = float(self.params.get('eps', 1e-12))
    
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
        raise ValueError(f"Unknown feature '{self.feature}'. Use 'magnitude', 'power', or 'log_power'.")
    
    def __reshape_input(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        C, T = x.shape
        n = T // self.window_size
        T_trunc = n * self.window_size
        x = x[:, :T_trunc].reshape(C, n, self.window_size).transpose(1, 0, 2)  # (T, C, window_size)
        y = y[:T_trunc]
        return x, y
    
    def __downsample_by_majority_voting(self, y: np.ndarray) -> np.ndarray:
        n = len(y) // self.window_size
        if n == 0:
            return np.array([], dtype=object)
        Y = y[: n*self.window_size].reshape(n, self.window_size)

        def vote(row):
            vals, counts = np.unique(row, return_counts=True)
            return vals[counts.argmax()]

        return np.apply_along_axis(vote, 1, Y)

    def __call__(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Documentation
        Processes the data. Input is of shape (C, T) and output is real-valued of shape (T, C, F),
        where F is features per channel, T is timestamps and C is channels.
        """
        x, y = self.__reshape_input(x,y)
        Xc = self.__perform_fft(x=x)
        X = self.__complex_to_real_features(Xc)
        Y = self.__downsample_by_majority_voting(y=y)
        return X, Y