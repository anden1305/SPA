
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
    
    def __perform_fft(self, x: np.ndarray) -> np.ndarray:
        return np.fft.rfft(x, axis=-1)
    
    def __reshape_input(self, samples: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        x, y = samples  # x: (C, T)
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
    
    def __call__(self, samples: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Documentation
        Processes the data. Input is of shape (C, T) and output should be of shape (T, C, F), 
        where F is features per channel, T is timestamps and C is channels.
        """
        x, y = samples
        x, y = self.__reshape_input(samples=samples)
        X = self.__perform_fft(x=x)
        Y = self.__downsample_by_majority_voting(y=y)
        return X, Y