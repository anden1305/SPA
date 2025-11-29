import numpy as np
from src.config import config
from src.preprocessing.helpers.base_transform import BaseTransform
from src.config.config import TransformsConfig

class Windows(BaseTransform):
    """
    Reshapes the data into windows using the window_size and stride parameters.
    """
    
    def __init__(self, 
                 window_size: int, 
                 stride: int, 
                 sampling_rate: int = None):
        self.window_size = window_size
        self.stride = stride
        self.sampling_rate = sampling_rate
        super().__init__(config, stage="preprocessing")
        
    def validate_config(self, _: TransformsConfig):
        assert isinstance(self.window_size, int) and self.window_size > 0, "window_size must be a positive integer."
        assert isinstance(self.stride, int) and 0 < self.stride <= self.window_size, "stride must be a positive integer less than or equal to window_size."
        assert isinstance(self.sampling_rate, (int, float)) and self.sampling_rate > 0, "sampling_rate must be a positive number."
    
    def __reshape_input(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Reshape input assuming x has shape (T, C) and y has length T.

        Produces (possibly overlapping) windows along the time axis of length
        `window_size` stepped by `stride` and returns an array shaped
        (n, window_size, C), where n = floor((T - window_size) / stride) + 1.
        """
        T, C = x.shape
        if self.window_size <= 0:
            raise ValueError("window_size must be a positive integer.")
        if self.stride <= 0:
            raise ValueError("stride must be a positive integer.")
        if self.window_size > T:
            return (
                np.empty((0, self.window_size, C), dtype=x.dtype),
                np.empty((0, self.window_size), dtype=y.dtype),
            )

        starts = np.arange(0, T - self.window_size + 1, self.stride, dtype=int)
        if starts.size == 0:
            return (
                np.empty((0, self.window_size, C), dtype=x.dtype),
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

        return x_windows, y_windows

    def __downsample_by_majority_voting(self, y_windows: np.ndarray) -> np.ndarray:
        if y_windows.size == 0:
            # preserve original dtype when returning an empty array
            return np.array([], dtype=y_windows.dtype)

        def vote(row):
            vals, counts = np.unique(row, return_counts=True)
            return vals[counts.argmax()]

        return np.apply_along_axis(vote, 1, y_windows)

    def __call__(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Processes the data.

        Input
        -----
        x : (T, C) array
            Time x Channels
        y : (T,) array

        Output
        ------
        X : ndarray, shape (B, window_size, C)
            B = number of windows.
        Y : ndarray, shape (B,)
            Labels downsampled by majority voting per window.
        """
        x_windows, y_windows = self.__reshape_input(x, y)
        Y = self.__downsample_by_majority_voting(y_windows=y_windows)
        return x_windows, Y
    
    def get_short_name(self):
        return "Windows"
    
    def __str__(self) -> str:
        return f"Windows(window_size={self.window_size}, stride={self.stride})"
