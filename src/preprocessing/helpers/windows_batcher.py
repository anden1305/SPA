import numpy as np

class WindowBatcher:
    """Documentation

    Abstract class for all preprocessing methods that is needed for this codebase.
    """
    
    def __init__(self, 
                 window_size: int, 
                 stride: int):
        self.window_size: int = window_size
        self.stride: int = stride
    
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

    def __downsample_by_majority_voting(self, y_windows: np.ndarray) -> np.ndarray:
        if y_windows.size == 0:
            # preserve original dtype when returning an empty array
            return np.array([], dtype=y_windows.dtype)

        def vote(row):
            vals, counts = np.unique(row, return_counts=True)
            return vals[counts.argmax()]

        return np.apply_along_axis(vote, 1, y_windows)

    def __call__(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X, y = self.__reshape_input(x, y)
        Y = self.__downsample_by_majority_voting(y_windows=y)
        return X, Y