"""Windowed raw EEG/EMG for CNN-front VAE (no FFT)."""

import numpy as np

from src.config.config import GlobalConfig, TransformsConfig
from src.preprocessing.helpers.base_transform import BaseTransform


class RawWindowPreprocessing(BaseTransform):
    """4 s windows in time domain with per-window z-score (replaces FFT path)."""

    def __init__(
        self,
        config: GlobalConfig,
        window_size: int,
        stride: int,
        sequence_length: int,
        normalize: bool,
        sampling_rate: int,
    ):
        self.window_size = window_size
        self.stride = stride
        self.sequence_length = sequence_length
        self.normalize = normalize
        self.sampling_rate = sampling_rate
        super().__init__(config, stage="postprocessing")
        self.global_config = config

    def validate_config(self, _: TransformsConfig):
        assert isinstance(self.window_size, int) and self.window_size > 0
        assert isinstance(self.stride, int) and 0 < self.stride <= self.window_size
        assert isinstance(self.sampling_rate, (int, float)) and self.sampling_rate > 0

    def _reshape_input(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        C, T = x.shape
        if y.shape[0] != T:
            raise ValueError(f"y must have length T={T}, got {y.shape[0]}.")
        starts = np.arange(0, T - self.window_size + 1, self.stride, dtype=int)
        if starts.size == 0:
            return (
                np.empty((0, C, self.window_size), dtype=x.dtype),
                np.empty((0, self.window_size), dtype=y.dtype),
            )
        x_windows = np.stack(
            [x[:, start : start + self.window_size] for start in starts], axis=0
        )
        y_windows = np.stack(
            [y[start : start + self.window_size] for start in starts], axis=0
        )
        return x_windows, y_windows

    def _downsample_by_majority_voting(self, y_windows: np.ndarray) -> np.ndarray:
        if y_windows.size == 0:
            return np.array([], dtype=y_windows.dtype)

        def vote(row):
            vals, counts = np.unique(row, return_counts=True)
            return vals[counts.argmax()]

        return np.apply_along_axis(vote, 1, y_windows)

    def _apply_sequence_length(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n_windows, C, features = x.shape
        n_sequences = n_windows // self.sequence_length
        x = x[: n_sequences * self.sequence_length].reshape(
            n_sequences, self.sequence_length, C, features
        )
        y = y[: n_sequences * self.sequence_length].reshape(n_sequences, self.sequence_length)
        return x, y

    def _normalize_windows(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True) + 1e-8
        return ((x - mean) / std).astype(np.float32)

    def __call__(
        self, x: np.ndarray, y: np.ndarray, for_raw: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        x, y = self._reshape_input(x, y)
        y = self._downsample_by_majority_voting(y)
        if not for_raw:
            x = self._normalize_windows(x)
        else:
            x = x.astype(np.float32)
        x, y = self._apply_sequence_length(x, y)
        return x, y

    def get_short_name(self) -> str:
        return "RawWindowPreprocessing"

    def __str__(self) -> str:
        return (
            f"RawWindowPreprocessing(window_size={self.window_size}, "
            f"stride={self.stride}, sequence_length={self.sequence_length}, "
            f"sampling_rate={self.sampling_rate})"
        )
