import numpy as np

from src.config.config import TransformsConfig
from src.data.base_dataset import BaseDataset
from src.preprocessing.base_transform import BaseTransform


class BatchRaw(BaseTransform):

    def __init__(self, config: TransformsConfig):
        super().__init__(config)
        self.window_size: int = self.config.params['window_size']

    def validate_config(self, config: TransformsConfig):
        assert 'window_size' in config.params, "BatchRaw transform requires 'window_size' parameter."
        assert isinstance(config.params['window_size'], int) and config.params['window_size'] > 0, "Window size must be a positive integer."

    def __batch(self, x: np.ndarray, y: np.ndarray):
        T, C = x.shape
        W = self.window_size
        B = T // W
        x_batched = x[:B * W].reshape(B, W, C)
        return x_batched, y

    def __call__(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Documentation
        x : (T, C) array
            Time x Channels
        y : (T,) array
        """
        return self.__batch(x, y)

    def __str__(self) -> str:
        return f"BatchRaw(window_size={self.window_size})"