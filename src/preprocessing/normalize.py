import numpy as np

from src.config.config import TransformsConfig
from src.data.base_dataset import BaseDataset
from src.preprocessing.base_transform import BaseTransform


class Normalize(BaseTransform):

    def __init__(self, config: TransformsConfig):
        """Normalize transform.

        Supported modes (configured via `config.mode`, defaults to 'dataset'):
        - 'dataset': compute mean/std over (B, T) -> stats shape (1,1,F)
        - 'sample': compute mean/std over T per sample -> stats shape (B,1,F)
        - 'batch': compute mean/std over B per time-step -> stats shape (1,T,F)
        - 'none': do not change input

        `config.eps` may be provided (float) to avoid division by zero. Default 1e-6.
        """
        super().__init__(config)
        self.eps = float(getattr(getattr(config, "params", {}), "eps", 1e-6))

    def validate_config(self, _: TransformsConfig):
        return True
    
    def __normalize(self, x: np.ndarray, y: np.ndarray):
        axes = (0, 1)
        x_mean = x.mean(axis=axes, keepdims=True)
        x_std = x.std(axis=axes, keepdims=True)
        eps = self.eps if self.eps is not None else 1e-6
        x_std = np.maximum(x_std, eps)
        x = (x - x_mean) / x_std
        return x, y

    def __call__(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Documentation
        Normalize the data. Input is of shape (B, T, F) and output should be of shape (B, T, F).
        """
        return self.__normalize(x, y)
    
    def __str__(self) -> str:
        return f"Normalize(dataset, eps={self.eps})"