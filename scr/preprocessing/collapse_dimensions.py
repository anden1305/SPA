import numpy as np

from scr.config.config import TransformsConfig
from scr.data.base_dataset import BaseDataset
from scr.preprocessing.base_transform import BaseTransform


class CollapseDimensions(BaseTransform):

    def __init__(self, config: TransformsConfig):
        super().__init__(config)

    def validate_config(self, _: TransformsConfig):
        pass

    def __collapse_dimensions(self, x: np.ndarray, y: np.ndarray):
        X = x.reshape(x.shape[0], -1)  # Collapse dimensions to (T, F*C)
        return X, y

    def __call__(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Documentation
        Processes the data. Input is of shape (T, C, F) and output should be of shape (T, F*C), 
        where F is features per channel, T is timestamps and C is channels.
        """
        return self.__collapse_dimensions(x, y)
    
    def __str__(self) -> str:
        return f"CollapseDimensions()"