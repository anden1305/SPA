import numpy as np

from src.config.config import TransformsConfig
from src.data.base_dataset import BaseDataset
from src.preprocessing.base_transform import BaseTransform


class Reshape(BaseTransform):

    def __init__(self, config: TransformsConfig):
        super().__init__(config, stage="preprocessing")

    def validate_config(self, _: TransformsConfig):
        pass
    
    def __reshape(self, x: np.ndarray, y: np.ndarray):
        return x.transpose(1, 0), y
    
    def __call__(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Documentation
        Processes the data. Input is of shape (C, T) and output should be of shape (T, C), 
        where C is channels and T is time.
        """
        return self.__reshape(x, y)
    
    def __str__(self) -> str:
        return f"Reshape()"