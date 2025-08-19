import numpy as np

from scr.data.base_dataset import BaseDataset
from scr.preprocessing.base_transform import BaseTransform


class CollapseDimensions(BaseTransform):

    def __init__(self, params: dict):
        self.params = params
        
    def __collapse_dimensions(self, samples: tuple[np.ndarray, np.ndarray]):
        x, y = samples
        X = x.reshape(x.shape[0], -1)  # Collapse dimensions to (T, F*C)
        return X, y

    def __call__(self, samples: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Documentation
        Processes the data. Input is of shape (T, C, F) and output should be of shape (T, F*C), 
        where F is features per channel, T is timestamps and C is channels.
        """
        return self.__collapse_dimensions(samples=samples)
        