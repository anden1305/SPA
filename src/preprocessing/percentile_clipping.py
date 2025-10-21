import numpy as np

from src.config.config import TransformsConfig
from src.data.base_dataset import BaseDataset
from src.preprocessing.base_transform import BaseTransform


class PercentileClipping(BaseTransform):

    def __init__(self, config: TransformsConfig):
        super().__init__(config, stage="preprocessing")
        self.percentile: int = self.config.params['percentile']

    def validate_config(self, _: TransformsConfig):
        assert 'percentile' in self.config.params, "PercentileClipping transform requires 'percentile' parameter."
        assert isinstance(self.config.params['percentile'], int) and 0 < self.config.params['percentile'] < 50, "Percentile must be an integer between 0 and 50."
    
    def __clip(self, x: np.ndarray, y: np.ndarray):
        lower_bound = np.percentile(x, self.percentile, axis=0)
        upper_bound = np.percentile(x, 100 - self.percentile, axis=0)
        x = np.clip(x, lower_bound, upper_bound)
        return x, y
    
    def __call__(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Documentation
        input of shape (T, C) and output should be of shape (T, C).
        """
        return self.__clip(x, y)

    def __str__(self) -> str:
        return f"PercentileClipping(percentile={self.percentile})"