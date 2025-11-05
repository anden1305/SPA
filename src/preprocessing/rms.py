
import numpy as np
from src.config.config import TransformsConfig
from src.preprocessing.helpers.base_transform import BaseTransform
from src.preprocessing.helpers.windows_batcher import WindowBatcher
from src.preprocessing.percentile_clipping import PercentileClipping


class RMS(BaseTransform):
    
    def __init__(self, config: TransformsConfig, window_size: int, stride: int):
        self.window_size = window_size
        self.stride = stride
        super().__init__(config, stage="postprocessing")
        self.window_batcher = WindowBatcher(window_size=self.window_size, stride=self.stride)
        self.pc = PercentileClipping(
            config=TransformsConfig(
                type="PercentileClipping",
                channel="None",
                params={"percentile": 2.5},
            )
        )
    
    def validate_config(self, _: TransformsConfig):
        assert isinstance(self.window_size, int) and self.window_size > 0, "window_size must be a positive integer."
        assert isinstance(self.stride, int) and 0 < self.stride <= self.window_size, "stride must be a positive integer less than or equal to window_size."
    
    def __call__(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # x, y = self.pc(x, y)
        x_windows, y_windows = self.window_batcher(x, y)
        # compute RMS over the window (time) axis
        rms = np.sqrt(np.mean(np.square(x_windows), axis=2))
        # x from (T, C) to (T, 1, C)
        rms = rms[:, np.newaxis, :]
        return rms.astype(np.float32, copy=False), y_windows
    
    def get_short_name(self):
        return "RMS"
    
    def __str__(self) -> str:
        return f"RMS()"