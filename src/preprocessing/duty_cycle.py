
import numpy as np
from src.config.config import TransformsConfig
from src.preprocessing.helpers.base_transform import BaseTransform
from src.preprocessing.helpers.windows_batcher import WindowBatcher
from src.preprocessing.percentile_clipping import PercentileClipping


class DutyCycle(BaseTransform):
    
    threshold_percentile: float = 90.0
    threshold_scale: float = 0.25
    eps: float = 1e-12
    
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
    
    @staticmethod
    def _moving_average(x: np.ndarray, k: int) -> np.ndarray:
        """Channel-wise centered moving average with reflection padding."""
        if k <= 1:
            return x
        pad = k // 2
        # Reflect-pad along time axis (axis=0), keep channels as-is
        x_pad = np.pad(x, ((pad, pad), (0, 0)), mode="reflect")
        kernel = np.ones(k, dtype=x.dtype) / k
        # Apply per channel with 1D conv along time
        out = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="valid"), axis=0, arr=x_pad)
        return out

    def __call__(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # x, y = self.pc(x, y)
        env = np.abs(x)
        thetas = self.threshold_scale * (np.percentile(env, self.threshold_percentile, axis=0) + self.eps)
        env_windows, y_windows = self.window_batcher(env, y)
        below = env_windows < (thetas[np.newaxis, :, np.newaxis])
        duty = below.mean(axis=2, keepdims=True)
        duty = np.transpose(duty, (0, 2, 1))
        return duty.astype(np.float32, copy=False), y_windows

    def get_short_name(self):
        return "DutyCycle"

    def __str__(self) -> str:
        return f"DutyCycle()"