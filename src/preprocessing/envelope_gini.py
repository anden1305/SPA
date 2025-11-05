import numpy as np

from src.config.config import TransformsConfig
from src.preprocessing.helpers.base_transform import BaseTransform
from src.preprocessing.helpers.windows_batcher import WindowBatcher
from src.preprocessing.band_pass_filter import BandPassFilter

# We rely on scipy.signal via your HighPassFilter file; import here for low-pass.
from scipy.signal import butter, filtfilt


class EnvelopeGini(BaseTransform):
    """
    Gini coefficient of EMG envelope (twitchy sparseness).
      - Band-pass (with your HighPassFilter used as band-pass): 30–300 Hz
      - Full-wave rectify, then low-pass to ~10 Hz to get envelope
      - Per-channel robust scale, then per-window Gini on envelope samples

    Output: (N, 1, C) float32, higher => sparser bursty envelope (REM > Awake ≳ NREM).
    """

    bp_low_hz: float = 0.5
    bp_high_hz: float = 300.0
    env_lowpass_hz: float = 10.0
    lowpass_order: int = 2
    eps: float = 1e-12

    def __init__(self, config: TransformsConfig, window_size: int, stride: int, sampling_rate: int):
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.sampling_rate = float(sampling_rate)

        super().__init__(config, stage="postprocessing")
        self.window_batcher = WindowBatcher(window_size=self.window_size, stride=self.stride)

        # Reuse your HighPassFilter as a band-pass by setting both cutoffs
        class _HPFConfig:
            def __init__(self, low, high):
                self.params = {"low_cutoff": low, "high_cutoff": high}

        self._bp = BandPassFilter(
            config=_HPFConfig(self.bp_low_hz,  min(self.bp_high_hz, self.sampling_rate / 2 - 1)),
            sampling_rate=self.sampling_rate,
        )

    def validate_config(self, _: TransformsConfig):
        assert isinstance(self.window_size, int) and self.window_size > 0, "window_size must be a positive integer."
        assert isinstance(self.stride, int) and 0 < self.stride <= self.window_size, "stride must be a positive integer ≤ window_size."
        assert self.sampling_rate > 0, "sampling_rate must be positive."

    # ---------- helpers ----------

    def _lowpass_envelope(self, x: np.ndarray) -> np.ndarray:
        """
        Envelope via rectification + low-pass Butterworth.
        x: (T, C) band-passed EMG
        returns A: (T, C) envelope
        """
        A = np.abs(x).astype(np.float64, copy=False)
        nyq = 0.5 * self.sampling_rate
        wc = self.env_lowpass_hz / nyq
        b, a = butter(self.lowpass_order, wc, btype="low")
        A_lp = filtfilt(b, a, A, axis=0)
        # Numerical guard: envelope must be non-negative
        return np.clip(A_lp, 0.0, None)

    @staticmethod
    def _gini_1d(x: np.ndarray, eps: float = 1e-12) -> float:
        """
        Gini for nonnegative 1D array.
        Uses the O(n log n) sorted formula. Returns 0 if mean ~ 0.
        """
        n = x.size
        if n == 0:
            return 0.0
        x = np.asarray(x, dtype=np.float64)
        # Ensure nonnegative
        x = np.clip(x, 0.0, None)
        s = x.sum()
        if s <= eps:
            return 0.0
        # Sort ascending
        xs = np.sort(x)
        # G = (1/(n*mean)) * (n+1 - 2 * sum((n+1-i)*x_i)/sum(x))
        # mean = s/n
        idx = np.arange(1, n + 1, dtype=np.float64)
        weighted_sum = np.sum((n + 1 - idx) * xs)
        g = (n + 1 - 2.0 * weighted_sum / s) / n
        return float(np.clip(g, 0.0, 1.0))

    def __call__(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        x: (T, C) EMG (raw or pre-cleaned)
        y: (T,)
        returns:
            features: (N, 1, C) float32  -- Gini of envelope per window & channel
            y_windows: (N,)
        """
        # 1) Band-pass (30–300 Hz) with your HighPassFilter
        x_bp, y_bp = self._bp(x, y)  # (T, C)

        # 2) Envelope (rectify + low-pass)
        A = self._lowpass_envelope(x_bp)  # (T, C)

        # 3) Per-channel robust scale to stabilize across sessions (keep nonnegative)
        #    Divide by median + eps to make the envelope dimensionless.
        med = np.median(A, axis=0)
        A_norm = A / (med + self.eps)

        # 4) Window and compute Gini in each window/channel
        Aw, yw = self.window_batcher(A_norm, y_bp)  # (N, C, W), (N,)
        N, C, W = Aw.shape

        out = np.empty((N, 1, C), dtype=np.float32)
        for i in range(N):
            for c in range(C):
                out[i, 0, c] = np.float32(self._gini_1d(Aw[i, c, :], eps=self.eps))

        return out, yw

    def get_short_name(self):
        return "EnvelopeGini"

    def __str__(self) -> str:
        return (
            "EnvelopeGini(window_size="
            f"{self.window_size}, stride={self.stride}, sampling_rate={self.sampling_rate}, "
            f"band=[{self.bp_low_hz}, {self.bp_high_hz}]Hz, env_lowpass={self.env_lowpass_hz}Hz)"
        )
