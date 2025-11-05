import numpy as np

from src.config.config import TransformsConfig
from src.preprocessing.helpers.base_transform import BaseTransform
from src.preprocessing.helpers.windows_batcher import WindowBatcher
from src.preprocessing.band_pass_filter import BandPassFilter
from src.preprocessing.percentile_clipping import PercentileClipping

class BurstRate(BaseTransform):
    """
    Burst rate via Teager–Kaiser Energy Operator (TKEO).

    Pipeline:
      1) Band-pass with existing HighPassFilter (used as band-pass) to 10–200 Hz.
      2) Window to (N, C, W).
      3) TKEO: psi[n] = x[n]^2 - x[n-1]*x[n+1].
      4) Robust z-score of psi with median/MAD per window & channel.
      5) Threshold z > thr_z and merge consecutive samples; keep bursts >= min_dur_ms.
      6) Feature = bursts_per_second per window & channel.
    
    Output: (N, 1, C) float32. Higher => more twitches (Awake > REM > NREM typically).
    """

    # Fixed design choices (no external params required)
    bp_low_hz: float = 10.0
    bp_high_hz: float = 200.0
    thr_z: float = 3.0          # robust z threshold on TKEO
    min_dur_ms: float = 5.0    # minimum burst duration to count
    eps: float = 1e-12

    def __init__(self, config: TransformsConfig, window_size: int, stride: int, sampling_rate: int):
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.sampling_rate = float(sampling_rate)

        super().__init__(config, stage="postprocessing")

        # Reuse your high_pass_filter transform as a band-pass by providing both cutoffs.
        # We construct a tiny config-like object for it, reusing the same BaseTransform style.
        class _HPFConfig:
            def __init__(self, low, high):
                self.params = {"low_cutoff": low, "high_cutoff": high}

        self._bp = BandPassFilter(
            config=_HPFConfig(self.bp_low_hz, min(self.bp_high_hz, self.sampling_rate / 2 - 1)),
            sampling_rate=self.sampling_rate,
        )
        
        self.pc = PercentileClipping(
            config=TransformsConfig(
                type="PercentileClipping",
                channel="None",
                params={"percentile": 5.0},
            )
        )

        self.window_batcher = WindowBatcher(window_size=self.window_size, stride=self.stride)

    def validate_config(self, _: TransformsConfig):
        assert isinstance(self.window_size, int) and self.window_size > 0, "window_size must be a positive integer."
        assert isinstance(self.stride, int) and 0 < self.stride <= self.window_size, "stride must be a positive integer ≤ window_size."
        assert self.sampling_rate > 0, "sampling_rate must be positive."

    @staticmethod
    def _mad(x: np.ndarray) -> float:
        med = np.median(x)
        return np.median(np.abs(x - med)) + 1e-12

    def _tkeo_burst_rate(self, x_win: np.ndarray, fs: float) -> float:
        """
        x_win: (W,) band-passed EMG for one channel & window
        returns bursts/sec (float)
        """
        # Guard: need at least 3 samples for TKEO
        if x_win.shape[0] < 3:
            return 0.0

        # Teager–Kaiser Energy Operator (length W-2)
        psi = x_win[1:-1] ** 2 - x_win[:-2] * x_win[2:]
        
        # Robust z-normalization per window
        med = np.median(psi)
        mad = self._mad(psi)
        z = (psi - med) / (1.4826 * mad)

        # Threshold and form bursts
        above = z > self.thr_z
        if not np.any(above):
            return 0.0

        # Find contiguous segments robustly with explicit boundary handling
        # Convert to int to avoid dtype surprises and compute differences
        a = above.astype(np.int8)
        # Differences with implicit 0 at both ends to capture edges
        da = np.diff(a, prepend=0, append=0)
        # starts: where it rises 0->1, ends: where it falls 1->0
        starts = np.flatnonzero(da == 1)
        ends = np.flatnonzero(da == -1)

        # Safety: ensure pairing lengths match (they should by construction)
        if ends.size != starts.size:
            m = min(ends.size, starts.size)
            starts = starts[:m]
            ends = ends[:m]

        # Keep bursts with minimum duration (in samples of psi domain)
        min_len = max(1, int(round(self.min_dur_ms * 1e-3 * fs)))
        valid = (ends - starts) >= min_len

        n_bursts = int(np.sum(valid))
        dur_sec = x_win.shape[0] / fs
        return float(n_bursts / max(dur_sec, 1e-9))

    def __call__(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        x: (T, C) EMG (raw or lightly pre-cleaned)
        y: (T,)
        returns:
            features: (N, 1, C) float32  -- bursts/sec
            y_windows: (N,)
        """
        # 1) Band-pass using the shared HighPassFilter (configured as band-pass)
        x_bp, y_bp = self._bp(x, y)  # still (T, C)
        # x_bp, y_bp = self.pc(x_bp, y_bp)

        # 2) Window to (N, C, W)
        xw, yw = self.window_batcher(x_bp, y_bp)
        N, C, W = xw.shape

        out = np.empty((N, 1, C), dtype=np.float32)
        fs = self.sampling_rate

        # 3) Compute burst rate per window & channel
        for i in range(N):
            for c in range(C):
                xi = xw[i, c, :].astype(np.float64, copy=False)
                # Demean to reduce DC bias into TKEO
                xi = xi - xi.mean()
                out[i, 0, c] = np.float32(self._tkeo_burst_rate(xi, fs))

        return out, yw

    def get_short_name(self):
        return "BurstRate"

    def __str__(self) -> str:
        return (
            "BurstRate(window_size="
            f"{self.window_size}, stride={self.stride}, sampling_rate={self.sampling_rate}, "
            f"band=[{self.bp_low_hz}, {self.bp_high_hz}]Hz, thr_z={self.thr_z}, "
            f"min_dur_ms={self.min_dur_ms})"
        )
