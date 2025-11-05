
import numpy as np

from src.config.config import TransformsConfig
from src.preprocessing.helpers.base_transform import BaseTransform
from src.preprocessing.helpers.fft import FFT


class ThetaPeakQuality(BaseTransform):
    """
    Theta Peak Quality (TPQ) per window & channel:
      TPQ = log(prominence) + beta * log(Q)
    where
      prominence = P0 / median(P_baseline)
      Q         = f0 / FWHM  (FWHM via -3 dB crossings within the binned spectrum)

    Output: (N_windows, 1, C)  -- higher TPQ => more REM-like theta.
    """

    def __init__(
        self,
        config: TransformsConfig,
        window_size: int,
        stride: int,
        sampling_rate: int,
        baseline_band: tuple[float, float] = (4.0, 14.0),
        theta_band: tuple[float, float] = (6.0, 9.0),
        bin_width_hz: float = 0.25,
        exclude_halfwidth_hz: float = 0.5,
        beta: float = 0.5,
    ):
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.sampling_rate = sampling_rate
        self.baseline_band = baseline_band
        self.theta_band = theta_band
        self.bin_width_hz = float(bin_width_hz)
        self.exclude_halfwidth_hz = float(exclude_halfwidth_hz)
        self.beta = float(beta)
        self.eps: np.float32 = np.float32(config.params.get("eps", 1e-12))

        # Build non-overlapping bands from baseline low to high (right-open intervals)
        f_lo, f_hi = self.baseline_band
        edges = np.arange(f_lo, f_hi, self.bin_width_hz, dtype=float)
        if edges[-1] < f_hi:  # ensure the last band reaches f_hi
            edges = np.append(edges, f_hi)
        self._band_edges = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
        self._band_centers = np.array([(a + b) / 2.0 for a, b in self._band_edges], dtype=np.float32)

        super().__init__(config, stage="postprocessing")

        # Compute band powers over the fine bins
        self.fft = FFT(
            window_size=self.window_size,
            stride=self.stride,
            sampling_rate=self.sampling_rate,
            feature="band_power",
            bands=self._band_edges,
        )

        # Masks
        centers = self._band_centers
        self._theta_mask = (centers >= self.theta_band[0]) & (centers <= self.theta_band[1])
        self._baseline_mask = (centers >= self.baseline_band[0]) & (centers <= self.baseline_band[1])

    def validate_config(self, _: TransformsConfig):
        assert isinstance(self.window_size, int) and self.window_size > 0
        assert isinstance(self.stride, int) and 0 < self.stride <= self.window_size
        assert isinstance(self.sampling_rate, (int, float)) and self.sampling_rate > 0
        assert self.baseline_band[0] < self.theta_band[0] < self.theta_band[1] < self.baseline_band[1]
        assert 0.0 < self.bin_width_hz <= 1.0
        assert 0.0 < self.exclude_halfwidth_hz <= 2.0

    @staticmethod
    def _interp_crossing(f: np.ndarray, p: np.ndarray, level: float, search_left: bool) -> float | None:
        """Linear interpolation to find f where p crosses 'level'."""
        if search_left:
            # walk from right to left
            for i in range(len(p) - 1, 0, -1):
                p0, p1 = p[i - 1], p[i]
                if (p0 - level) * (p1 - level) <= 0 and p0 != p1:
                    t = (level - p0) / (p1 - p0)
                    return f[i - 1] + t * (f[i] - f[i - 1])
        else:
            # walk left to right
            for i in range(len(p) - 1):
                p0, p1 = p[i], p[i + 1]
                if (p0 - level) * (p1 - level) <= 0 and p0 != p1:
                    t = (level - p0) / (p1 - p0)
                    return f[i] + t * (f[i + 1] - f[i])
        return None

    def __call__(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        x: (T, C) EEG; y: (T,)
        returns: (tpq, y_windows) with tpq shape (N_windows, 1, C)
        """
        num_channels = x.shape[1]

        band_features, labels = self.fft(x, y)  # expected shape: (N_windows, C * n_bins)
        n_bins = len(self._band_edges)

        # Reshape like your original PeakFrequency: (N, C, n_bins)
        band_features = band_features.reshape(band_features.shape[0], num_channels, n_bins)

        centers = self._band_centers
        theta_mask = self._theta_mask
        baseline_mask_all = self._baseline_mask

        N = band_features.shape[0]
        tpq = np.empty((N, 1, num_channels), dtype=np.float32)

        P_half_factor = 1.0 / np.sqrt(2.0)
        min_width = (centers[1] - centers[0]) if n_bins > 1 else 0.1
        fallback_fwhm = 4.0  # Hz if crossings not found

        for w in range(N):
            P = band_features[w]  # (C, n_bins)
            for c in range(num_channels):
                Pc = P[c]

                # Peak in theta band
                if not np.any(theta_mask):
                    tpq[w, 0, c] = 0.0
                    continue

                theta_idxs = np.where(theta_mask)[0]
                i_rel = int(np.argmax(Pc[theta_mask]))
                i0 = theta_idxs[i_rel]
                f0 = float(centers[i0])
                P0 = float(Pc[i0])

                # Robust baseline: median in 4–14 Hz excluding ±exclude_halfwidth around f0
                exclude_mask = (centers >= (f0 - self.exclude_halfwidth_hz)) & (centers <= (f0 + self.exclude_halfwidth_hz))
                baseline_mask = baseline_mask_all & (~exclude_mask)
                baseline_vals = Pc[baseline_mask]
                if baseline_vals.size == 0:
                    baseline = float(np.median(Pc[baseline_mask_all])) + float(self.eps)
                else:
                    baseline = float(np.median(baseline_vals)) + float(self.eps)

                prominence = (P0 + self.eps) / baseline

                # FWHM via -3 dB crossings
                P_half = P0 * P_half_factor
                f_lo = self._interp_crossing(centers[: i0 + 1], Pc[: i0 + 1], P_half, search_left=True)
                f_hi = self._interp_crossing(centers[i0:], Pc[i0:], P_half, search_left=False)
                if (f_lo is None) or (f_hi is None):
                    fwhm = fallback_fwhm
                else:
                    fwhm = max(min_width, float(f_hi - f_lo))

                Q = f0 / (fwhm + self.eps)

                # Theta Peak Quality
                tpq_val = np.log(max(prominence, 1.0) + self.eps) + self.beta * np.log(max(Q, 1.0) + self.eps)
                tpq[w, 0, c] = np.float32(tpq_val)

        return tpq, labels

    def get_short_name(self):
        return "ThetaPeakQuality"

    def __str__(self) -> str:
        return (
            "ThetaPeakQuality(window_size="
            f"{self.window_size}, stride={self.stride}, sampling_rate={self.sampling_rate}, "
            f"baseline_band={self.baseline_band}, theta_band={self.theta_band}, "
            f"bin_width_hz={self.bin_width_hz}, exclude_halfwidth_hz={self.exclude_halfwidth_hz}, "
            f"beta={self.beta})"
        )