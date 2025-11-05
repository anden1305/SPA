import numpy as np

from src.config.config import TransformsConfig
from src.preprocessing.helpers.base_transform import BaseTransform
from src.preprocessing.helpers.windows_batcher import WindowBatcher
from src.preprocessing.percentile_clipping import PercentileClipping


class BetaDeltaPAC(BaseTransform):
    """
    Beta–Delta Phase–Amplitude Coupling (Tort Modulation Index) per window & channel.

    Purpose:
      - Explores coupling between beta phase (13–30 Hz) and delta amplitude (0.5–4 Hz).
      - Provided to mirror the structure of ThetaGammaPAC with different bands.

    Input:  x (T, C), y (T,)
    Batcher: (N, C, W)
    Output: (N, 1, C) with MI in [0, 1], higher => stronger coupling.
    """

    def __init__(self, config: TransformsConfig, window_size: int, stride: int, sampling_rate: int):
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.sampling_rate = float(sampling_rate)

        # Fixed design (no user-tunable mode/params)
        self.beta_lo = 13.0
        self.beta_hi = 30.0
        self.delta_lo = 0.5
        self.delta_hi = 4.0
        self.n_phase_bins = 18  # Tort et al. standard-ish
        self.eps = 1e-12

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
        assert isinstance(self.stride, int) and 0 < self.stride <= self.window_size, "stride must be in (0, window_size]."
        assert self.sampling_rate > 0, "sampling_rate must be positive."

    # ---------- helpers (NumPy-only) ----------

    def _fft_bandpass(self, x: np.ndarray, f_lo: float, f_hi: float, fs: float) -> np.ndarray:
        """Simple FFT mask bandpass (real → real)."""
        n = x.shape[0]
        X = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        mask = (freqs >= f_lo) & (freqs <= f_hi)
        X_f = np.zeros_like(X)
        X_f[mask] = X[mask]
        return np.fft.irfft(X_f, n=n)

    def _hilbert_analytic(self, x: np.ndarray) -> np.ndarray:
        """Analytic signal via FFT-based Hilbert transform (no SciPy)."""
        n = x.shape[0]
        X = np.fft.fft(x)
        H = np.zeros(n, dtype=np.float64)
        if n % 2 == 0:
            H[0] = 1.0
            H[n // 2] = 1.0
            H[1 : n // 2] = 2.0
        else:
            H[0] = 1.0
            H[1 : (n + 1) // 2] = 2.0
        Z = np.fft.ifft(X * H)
        return Z

    # ---------- main ----------

    def __call__(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        returns:
          mi: (N, 1, C) Tort MI in [0, 1]
          y_windows: (N,)
        """
        # x, y = self.pc(x, y)
        fs = self.sampling_rate
        x_windows, y_windows = self.window_batcher(x, y)  # (N, C, W), (N,)
        N, C, W = x_windows.shape

        # Precompute phase bin edges
        edges = np.linspace(-np.pi, np.pi, self.n_phase_bins + 1, endpoint=True)

        out = np.empty((N, 1, C), dtype=np.float32)

        for i in range(N):
            for c in range(C):
                xi = x_windows[i, c, :].astype(np.float64, copy=False)
                xi = xi - xi.mean()

                # 1) Beta phase
                x_beta = self._fft_bandpass(xi, self.beta_lo, self.beta_hi, fs)
                z_beta = self._hilbert_analytic(x_beta)
                phase = np.angle(z_beta)  # [-pi, pi]

                # 2) Delta amplitude envelope
                x_delta = self._fft_bandpass(xi, self.delta_lo, self.delta_hi, fs)
                z_delta = self._hilbert_analytic(x_delta)
                amp = np.abs(z_delta) + self.eps

                # 3) Tort MI: KL divergence between amp-by-phase distribution and uniform
                # Bin phases, collect mean amplitude per bin
                bin_idx = np.digitize(phase, edges) - 1
                bin_idx = np.clip(bin_idx, 0, self.n_phase_bins - 1)

                A_mean = np.zeros(self.n_phase_bins, dtype=np.float64)
                counts = np.zeros(self.n_phase_bins, dtype=np.int64)
                for k in range(self.n_phase_bins):
                    mask = (bin_idx == k)
                    if np.any(mask):
                        A_mean[k] = amp[mask].mean()
                        counts[k] = mask.sum()

                # If some bins are empty (short windows), softly fill with global mean
                if (counts == 0).any():
                    A_global = amp.mean()
                    A_mean[counts == 0] = A_global

                P = A_mean / (A_mean.sum() + self.eps)  # probability over phase bins
                U = 1.0 / self.n_phase_bins

                # KL divergence to uniform, normalized by log(n_bins)
                kl = np.sum(P * (np.log(P + self.eps) - np.log(U)))
                mi = kl / np.log(self.n_phase_bins)

                # Clip to [0,1] for numerical stability
                mi = float(np.clip(mi, 0.0, 1.0))
                out[i, 0, c] = np.float32(mi)

        return out, y_windows

    def get_short_name(self):
        return "BetaDeltaPAC"

    def __str__(self) -> str:
        return (
            "BetaDeltaPAC(window_size="
            f"{self.window_size}, stride={self.stride}, sampling_rate={self.sampling_rate})"
        )
