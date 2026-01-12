
import numpy as np
from scipy import signal
from src.config.config import TransformsConfig
from src.preprocessing.helpers.base_transform import BaseTransform
from scipy.signal import butter, sosfiltfilt


class VAEPreprocessing(BaseTransform):
    """Documentation

    Abstract class for all preprocessing methods that is needed for this codebase.
    """
    
    # _BANDPASS_HZ = {
    #     0: (0.5, 30.0),
    #     1: (0.5, 30.0),
    #     2: (10.0, 63.0),
    # }
    _BANDPASS_HZ = {
        0: (0.5, 30.0),
        1: (0.5, 30.0),
        2: (None,None),
    }
    _BANDPASS_ORDER = 4
    
    def __init__(self, 
                 config: TransformsConfig, 
                 window_size: int, 
                 stride: int, 
                 sequence_length: int,
                 normalize: bool,
                 sampling_rate: int):
        self.window_size: int = window_size
        self.stride: int = stride
        self.sequence_length: int = sequence_length
        self.normalize: bool = normalize
        self.sampling_rate: int = sampling_rate
        # self._bandpass_sos = self.__build_bandpass_sos()
        super().__init__(config, stage="postprocessing")
    
    def validate_config(self, _: TransformsConfig):
        assert type(self.window_size) == int and self.window_size > 0, "window_size must be a positive integer."
        assert type(self.stride) == int and self.stride > 0 and self.stride <= self.window_size, "stride must be a positive integer less than or equal to window_size."
        assert type(self.sampling_rate) in [int, float] and self.sampling_rate > 0, "sampling_rate must be a positive number."
    
    def __perform_fft(self, x: np.ndarray) -> np.ndarray:
        return np.fft.rfft(x, axis=-1)
    
    def __perform_stft(self, x: np.ndarray) -> np.ndarray:
        """Compute the Short-Time Fourier Transform (STFT) of the input signal."""
        _, _, Zxx = signal.stft(x, fs=self.sampling_rate, window='hann', nperseg=self.window_size, noverlap=self.window_size - self.stride, axis=-1, padded=False, boundary=None)
        print(f"STFT raw shape: {Zxx.shape}")
        return Zxx
    
    def __perform_hanning_window(self, x: np.ndarray) -> np.ndarray:
        w = np.hanning(self.window_size).astype(np.float32)
        return x * w[None, None, :]

    def __complex_to_real_features(self, Xc: np.ndarray) -> np.ndarray:
        """Convert complex spectrum to a real-valued representation.

        Parameters
        ----------
        Xc : np.ndarray
            Complex-valued rFFT output with shape (..., F).

        Returns
        -------
        np.ndarray
            Real-valued features with the same shape as Xc, dtype float32.
        """
        mag = np.abs(Xc)
        power = mag ** 2
        return np.log(power + 1e-12).astype(np.float32)
    
    def __reshape_input(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Reshape input assuming x has shape (C, T) and y has length T.

        Produces (possibly overlapping) windows along the time axis of length
        `window_size` stepped by `stride` and returns an array shaped
        (n, C, window_size), where n = floor((T - window_size) / stride) + 1.
        """
        C, T = x.shape
        
        if self.window_size <= 0:
            raise ValueError("window_size must be a positive integer.")
        if self.stride <= 0:
            raise ValueError("stride must be a positive integer.")
        if y.shape[0] != T:
            raise ValueError(f"y must have length T={T}, got {y.shape[0]}.")
        if self.window_size > T:
            raise ValueError("window_size must be less than or equal to the length of the input sequence.")

        starts = np.arange(0, T - self.window_size + 1, self.stride, dtype=int)
        if starts.size == 0:
            return (
                np.empty((0, C, self.window_size), dtype=x.dtype),
                np.empty((0, self.window_size), dtype=y.dtype),
            )

        # x windows: slice time axis => each is (C, window_size), stack => (n, C, window_size)
        x_windows = np.stack(
            [x[:, start : start + self.window_size] for start in starts], axis=0
        )

        # y windows: each is (window_size,), stack => (n, window_size)
        y_windows = np.stack(
            [y[start : start + self.window_size] for start in starts], axis=0
        )

        return x_windows, y_windows


    def __downsample_by_majority_voting(self, y_windows: np.ndarray) -> np.ndarray:
        if y_windows.size == 0:
            # preserve original dtype when returning an empty array
            return np.array([], dtype=y_windows.dtype)

        def vote(row):
            vals, counts = np.unique(row, return_counts=True)
            return vals[counts.argmax()]

        return np.apply_along_axis(vote, 1, y_windows)
    
    def __apply_sequence_length(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Reshape input into sequences of length `sequence_length`."""        
        n_windows, C, features = x.shape
        n_sequences = n_windows // self.sequence_length
        x = x[:n_sequences * self.sequence_length].reshape(n_sequences, self.sequence_length, C, features)
        y = y[:n_sequences * self.sequence_length].reshape(n_sequences, self.sequence_length)
        return x, y
    
    def __normalize(self, x: np.ndarray, TYPE: str) -> np.ndarray:
        if TYPE == "pre-norm":
            mean = np.mean(x, axis=(1), keepdims=True)
            std = np.std(x, axis=(1), keepdims=True) +  1e-8
        if TYPE == "raw":
            mean = np.mean(x, axis=(0,1,3), keepdims=True)
            std = np.std(x, axis=(0,1,3), keepdims=True) + 1e-8
            # return x
        elif TYPE == "fft":
            mean = np.mean(x, axis=(0, 1), keepdims=True)
            std = np.std(x, axis=(0, 1), keepdims=True) + 1e-8
        elif TYPE == "stft":
            mean = np.mean(x, axis=(0, 1, 3), keepdims=True)
            std = np.std(x, axis=(0, 1, 3), keepdims=True) + 1e-8
        return (x - mean) / std
    
    def __build_bandpass_sos(self) -> dict[int, np.ndarray]:
        nyq = 0.5 * float(self.sampling_rate)
        sos_by_c = {}

        for c, (low_hz, high_hz) in self._BANDPASS_HZ.items():
            if low_hz <= 0 or high_hz <= 0 or low_hz >= high_hz:
                raise ValueError(f"Invalid band for channel {c}: {(low_hz, high_hz)}")

            low = low_hz / nyq
            high = high_hz / nyq
            if high >= 1.0:
                raise ValueError(
                    f"high_hz={high_hz} for channel {c} must be < Nyquist ({nyq} Hz)"
                )

            sos_by_c[c] = butter(
                N=self._BANDPASS_ORDER,
                Wn=[low, high],
                btype="bandpass",
                output="sos",
            )
        return sos_by_c
    
    def __bandpass_filter_continuous(self, x: np.ndarray) -> np.ndarray:
        """
        x: (C, T) float array
        Returns filtered x with same shape.
        """
        if not self._bandpass_sos:
            return x

        x_f = x.astype(np.float32, copy=True)
        C, _ = x_f.shape

        for c in range(C):
            sos = self._bandpass_sos.get(c, None)
            x_f[c] = sosfiltfilt(sos, x_f[c], axis=-1)

        return x_f
    
    def __cut_down_signal(self, x: np.ndarray) -> np.ndarray:
        """Cut down FFT signal to retain frequencies up to 45 Hz."""
        freq_resolution = self.sampling_rate / self.window_size
        max_freq_eeg = 30.0  # Hz
        max_index_eeg = int(np.floor(max_freq_eeg / freq_resolution)) + 1  # +1 to include max_freq
        min_freq_emg = 30.0
        max_freq_emg = 60.0
        max_index_emg = int(np.floor(max_freq_emg / freq_resolution)) + 1
        min_index_emg = int(np.floor(min_freq_emg / freq_resolution))
        eeg = x[:, :2, :max_index_eeg]
        emg = x[:, 2:, min_index_emg:max_index_emg]
        return np.concatenate((eeg, emg), axis=1)
    
    def percentile_clip_channels(
        self,
        x: np.ndarray,
        lower_percentile: float = 0.5,
        upper_percentile: float = 99.5,
        axis: int = -1,
    ) -> np.ndarray:
        lower = np.percentile(x, lower_percentile, axis=axis, keepdims=True)
        upper = np.percentile(x, upper_percentile, axis=axis, keepdims=True)
        return np.clip(x, lower, upper)
    
    def band_pass_filter_fft(self, x: np.ndarray) -> np.ndarray:
        for c in range(x.shape[1]):
            low, high = self._BANDPASS_HZ.get(c, (None, None))
            if low is not None and high is not None:
                low_fft = int(np.floor(low * self.window_size / self.sampling_rate))
                high_fft = int(np.ceil(high * self.window_size / self.sampling_rate))
                x[:, c, :low_fft] = 0
                x[:, c, high_fft:] = 0
        return x
            
    
    def __call__(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        TYPE = "fft"
        x = self.percentile_clip_channels(x)
        # x = self.__bandpass_filter_continuous(x)
        if TYPE == "fft":
            if self.normalize:
                x = self.__normalize(x, TYPE="pre-norm")
            x, y = self.__reshape_input(x, y)
            # x = self.__perform_hanning_window(x=x)
            x = self.__perform_fft(x=x)
            x = self.band_pass_filter_fft(x)
            x = self.__complex_to_real_features(x)
        elif TYPE == "stft":
            x = self.__perform_stft(x=x)
            x = self.__complex_to_real_features(x)
        elif TYPE == "raw":
            x, y = self.__reshape_input(x, y)
        y = self.__downsample_by_majority_voting(y_windows=y)
        x, y = self.__apply_sequence_length(x, y)
        if self.normalize:
            x = self.__normalize(x, TYPE=TYPE)
        return x, y
    
    def get_short_name(self):
        return "VAEPreprocessing"
    
    def __str__(self) -> str:
        return f"VAEPreprocessing(window_size={self.window_size}, stride={self.stride}, sequence_length={self.sequence_length}, sampling_rate={self.sampling_rate})"