"""Helpers tailored for synthetic data exploration.
Reuse plotting colors and simple utility functions.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8')

SYN_SLEEP_STAGE_COLORS = {
    'AWAKE': '#FF6B6B',
    'NREM': '#4ECDC4',
    'REM': '#45B7D1',
}
STAGE_ORDER = ['AWAKE', 'NREM', 'REM']


def ensure_output_directory(base: str = 'results/synthetic_exploration') -> Path:
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p


def compute_epoch_psd(epoch: np.ndarray, fs: int):
    # Simple Welch replacement using FFT (single epoch)
    n = len(epoch)
    win = np.hanning(n)
    spec = np.fft.rfft(epoch * win)
    freqs = np.fft.rfftfreq(n, 1/fs)
    psd = (np.abs(spec)**2) / (fs * np.sum(win**2))
    return freqs, psd


def band_power(freqs, psd, low, high):
    mask = (freqs >= low) & (freqs < high)
    if not mask.any():
        return 0.0
    return psd[mask].sum()
