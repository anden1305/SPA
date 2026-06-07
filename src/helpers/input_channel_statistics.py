"""Per-channel input-space statistics for VAE log-power spectra (pre-encoder).

Aggregates log-power FFT bins into interpretable bands per EEG/EMG channel and
state label. Used to diagnose whether REM atonia / theta cues exist before the
encoder, independent of latent `feature_amplitude_per_state` plots.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.data.data_loader_collection import DataLoaderCollection

# Hz bands for staging diagnostics (mouse EEG @ 128 Hz, 512-sample windows).
EEG_BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (6.0, 10.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}
EMG_BANDS: dict[str, tuple[float, float]] = {
    "emg_low": (1.0, 30.0),
    "emg_mid": (30.0, 60.0),
    "emg_total": (1.0, 60.0),
}


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _hz_to_bins(low_hz: float, high_hz: float, freq_resolution: float, n_bins: int) -> slice:
    lo = max(0, int(np.floor(low_hz / freq_resolution)))
    hi = min(n_bins, int(np.ceil(high_hz / freq_resolution)))
    if hi <= lo:
        return slice(0, 0)
    return slice(lo, hi)


def _channel_is_emg(name: str) -> bool:
    return name.upper().startswith("EMG")


def _band_means(
    spectra: np.ndarray,
    freq_resolution: float,
    bands: dict[str, tuple[float, float]],
) -> dict[str, float]:
    n_bins = spectra.shape[-1]
    out: dict[str, float] = {}
    for band_name, (lo_hz, hi_hz) in bands.items():
        sl = _hz_to_bins(lo_hz, hi_hz, freq_resolution, n_bins)
        if sl.stop <= sl.start:
            out[band_name] = float("nan")
        else:
            out[band_name] = float(np.mean(spectra[..., sl]))
    out["total"] = float(np.mean(spectra))
    return out


def compute_input_channel_statistics(
    x: torch.Tensor | np.ndarray,
    y: torch.Tensor | np.ndarray,
    data_loader: DataLoaderCollection,
) -> dict[str, Any]:
    """Compute per-state channel and band summaries from VAE input tensors.

    Expects ``x`` shaped ``(N, S, C, F)`` (log-power spectra entering the conv
    encoder). Returns JSON-serialisable diagnostics under ``input_channel_statistics``.
    """
    x_np = _to_numpy(x)
    y_np = _to_numpy(y)
    if x_np.ndim != 4:
        return {}

    n, s, c, _f = x_np.shape
    y_flat = y_np.reshape(-1).astype(int)
    x_flat = x_np.reshape(n * s, c, -1)
    if y_flat.shape[0] != x_flat.shape[0]:
        return {}

    channels, _n_bins, _seq_len, _ = data_loader.get_vae_dims()
    if channels != c:
        return {}

    window_size = int(data_loader.global_config.dataloader.window_size or 512)
    sampling_rate = float(data_loader.datasets[0].config.get("sampling_rate", 128))
    freq_resolution = sampling_rate / window_size

    signal_names = data_loader.get_channel_signal_names()
    if len(signal_names) != c:
        signal_names = [f"ch{i + 1}" for i in range(c)]

    n_bins = x_flat.shape[-1]
    states = sorted(int(st) for st in np.unique(y_flat))
    per_state: dict[str, Any] = {}
    separation: dict[str, float] = {}

    for state in states:
        mask = y_flat == state
        if not mask.any():
            continue
        state_key = str(state)
        ch_stats: dict[str, Any] = {}
        for ch_idx, ch_name in enumerate(signal_names):
            spec = x_flat[mask, ch_idx, :]
            bands = EMG_BANDS if _channel_is_emg(ch_name) else EEG_BANDS
            means = _band_means(spec, freq_resolution, bands)
            stds: dict[str, float] = {}
            for key, val in means.items():
                if key == "total":
                    stds[key] = float(np.std(spec))
                elif key in bands:
                    sl = _hz_to_bins(*bands[key], freq_resolution, n_bins)
                    stds[key] = float(np.std(spec[..., sl])) if sl.stop > sl.start else float("nan")
                else:
                    stds[key] = float("nan")
            ch_stats[ch_name] = {"mean": means, "std": stds}
        per_state[state_key] = ch_stats

    # Key separation gaps for REM troubleshooting (state 0=Awake, 1=NREM, 2=REM).
    if "2" in per_state and "1" in per_state:
        for ch_name in signal_names:
            if ch_name not in per_state["2"] or ch_name not in per_state["1"]:
                continue
            emg_key = "emg_total" if _channel_is_emg(ch_name) else "total"
            rem_val = per_state["2"][ch_name]["mean"].get(emg_key, float("nan"))
            nrem_val = per_state["1"][ch_name]["mean"].get(emg_key, float("nan"))
            separation[f"{ch_name}_rem_minus_nrem_{emg_key}"] = float(rem_val - nrem_val)
    if "2" in per_state and "0" in per_state:
        emg_channels = [n for n in signal_names if _channel_is_emg(n)]
        if emg_channels:
            ch = emg_channels[0]
            rem_val = per_state["2"][ch]["mean"].get("emg_total", float("nan"))
            awake_val = per_state["0"][ch]["mean"].get("emg_total", float("nan"))
            separation["emg_awake_minus_rem_total"] = float(awake_val - rem_val)

    return {
        "input_channel_statistics": {
            "channels": signal_names,
            "freq_resolution_hz": freq_resolution,
            "eeg_bands_hz": EEG_BANDS,
            "emg_bands_hz": EMG_BANDS,
            "per_state": per_state,
            "separation_gaps": separation,
        }
    }
