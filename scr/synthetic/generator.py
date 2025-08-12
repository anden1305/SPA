"""Synthetic sleep EEG/EMG data generator (draft)

This module provides a simple, easily extensible generator for mouse sleep stage
signals using a first‑order Hidden Markov Model (HMM) over coarse sleep stages.

Goals of this draft:
- Intuitive structure – a single class with a small config dict you can tweak.
- Deterministic option via random seed.
- Explicit stage spectral fingerprints (relative band power distributions) you can
  edit in one place.
- Customizable transition matrix (rows = current state, columns = next state).
- Produces: EEG (single channel), optional EMG (single channel), integer labels.

Key simplifications (you can refine later):
- Each epoch is generated independently conditional on its sampled state (no MAR /
  temporal dynamics within an epoch beyond a sum of sinusoids + colored noise).
- Spectral profile per stage implemented as a multinomial over frequency bins.
- Within each chosen bin, a frequency is sampled uniformly.
- Amplitudes scale with the bin weight (band power proxy) + random jitter.

Planned future extensions (not yet implemented):
- Multi‑channel EEG with electrode‑specific noise profiles.
- More realistic state duration modeling (e.g., explicit bout length distribution).
- MAR / AR modeling to create temporal correlations across epochs.
- Separate light vs deep NREM substates (N1/N2/N3) or artifacts class.
- EMG burst modeling for wake micro‑arousals.

Usage example (quick test):
from scr.synthetic.generator import SyntheticSleepGenerator, DEFAULT_CONFIG

gen = SyntheticSleepGenerator(DEFAULT_CONFIG, seed=123)
sequence = gen.generate(epochs=200)  # 200 epochs
print(sequence['eeg'].shape, sequence['labels'][:10])

Config structure (keys):
- sampling_rate_hz: int
- epoch_length_s: int
- stages: list[str] (order defines label integers)
- transition_matrix: list[list[float]] (square, row‑stochastic)
- frequency_bins: list[[low, high]] (Hz, low inclusive, high exclusive)
- stage_band_powers: mapping stage -> list[float] (len == n_bins)
- n_components_per_epoch: int (sinusoids sampled per epoch)
- base_amplitude: float (scales EEG)
- noise:
    gaussian_std: float (additive white noise std)
    colored_beta: float (spectral 1/f^beta factor; 1.0 ~ pink, 0 ~ white)
    colored_scale: float (scales colored noise component)
- emg:
    enabled: bool
    wake_std: float
    rem_std: float
    nrem_std: float
- mains_artifact:
    enabled: bool
    frequency: float (default 50)
    amplitude: float

All numeric arrays are converted to numpy arrays internally.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Any
from pathlib import Path
import json
import numpy as np

"""
HOW IT WORKS:
Here's how the raw EEG is created:

For each epoch, the code generates a signal by summing multiple sinusoids 
(based on n_components_per_epoch), each sinusoid having a frequency sampled 
from a chosen frequency bin (defined in frequency_bins), and a random phase.

So each component picks 1 bin (based on the frequency_bins) and uniformly samples a
frequency within that bin. It also samples a random phase so that the sinusoids start 
at different points in time.

The signal is then scaled based on the band power (weight) for that stage and 
the base_amplitude.

Then, the final signal is obtained by summing all the individual components.
Normalisation occurs after this step, based on the chosen normalization_mode.

Each epoch is synthesized independently, which makes it have hard boundaries between epochs.
This means that transitions between stages (e.g., from NREM to REM) are not gradual but rather abrupt, 
reflecting the discrete nature of sleep stage changes.
"""

# --------------------------- Default Config ---------------------------------- #
DEFAULT_CONFIG: Dict[str, Any] = {
    # Match paper: resampled to 128 Hz, 4-s epochs
    "sampling_rate_hz": 128,
    "epoch_length_s": 4,
    "stages": ["AWAKE", "NREM", "REM"],  # label integers will follow this order
    # Simple example: strong self‑transitions (80%), equal prob to others (10%, 10%)
    "transition_matrix": [
        [0.80, 0.10, 0.10],  # WAKE -> (W, N, R)
        [0.10, 0.80, 0.10],  # NREM -> (W, N, R)
        [0.10, 0.10, 0.80],  # REM  -> (W, N, R)
    ],
    # Coarse frequency bins (Hz). Adjust or extend as needed.
    "frequency_bins": [
        [0.5, 4.0],   # delta
        [4.0, 8.0],   # theta
        [8.0, 12.0],  # alpha
        [12.0, 16.0], # sigma/spindle-ish (broad placeholder)
        [16.0, 22.0], # beta low
        [22.0, 30.0], # beta high
        [30.0, 40.0], # gamma low
        [40.0, 50.0], # gamma mid
        [50.0, 60.0], # gamma high (note: mains artifact separately at 50 Hz)
    ],
    # stage_band_powers now represent RELATIVE AMPLITUDE PROTOTYPES per stage (NOT probabilities).
    # A separate optional mapping 'stage_bin_selection_probs' can provide sampling probabilities over bins.
    # If 'stage_bin_selection_probs' is absent, probabilities are inferred by normalizing these amplitudes (backward compatible).
    "stage_band_powers": {  # amplitude prototypes per bin: delta..gammaH
        "AWAKE": [0.76, 0.484, 0.168, 0.0344, 0.0344, 0.0344, 0.00263, 0.0026, 0.0026],
        "NREM": [0.035, 0.0178, 0.0078, 0.00139, 0.00139, 0.00139, 0.000088, 0.000088, 0.000088],
        "REM":  [0.017, 0.032, 0.0132, 0.00246, 0.00246, 0.00246, 0.0001, 0.0001, 0.0001],
    },
    # Optional explicit per-stage selection probabilities (same length as frequency_bins). Uncomment & fill to override.
    # Below we provide a concrete default separating selection probability from amplitude:
    # - AWAKE: softer (sqrt-based) distribution to keep some diversity while reflecting amplitude trend.
    # - NREM: mirrored (strong delta dominance) to emphasize slow waves.
    # - REM: mirrored to highlight theta dominance.
    # Each list corresponds to bins: [delta, theta, alpha, sigma, betaL, betaH, gammaL, gammaM, gammaH].
    "stage_bin_selection_probs": {
        "AWAKE": [1/9] * 9,
        "NREM":  [1/9] * 9,
        "REM":   [1/9] * 9,
    },
    # How many sinusoids to generate per epoch - number of bands clearly represented in an epoch
    # - 1-2 would be unreasonably simple and 15+ would be very complex
    "n_components_per_epoch": 20,
    "base_amplitude": 1.0,  # Target global std in microvolts (before scaling)
    # How to normalize amplitudes after synthesis:
    #   'per_epoch' -> each epoch individually scaled to have std ~= base_amplitude (old behavior)
    #   'global'   -> scale entire sequence once (preserves relative epoch differences)
    #   'none'     -> no normalization (raw mixture amplitudes)
    "normalization_mode": "global",  # 'per_epoch'|'global'|'none'
    # Generation mode: 'epoch' (independent epochs, random phases) | 'continuous' (phase continuity across epochs)
    "generation_mode": "epoch", # 'continuous'
    # Post-synthesis preprocessing (disabled by default). Enable by setting bandpass and/or median_iqr_scale True.
    "preprocess": {
        "bandpass": None,        # e.g. [0.3, 35.0]
        "median_iqr_scale": False,
    },
    "noise": {
        "enabled": False,          # master switch for any noise below
        "gaussian_std": 2.0,
        "colored_beta": 1.0,      # 1/f^beta shaping for colored noise (1 ~ pink)
        "colored_scale": 3.0,
    },
    # EMG off by default (set enabled True to include)
    "emg": {
        "enabled": False,
        "wake_std": 12.0,
        "rem_std": 3.0,
        "nrem_std": 5.0,
    },
    # Mains artifact off by default (measurement noise you don't want in synthetic)
    "mains_artifact": {
        "enabled": False,
        "frequency": 50.0,
        "amplitude": 5.0,
    },
}

# --------------------------- Helper Functions -------------------------------- #

def _validate_transition_matrix(tm: np.ndarray):
    if tm.ndim != 2 or tm.shape[0] != tm.shape[1]:
        raise ValueError("Transition matrix must be square")
    row_sums = tm.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError("Each row of transition matrix must sum to 1. Row sums: " + str(row_sums))


def _normalize(weights: Sequence[float]) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    total = w.sum()
    if total <= 0:
        raise ValueError("Weights must sum to > 0")
    return w / total


def _colored_noise(n: int, beta: float, rng: np.random.Generator) -> np.ndarray:
    """Generate approximate 1/f^beta noise using frequency-domain shaping."""
    if beta <= 1e-8:
        return rng.normal(0, 1, size=n)
    freqs = np.fft.rfftfreq(n, d=1.0)
    # Avoid divide by zero at DC.
    scale = np.ones_like(freqs)
    nonzero = freqs > 0
    scale[nonzero] = 1 / (freqs[nonzero] ** (beta / 2.0))
    white = rng.normal(0, 1, size=freqs.shape)
    # Random phases for real FFT synthesis.
    phases = rng.uniform(0, 2 * np.pi, size=freqs.shape)
    spectrum = white * scale * np.exp(1j * phases)
    time = np.fft.irfft(spectrum, n=n)
    return time

# --------------------------- Main Generator Class ---------------------------- #

@dataclass
class SyntheticSleepGenerator:
    config: Dict[str, Any]
    seed: int | None = None

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.sampling_rate = int(self.config["sampling_rate_hz"])
        self.epoch_length_s = int(self.config["epoch_length_s"])
        self.samples_per_epoch = self.sampling_rate * self.epoch_length_s
        self.stages: List[str] = list(self.config["stages"])
        self.stage_to_int = {s: i for i, s in enumerate(self.stages)}
        tm = np.asarray(self.config["transition_matrix"], dtype=float)
        _validate_transition_matrix(tm)
        if tm.shape[0] != len(self.stages):
            raise ValueError("Transition matrix size does not match number of stages")
        self.transition_matrix = tm
        self.frequency_bins: List[Tuple[float, float]] = [tuple(b) for b in self.config["frequency_bins"]]
        self.n_bins = len(self.frequency_bins)
        # Amplitude prototypes (non-normalized) per stage
        self.stage_band_amplitudes: Dict[str, np.ndarray] = {}
        raw_amp_cfg = self.config.get("stage_band_powers", {})  # keep key name for backward compatibility
        for stage, amps in raw_amp_cfg.items():
            if stage not in self.stage_to_int:
                raise ValueError(f"Stage {stage} in stage_band_powers not in stages list")
            if len(amps) != self.n_bins:
                raise ValueError(f"Stage {stage} has {len(amps)} amplitudes but expected {self.n_bins}")
            arr = np.asarray(amps, dtype=float)
            if np.any(arr < 0):
                raise ValueError("Amplitude prototypes must be non-negative")
            if not np.any(arr > 0):
                raise ValueError("At least one amplitude per stage must be > 0")
            self.stage_band_amplitudes[stage] = arr
        # Selection probabilities (explicit or inferred by normalizing amplitudes)
        self.stage_bin_selection_probs: Dict[str, np.ndarray] = {}
        explicit_sel = self.config.get("stage_bin_selection_probs")
        if explicit_sel is not None:
            for stage, probs in explicit_sel.items():
                if stage not in self.stage_to_int:
                    raise ValueError(f"Stage {stage} in stage_bin_selection_probs not in stages list")
                if len(probs) != self.n_bins:
                    raise ValueError(f"Stage {stage} selection probs length {len(probs)} != {self.n_bins}")
                self.stage_bin_selection_probs[stage] = _normalize(probs)
        else:
            for stage, arr in self.stage_band_amplitudes.items():
                self.stage_bin_selection_probs[stage] = _normalize(arr)
        self.n_components = int(self.config.get("n_components_per_epoch", 6))
        self.base_amplitude = float(self.config.get("base_amplitude", 10.0))
        self.normalization_mode = self.config.get("normalization_mode", "global")
        self.generation_mode = self.config.get("generation_mode", "epoch")
        self.noise_cfg = self.config.get("noise", {})
        self.emg_cfg = self.config.get("emg", {})
        self.mains_cfg = self.config.get("mains_artifact", {})
        self.pre_cfg = self.config.get("preprocess", {})
        # Design bandpass filter once if requested
        self._bp_sos = None
        bp = self.pre_cfg.get("bandpass")
        if bp is not None:
            try:
                from scipy.signal import butter, sosfiltfilt
                nyq = 0.5 * self.sampling_rate
                low, high = bp
                if high >= nyq:
                    high = nyq - 1e-3
                sos = butter(4, [low/nyq, high/nyq], btype="band", output="sos")
                self._bp_sos = sos
            except Exception:
                self._bp_sos = None  # silently skip if scipy missing at runtime

    # --------------------- Public API --------------------- #
    def generate(
        self,
        epochs: int,
        initial_stage: str | None = None,
        clean: bool = False,
        save: bool = False,
        save_dir: str | Path = Path("data") / "synthetic",
        prefix: str = "synthetic"
    ) -> Dict[str, Any]:
        """Generate a synthetic dataset.

        Returns dict with keys: eeg (epochs, samples), labels (epochs,), optionally emg.
        If save=True, writes:
          <save_dir>/<prefix>_eeg.npy
          <save_dir>/<prefix>_labels.npy
          <save_dir>/<prefix>_metadata.json
        
        Parameters
        ----------
        epochs : int
            Number of epochs to synthesize.
        initial_stage : str | None
            Force the initial sleep stage. If None (default) the generator now
            attempts to start in 'WAKE' (case-insensitive) when that stage
            exists in the configured stage list; if not present it falls back
            to a uniform random initial stage (previous behaviour).
        """
        # Dispatch if continuous mode requested
        if getattr(self, "generation_mode", "epoch") == "continuous":
            return self._generate_continuous_phase(
                epochs=epochs,
                initial_stage=initial_stage,
                clean=clean,
                save=save,
                save_dir=save_dir,
                prefix=prefix,
            )

        labels = self._sample_state_sequence(epochs, initial_stage)
        # Optionally force a clean generation (no noise, no artifact, no EMG)
        if clean:
            prev_noise_enabled = self.noise_cfg.get("enabled", False)
            prev_mains_enabled = self.mains_cfg.get("enabled", False)
            prev_emg_enabled = self.emg_cfg.get("enabled", False)
            self.noise_cfg["enabled"] = False
            self.mains_cfg["enabled"] = False
            self.emg_cfg["enabled"] = False
        else:
            prev_noise_enabled = prev_mains_enabled = prev_emg_enabled = None
        eeg = np.zeros((epochs, self.samples_per_epoch), dtype=float)
        emg = np.zeros_like(eeg) if self.emg_cfg.get("enabled", False) else None
        t = np.arange(self.samples_per_epoch) / self.sampling_rate
        raw_epochs = []  # store for possible global normalization

        for i, state_idx in enumerate(labels):
            stage = self.stages[state_idx]
            epoch_sig = self._synthesize_epoch(stage, t)
            raw_epochs.append(epoch_sig)
            if self.normalization_mode == "per_epoch":
                std = np.std(epoch_sig)
                if std > 0:
                    epoch_sig = epoch_sig / std * self.base_amplitude
            eeg[i] = epoch_sig
            if emg is not None:
                emg[i] = self._synthesize_emg(stage)
        if self.normalization_mode == "global":
            global_std = np.std(eeg)
            if global_std > 0:
                eeg = eeg / global_std * self.base_amplitude
        # Post-processing: bandpass + median/IQR scaling
        eeg = self._postprocess(eeg)
        if clean:
            # restore original flags
            self.noise_cfg["enabled"] = prev_noise_enabled
            self.mains_cfg["enabled"] = prev_mains_enabled
            self.emg_cfg["enabled"] = prev_emg_enabled
        # Epoch-wise matrix (epochs, samples) kept for higher-level feature extraction.
        out = {"eeg": eeg, "labels": labels}
        # Additionally create a continuous flattened raw signal that can directly feed a
        # sample-level HMM (obs_dim = 1). We provide both 1D and 2D variants.
        continuous = eeg.reshape(-1)  # (epochs * samples_per_epoch,)
        out["eeg_raw"] = continuous
        out["eeg_raw_2d"] = continuous[:, None]  # (T,1)
        if emg is not None:
            out["emg"] = emg
        out["metadata"] = {
            "stages": self.stages,
            "frequency_bins": self.frequency_bins,
            "config_used": self.config,
            "clean": clean,
        }
        if save:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            np.save(save_path / f"{prefix}_eeg.npy", eeg)
            np.save(save_path / f"{prefix}_labels.npy", labels)
            np.save(save_path / f"{prefix}_eeg_raw.npy", continuous)
            # metadata JSON (ensure serializable)
            meta = out["metadata"].copy()
            with open(save_path / f"{prefix}_metadata.json", "w") as f:
                json.dump(meta, f, indent=2)
        return out

    # --------------------- Internals ---------------------- #
    def _sample_state_sequence(self, epochs: int, initial_stage: str | None) -> np.ndarray:
        # Resolve initial stage. If caller supplies one, accept case-insensitively.
        # If none supplied, prefer 'WAKE' if available; else fall back to previous
        # behaviour (uniform random start).
        if initial_stage is None:
            start_key = None
            for s in self.stages:
                if s.lower() in ("wake", "awake"):
                    start_key = s
                    break
            if start_key is not None:
                current = self.stage_to_int[start_key]
            else:
                current = self.rng.integers(0, len(self.stages))
        else:
            # Case-insensitive lookup
            lowered = {s.lower(): s for s in self.stages}
            lookup = str(initial_stage).lower()
            # Accept 'wake' as alias for 'awake' for backward compatibility
            if lookup == "wake" and "awake" in lowered:
                lookup = "awake"
            key = lowered.get(lookup)
            if key is None:
                raise ValueError(f"Initial stage '{initial_stage}' not found in stages {self.stages}")
            current = self.stage_to_int[key]
        labels = np.empty(epochs, dtype=int)
        for i in range(epochs):
            labels[i] = current
            probs = self.transition_matrix[current]
            current = self.rng.choice(len(self.stages), p=probs)
        return labels

    def _synthesize_epoch(self, stage: str, t: np.ndarray) -> np.ndarray:
        # Sum of sinusoids sampled from stage-specific band distribution.
        sel_probs = self.stage_bin_selection_probs[stage]
        amp_proto = self.stage_band_amplitudes[stage]
        max_amp = amp_proto.max() if amp_proto.size else 1.0
        signal = np.zeros_like(t)
        for _ in range(self.n_components):
            bin_idx = self.rng.choice(self.n_bins, p=sel_probs)
            f_low, f_high = self.frequency_bins[bin_idx]
            freq = self.rng.uniform(f_low, f_high)
            phase = self.rng.uniform(0, 2 * np.pi)
            # Amplitude purely from prototype (relative to max) * jitter * base_amplitude
            amp_scale = (amp_proto[bin_idx] / max_amp) if max_amp > 0 else 0.0
            amplitude = self.base_amplitude * amp_scale * self.rng.uniform(0.8, 1.2)
            signal += amplitude * np.sin(2 * np.pi * freq * t + phase)
        # Optional noise components
        if self.noise_cfg.get("enabled", True):
            beta = float(self.noise_cfg.get("colored_beta", 1.0))
            colored = _colored_noise(len(t), beta, self.rng)
            colored *= float(self.noise_cfg.get("colored_scale", 1.0))
            white = self.rng.normal(0, float(self.noise_cfg.get("gaussian_std", 1.0)), size=len(t))
            signal += colored + white
        # Add mains artifact if enabled
        if self.mains_cfg.get("enabled", False):
            mains_freq = float(self.mains_cfg.get("frequency", 50.0))
            mains_amp = float(self.mains_cfg.get("amplitude", 5.0))
            signal += mains_amp * np.sin(2 * np.pi * mains_freq * t)
        if self.normalization_mode == "none":
            return signal
        # When per_epoch, normalization is applied in generate(); for global we leave raw here.
        return signal

    def _synthesize_emg(self, stage: str) -> np.ndarray:
        std_map = {
            "AWAKE": float(self.emg_cfg.get("wake_std", 10.0)),  # key renamed from WAKE -> AWAKE
            "REM": float(self.emg_cfg.get("rem_std", 4.0)),
            "NREM": float(self.emg_cfg.get("nrem_std", 6.0)),
            # Backward compatibility if someone passes 'WAKE' explicitly
            "WAKE": float(self.emg_cfg.get("wake_std", 10.0)),
        }
        std = std_map.get(stage, 5.0)
        # EMG as high‑frequency noise bandpassed synthetic (simplified as white noise) + mains artifact if any.
        emg = self.rng.normal(0, std, size=self.samples_per_epoch)
        if self.mains_cfg.get("enabled", False):
            mains_freq = float(self.mains_cfg.get("frequency", 50.0))
            mains_amp = float(self.mains_cfg.get("amplitude", 5.0)) * 0.5
            t = np.arange(self.samples_per_epoch) / self.sampling_rate
            emg += mains_amp * np.sin(2 * np.pi * mains_freq * t)
        return emg

    # ---------------- Convenience ------------------------- #
    def sample_one_epoch(self, stage: str) -> np.ndarray:
        t = np.arange(self.samples_per_epoch) / self.sampling_rate
        return self._synthesize_epoch(stage, t)

    def _postprocess(self, eeg: np.ndarray) -> np.ndarray:
        # Bandpass
        if self._bp_sos is not None:
            try:
                from scipy.signal import sosfiltfilt
                eeg = np.array([sosfiltfilt(self._bp_sos, ep) for ep in eeg])
            except Exception:
                pass
        # Median/IQR scaling (per paper) per "channel" (epochs treated independently for amplitude, we apply across flattened epochs concatenated)
        if self.pre_cfg.get("median_iqr_scale", False):
            # Flatten across epochs to compute robust stats, then apply per-epoch shift/scale
            flat = eeg.flatten()
            med = np.median(flat)
            q1, q3 = np.percentile(flat, [25, 75])
            iqr = q3 - q1
            if iqr > 0:
                eeg = (eeg - med) / iqr
            else:
                eeg = eeg - med
        return eeg

    # ---------------- Continuous Generation ----------------- #
    def _generate_continuous_phase(
        self,
        epochs: int,
        initial_stage: str | None,
        clean: bool,
        save: bool,
        save_dir: str | Path,
        prefix: str,
    ) -> Dict[str, Any]:
        """Generate continuous signal with phase continuity across epochs.

        Approach:
          - Sample state sequence (epochs).
          - Maintain persistent phases for each of n_components oscillators.
          - For each epoch, each component samples a bin (selection probs) and a frequency inside it.
          - Phase offset for each component carried over for continuity.
          - Noise (colored + white) added over entire sequence (unless clean).
        """
        """Phase continuity explanation:
        Each oscillator/component keeps a running phase value stored in the 'phases' array.
        For an epoch of duration T_epoch with sampled frequency f_c, the phase increment is
            Δφ_c = 2π f_c * T_epoch.
        After synthesizing the samples for that epoch we advance phases[c] by Δφ_c (mod 2π).
        The next epoch's sinusoid for that component starts at this updated phase, so the
        instantaneous value at the boundary equals the final value of the previous epoch,
        preventing discontinuities. If frequency changes between epochs the waveform's
        instantaneous phase is still continuous; only its instantaneous frequency (slope
        of phase) changes at the boundary. This mimics natural gradual drifts while
        avoiding artificial hard resets present in independent-epoch synthesis.
        """
        labels = self._sample_state_sequence(epochs, initial_stage)
        if clean:
            prev_noise_enabled = self.noise_cfg.get("enabled", False)
            prev_mains_enabled = self.mains_cfg.get("enabled", False)
            prev_emg_enabled = self.emg_cfg.get("enabled", False)
            self.noise_cfg["enabled"] = False
            self.mains_cfg["enabled"] = False
            self.emg_cfg["enabled"] = False
        else:
            prev_noise_enabled = prev_mains_enabled = prev_emg_enabled = None
        total_samples = epochs * self.samples_per_epoch
        t_global = np.arange(total_samples) / self.sampling_rate
        signal = np.zeros(total_samples)
        phases = self.rng.uniform(0, 2 * np.pi, size=self.n_components)
        epoch_duration = self.epoch_length_s
        emg = np.zeros((epochs, self.samples_per_epoch), dtype=float) if self.emg_cfg.get("enabled", False) else None
        for e in range(epochs):
            stage = self.stages[labels[e]]
            sel_probs = self.stage_bin_selection_probs[stage]
            amp_proto = self.stage_band_amplitudes[stage]
            max_amp = amp_proto.max() if amp_proto.size else 1.0
            start = e * self.samples_per_epoch
            end = start + self.samples_per_epoch
            t_seg = t_global[start:end]
            bin_indices = self.rng.choice(self.n_bins, size=self.n_components, p=sel_probs, replace=True)
            freqs = np.array([
                self.rng.uniform(self.frequency_bins[b][0], self.frequency_bins[b][1]) for b in bin_indices
            ])
            amp_scales = np.where(max_amp > 0, amp_proto[bin_indices] / max_amp, 0.0)
            amplitudes = self.base_amplitude * amp_scales * self.rng.uniform(0.8, 1.2, size=self.n_components)
            for c in range(self.n_components):
                signal[start:end] += amplitudes[c] * np.sin(2 * np.pi * freqs[c] * t_seg + phases[c])
                phases[c] = (phases[c] + 2 * np.pi * freqs[c] * epoch_duration) % (2 * np.pi)
            if emg is not None:
                emg[e] = self._synthesize_emg(stage)
        if self.noise_cfg.get("enabled", True):
            beta = float(self.noise_cfg.get("colored_beta", 1.0))
            colored = _colored_noise(total_samples, beta, self.rng)
            colored *= float(self.noise_cfg.get("colored_scale", 1.0))
            white = self.rng.normal(0, float(self.noise_cfg.get("gaussian_std", 1.0)), size=total_samples)
            signal += colored + white
        if self.mains_cfg.get("enabled", False):
            mains_freq = float(self.mains_cfg.get("frequency", 50.0))
            mains_amp = float(self.mains_cfg.get("amplitude", 5.0))
            signal += mains_amp * np.sin(2 * np.pi * mains_freq * t_global)
        if self.normalization_mode == "per_epoch":
            eeg_epochs = signal.reshape(epochs, self.samples_per_epoch).copy()
            for e in range(epochs):
                std = eeg_epochs[e].std()
                if std > 0:
                    eeg_epochs[e] = eeg_epochs[e] / std * self.base_amplitude
            signal = eeg_epochs.reshape(-1)
        elif self.normalization_mode == "global":
            std = signal.std()
            if std > 0:
                signal = signal / std * self.base_amplitude
        eeg_epochs = signal.reshape(epochs, self.samples_per_epoch)
        eeg_epochs = self._postprocess(eeg_epochs)
        continuous = eeg_epochs.reshape(-1)
        out = {
            "eeg": eeg_epochs,
            "labels": labels,
            "eeg_raw": continuous,
            "eeg_raw_2d": continuous[:, None],
            "metadata": {
                "stages": self.stages,
                "frequency_bins": self.frequency_bins,
                "mode": "continuous",
                "config_used": self.config,
            },
        }
        if emg is not None:
            out["emg"] = emg
        if clean:
            self.noise_cfg["enabled"] = prev_noise_enabled
            self.mains_cfg["enabled"] = prev_mains_enabled
            self.emg_cfg["enabled"] = prev_emg_enabled
        if save:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            np.save(save_path / f"{prefix}_eeg.npy", eeg_epochs)
            np.save(save_path / f"{prefix}_labels.npy", labels)
            np.save(save_path / f"{prefix}_eeg_raw.npy", continuous)
            meta = out["metadata"].copy()
            with open(save_path / f"{prefix}_metadata.json", "w") as f:
                json.dump(meta, f, indent=2)
        return out


__all__ = ["SyntheticSleepGenerator", "DEFAULT_CONFIG"]

if __name__ == "__main__":  # Simple default generation & save
    gen = SyntheticSleepGenerator(DEFAULT_CONFIG, seed=42)
    data = gen.generate(epochs=10000, save=True, prefix="synthetic_default", clean=False)
    print("Saved synthetic dataset (128 Hz, 4-s epochs, no preprocessing) to data/synthetic with prefix 'synthetic_default'.")
    print("EEG shape:", data["eeg"].shape, "Labels distribution:", {int(i):int((data['labels']==i).sum()) for i in np.unique(data['labels'])})
