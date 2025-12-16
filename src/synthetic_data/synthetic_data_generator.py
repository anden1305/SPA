"""Unified synthetic EEG generator with hidden stage sequence.

Supports two modes:
1. Spectral (Sinusoidal) - original mode
2. MAR (Multivariate Autoregressive) - new mode

Reads a YAML config and produces:
    - eeg.npy (continuous 1D signal)
    - labels.npy (epoch stage indices)
    - config_copy.yml (exact config used)

Directory: data/synthetic_data/<name>/

Config (YAML) required keys:
  name: str                      # Output directory name
  sampling_rate_hz: int          # Samples per second
  epoch_length_s: int            # Epoch length in seconds
  n_epochs: int                  # Number of epochs to synthesize
  n_stages: int                  # Number of discrete stages
  transition_matrix: list[list[float]]  # n_stages x n_stages, rows sum to 1
  
  # Mode specific keys:
  # Spectral:
  frequency_bins: list[[low, high]]     
  stage_band_powers: list[list[float]]  
  stage_bin_selection_probability: list[list[float]] 
  components_per_epoch: int      
  
  # MAR:
  type: 'mar' (or mar_mode: True)
  lags: list[int] (optional, default [1])
  stage_coeffs: list[list[float]] (optional)

  # Common:
  stage_mean_amplitudes: list[float] (optional) 
  stage_rms_scales: list[float] (optional) 
  normalize_epoch: bool (optional, default True) 
  seed: int (optional)           

Run as script:
    python -m src.synthetic_data.synthetic_data_generator path/to/config.yml
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence
import numpy as np
import yaml
import sys

# ------------------------------- Core Class ---------------------------------

@dataclass
class SyntheticEEGGenerator:
    """
    Unified generator that can produce either:
    1. Spectral (Sinusoidal) data
    2. MAR (Autoregressive) data
    """
    name: str
    sampling_rate: int
    epoch_length_s: int
    n_epochs: int
    n_stages: int
    transition_matrix: np.ndarray  # (S,S)
    
    # --- Spectral Mode Parameters ---
    frequency_bins: List[tuple] | None = None
    stage_band_powers: np.ndarray | None = None 
    stage_bin_probs: np.ndarray | None = None    
    components_per_epoch: int = 1
    
    # --- MAR Mode Parameters ---
    mar_mode: bool = False
    mar_lags: List[int] | None = None
    # Optional manual MAR coeffs (S, n_lags)
    stage_ar_coeffs: np.ndarray | None = None 
    noise_scale: float = 1.0

    # --- Common Parameters ---
    seed: int | None = None
    jitter_low: float = 0.8
    jitter_high: float = 1.2
    stage_mean_offsets: np.ndarray | None = None  
    stage_rms_scales: np.ndarray | None = None    
    normalize_epoch: bool = True                  
    overlap_samples: int = 0 

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.samples_per_epoch = self.sampling_rate * self.epoch_length_s
        self.total_samples = self.samples_per_epoch * self.n_epochs
        
        # 1. Validate Transition Matrix
        if self.transition_matrix.shape != (self.n_stages, self.n_stages):
            raise ValueError("transition_matrix shape mismatch")
        if not np.allclose(self.transition_matrix.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("Each row of transition_matrix must sum to 1.0")
        
        # 2. Initialize Common Offsets/Scales
        if self.stage_mean_offsets is None:
            self.stage_mean_offsets = np.zeros(self.n_stages, dtype=float)
        if self.stage_rms_scales is None:
            # Default to 1.0 scale if not provided
            self.stage_rms_scales = np.ones(self.n_stages, dtype=float)

        # 3. Mode Specific Initialization
        if self.mar_mode:
            self._init_mar_mode()
        else:
            self._init_spectral_mode()

        # Internal state tracking
        self._initialized = False
        self._mar_history = None # Keeps track of last P samples for continuity

    def _init_spectral_mode(self):
        """Validate and setup Sinusoidal generation."""
        if self.frequency_bins is None:
            raise ValueError("Spectral mode requires 'frequency_bins'")
        n_bins = len(self.frequency_bins)
        if self.stage_band_powers.shape != (self.n_stages, n_bins):
            raise ValueError("stage_band_powers shape mismatch")
        
        # Component tracking (Phase continuity)
        self._component_phases = np.zeros(self.components_per_epoch)
        self._component_freqs = np.zeros(self.components_per_epoch)

    def _init_mar_mode(self):
        """Validate and setup AR generation."""
        if not self.mar_lags:
            # Default to lag 1 if not specified
            self.mar_lags = [1]
        
        self.max_lag = max(self.mar_lags)
        n_coeffs = len(self.mar_lags)

        # If coefficients not provided, generate random STABLE ones
        if self.stage_ar_coeffs is None:
            print("MAR Mode: No coefficients provided. Generating random stable AR processes...")
            self.stage_ar_coeffs = np.zeros((self.n_stages, n_coeffs))
            for s in range(self.n_stages):
                # Simple strategy: Generate random roots inside unit circle
                # or just small random numbers for lag 1 to ensure stability |a| < 1
                if self.max_lag == 1:
                    self.stage_ar_coeffs[s, 0] = self.rng.uniform(0.5, 0.95) * self.rng.choice([-1, 1])
                else:
                    # Fallback for multi-lag: small random weights scaled down by number of lags
                    self.stage_ar_coeffs[s, :] = self.rng.uniform(-0.5, 0.5, size=n_coeffs) / n_coeffs

        # Validate provided / generated coefficients for stability & shape
        self._validate_mar_coeffs()

    def _validate_mar_coeffs(self):
        """Assert that stage_ar_coeffs are well-formed and stable.

        Stability criterion: all roots of the AR characteristic polynomial
        1 - sum_k a_k z^{lag_k} must lie outside the unit circle.
        """
        if self.stage_ar_coeffs is None:
            raise ValueError("MAR mode requires stage_ar_coeffs after initialization")
        expected_shape = (self.n_stages, len(self.mar_lags))
        if self.stage_ar_coeffs.shape != expected_shape:
            raise ValueError(f"stage_ar_coeffs shape {self.stage_ar_coeffs.shape} != {expected_shape}")

        for s in range(self.n_stages):
            coeffs = self.stage_ar_coeffs[s]
            
            # Build characteristic polynomial: 1 - sum(coeff * z^lag)
            poly = np.zeros(self.max_lag + 1)
            poly[0] = 1.0
            for lag, coeff in zip(self.mar_lags, coeffs):
                poly[lag] = -coeff
            
            # Check stability: all roots must be outside unit circle
            try:
                roots = np.roots(poly)
                max_root_magnitude = np.max(np.abs(roots))
                if max_root_magnitude >= 1.0:
                    raise ValueError(
                        f"Unstable AR coefficients in stage {s+1}: "
                        f"max root magnitude = {max_root_magnitude:.4f} (must be < 1.0). "
                        f"Coefficients: {coeffs}"
                    )
            except np.linalg.LinAlgError as e:
                raise ValueError(f"Failed to compute roots for stage {s+1}: {e}")

    # ------------------------- Public API -------------------------
    def generate(self):
        labels = self._sample_stage_sequence()
        eeg = np.zeros(self.total_samples, dtype=float)
        cursor = 0
        
        # For continuity smoothing (Spectral mode)
        prev_last_val = None
        win_n = int(self.overlap_samples) if self.overlap_samples > 0 else 0
        
        # For continuity (MAR mode)
        # Start with zero history
        if self.mar_mode:
            self._mar_history = np.zeros(self.max_lag)

        for epoch_idx, stage in enumerate(labels):
            # --- Branch Logic Here ---
            if self.mar_mode:
                epoch_sig = self._synthesize_epoch_mar(stage)
            else:
                epoch_sig = self._synthesize_epoch_spectral(stage)
                
            # Apply linear smoothing (for both modes, to handle mean/scale shifts)
            if prev_last_val is not None and win_n > 0:
                delta = float(epoch_sig[0] - prev_last_val)
                ramp = np.linspace(delta, 0.0, win_n, endpoint=False, dtype=float)
                n_avail = min(win_n, epoch_sig.shape[0])
                epoch_sig[:n_avail] -= ramp[:n_avail]
            
            prev_last_val = float(epoch_sig[-1])

            # Store
            eeg[cursor:cursor + self.samples_per_epoch] = epoch_sig
            cursor += self.samples_per_epoch
            
        return eeg, labels

    def save(self, eeg: np.ndarray, labels: np.ndarray, config_path: Path):
        new_base = Path("data") / "synthetic_data"
        out_dir = new_base / self.name
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "eeg.npy", eeg)
        np.save(out_dir / "labels.npy", labels)
        target_cfg = out_dir / "config_copy.yml"
        if config_path.exists():
            target_cfg.write_text(config_path.read_text())
        print(f"Saved synthetic data to {out_dir}")

    # ------------------------ Internals --------------------------
    def _sample_stage_sequence(self) -> np.ndarray:
        labels = np.zeros(self.n_epochs, dtype=int)
        labels[0] = self.rng.integers(0, self.n_stages)
        for i in range(1, self.n_epochs):
            labels[i] = self.rng.choice(self.n_stages, p=self.transition_matrix[labels[i-1]])
        return labels

    def _synthesize_epoch_spectral(self, stage: int) -> np.ndarray:
        """Original sinusoidal logic."""
        n = self.samples_per_epoch
        t = np.arange(n) / self.sampling_rate
        epoch_sig = np.zeros(n, dtype=float)
        
        bin_probs = self.stage_bin_probs[stage]
        band_powers = self.stage_band_powers[stage]
        chosen_bins = self.rng.choice(len(self.frequency_bins), size=self.components_per_epoch, p=bin_probs)
        
        if not self._initialized:
            for c in range(self.components_per_epoch):
                b = chosen_bins[c]
                f_low, f_high = self.frequency_bins[b]
                self._component_freqs[c] = self.rng.uniform(f_low, f_high)
                self._component_phases[c] = self.rng.uniform(0, 2*np.pi)
            self._initialized = True

        for c in range(self.components_per_epoch):
            b = chosen_bins[c]
            f_low, f_high = self.frequency_bins[b]
            
            # Simple frequency drift logic
            if self.rng.random() < 0.3:
                new_f = np.clip(self._component_freqs[c] + self.rng.normal(0, (f_high - f_low)/10), f_low, f_high)
            else:
                new_f = self.rng.uniform(f_low, f_high)
            self._component_freqs[c] = new_f
            
            amp = band_powers[b] * self.rng.uniform(self.jitter_low, self.jitter_high)
            phase0 = self._component_phases[c]
            
            component = amp * np.sin(2 * np.pi * new_f * t + phase0)
            epoch_sig += component
            
            end_phase = (phase0 + 2 * np.pi * new_f * (n / self.sampling_rate)) % (2 * np.pi)
            self._component_phases[c] = end_phase

        return self._post_process(epoch_sig, stage)

    def _synthesize_epoch_mar(self, stage: int) -> np.ndarray:
        """Simplified 1D AR synthesis."""
        n = self.samples_per_epoch
        
        # Get coefficients for this stage: shape (n_lags,)
        coeffs = self.stage_ar_coeffs[stage] 
        
        # Generate white noise innovation
        noise = self.rng.normal(0, self.noise_scale, size=n)
        
        # Pre-allocate output signal
        sig = np.zeros(n)
        
        # AR Process: x[t] = noise[t] + sum(coeff[i] * x[t-lag[i]])
        for t in range(n):
            val = noise[t]
            
            for i, lag in enumerate(self.mar_lags):
                # Look back 'lag' samples from current position t
                lookback_idx = t - lag
                
                if lookback_idx >= 0:
                    # Use current epoch's signal
                    val += coeffs[i] * sig[lookback_idx]
                    
                else:
                    # Reach into history from previous epoch
                    history_idx = self.max_lag + lookback_idx  # lookback_idx is negative
                    val += coeffs[i] * self._mar_history[history_idx]
            
            sig[t] = val
        
        # Update history for next epoch (take the last max_lag samples)
        self._mar_history = sig[-self.max_lag:]
        
        return self._post_process(sig, stage)

    def _post_process(self, sig: np.ndarray, stage: int) -> np.ndarray:
        """Common normalization and scaling."""
        std = np.std(sig)
        if self.normalize_epoch and std > 0:
            sig = sig / std
        
        if self.stage_rms_scales is not None:
            sig = sig * float(self.stage_rms_scales[stage])
            
        sig = sig + float(self.stage_mean_offsets[stage])
        return sig


# --------------------------- YAML Loader -------------------------------------

def load_config(path: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    return data

def build_generator(cfg: dict):
    name = cfg['name']
    sr = int(cfg['sampling_rate_hz'])
    epoch_len = int(cfg['epoch_length_s'])
    n_epochs = int(cfg['n_epochs'])
    n_stages = int(cfg['n_stages'])
    tm = np.asarray(cfg['transition_matrix'], dtype=float)
    seed = int(cfg.get('seed')) if 'seed' in cfg else None
    
    # Detect Mode
    is_mar = cfg.get('type') == 'mar' or cfg.get('mar_mode', False)
    
    # Common optionals
    mean_offsets = np.asarray(cfg['stage_mean_amplitudes']) if 'stage_mean_amplitudes' in cfg else None
    rms_scales = np.asarray(cfg['stage_rms_scales']) if 'stage_rms_scales' in cfg else None
    normalize = bool(cfg.get('normalize_epoch', True))
    overlap_s = float(cfg.get('overlap_s', 0))
    overlap_samples = int(overlap_s * sr)

    if is_mar:
        # MAR Params
        lags = [int(l) for l in cfg.get('lags', [1])]
        coeffs = None
        if 'stage_coeffs' in cfg:
            coeffs = np.asarray(cfg['stage_coeffs'], dtype=float)
        
        return SyntheticEEGGenerator(
            name=name, sampling_rate=sr, epoch_length_s=epoch_len, n_epochs=n_epochs,
            n_stages=n_stages, transition_matrix=tm, seed=seed,
            mar_mode=True, mar_lags=lags, stage_ar_coeffs=coeffs,
            stage_mean_offsets=mean_offsets, stage_rms_scales=rms_scales, normalize_epoch=normalize,
            overlap_samples=overlap_samples
        )
    else:
        # Spectral Params
        freq_bins = [tuple(b) for b in cfg['frequency_bins']]
        stage_band_powers = np.asarray(cfg['stage_band_powers'], dtype=float)
        stage_bin_probs = np.asarray(cfg['stage_bin_selection_probability'], dtype=float)
        components = int(cfg['components_per_epoch'])

        return SyntheticEEGGenerator(
            name=name, sampling_rate=sr, epoch_length_s=epoch_len, n_epochs=n_epochs,
            n_stages=n_stages, transition_matrix=tm, seed=seed,
            mar_mode=False,
            frequency_bins=freq_bins, stage_band_powers=stage_band_powers, 
            stage_bin_probs=stage_bin_probs, components_per_epoch=components,
            stage_mean_offsets=mean_offsets, stage_rms_scales=rms_scales, 
            normalize_epoch=normalize, overlap_samples=overlap_samples
        )

# ----------------------------- CLI Entry -------------------------------------

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python script.py config.yml")
        sys.exit(1)
        
    cfg_path = Path(sys.argv[1])
    if not cfg_path.exists():
        print("Config file not found")
        sys.exit(1)
        
    cfg = load_config(cfg_path)
    gen = build_generator(cfg)
    eeg, labels = gen.generate()
    gen.save(eeg, labels, cfg_path)
