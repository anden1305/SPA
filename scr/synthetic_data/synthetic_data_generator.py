"""Simple synthetic EEG generator with hidden stage sequence.

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
  frequency_bins: list[[low, high]]     # Frequency bins (Hz)
  stage_band_powers: list[list[float]]  # shape (n_stages, n_bins) amplitude prototypes
    stage_bin_selection_probability: list[list[float]] # shape (n_stages, n_bins) sampling probs per bin (rows sum to 1)
    components_per_epoch: int      # Number of sinusoidal components per epoch
    stage_mean_amplitudes: list[float] (optional) # Per-stage mean (DC offset) added after synthesis & normalization
    stage_rms_scales: list[float] (optional) # Per-stage scale applied after optional normalization
    amplitude_jitter_range: [low, high] (optional, default [0.8,1.2])
    normalize_epoch: bool (optional, default True) # If True, normalize each epoch to unit std before scaling
  seed: int (optional)           # Random seed

Generation flow per epoch:
 1. Sample stage (uniform for first epoch, then Markov using transition_matrix)
 2. Sample bins for each component using stage-specific selection probability
 3. For each component sample a frequency uniformly in its bin, amplitude derived from stage_band_powers
 4. Sum sinusoids to form epoch signal (phase continuity enforced across epochs)

Optional continuity smoothing:
    - To avoid visible value jumps between epochs without reintroducing DC drift,
        we support a short linear fade ("de-click") at epoch boundaries controlled by overlap_s.
        This linearly bridges the first samples of the new epoch to match the last value
        of the previous epoch while preserving overall length.

Note: Value-continuity shifting between epochs has been disabled to avoid baseline drift / DC offsets.

Continuity:
  - We keep cumulative phase for each component index across epochs if the bin selected is the same.
  - Regardless of component changes, we shift (add constant offset to) the next epoch so its first sample equals the last sample of the previous epoch (value continuity).

Simplicity decisions:
  - Single EEG channel only.
  - No extra noise / artifacts (extend later if needed).
  - Stage names are integers 0..n_stages-1; user can map later.

Run as script:
    python -m scr.synthetic_data.synthetic_data_generator path/to/config.yml
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
class SimpleSyntheticEEGGenerator:
    name: str
    sampling_rate: int
    epoch_length_s: int
    n_epochs: int
    n_stages: int
    transition_matrix: np.ndarray  # (S,S)
    frequency_bins: List[tuple]
    stage_band_powers: np.ndarray  # (S,B)
    stage_bin_probs: np.ndarray    # (S,B)
    components_per_epoch: int
    seed: int | None = None
    jitter_low: float = 0.8
    jitter_high: float = 1.2
    stage_mean_offsets: np.ndarray | None = None  # (S,)
    stage_rms_scales: np.ndarray | None = None    # (S,) per-stage post-normalization scale
    normalize_epoch: bool = True                  # whether to normalize each epoch to unit std
    overlap_samples: int = 0  # linear boundary smoothing window (samples); 0 disables

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        # Validations
        if self.transition_matrix.shape != (self.n_stages, self.n_stages):
            raise ValueError("transition_matrix shape mismatch")
        if not np.allclose(self.transition_matrix.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("Each row of transition_matrix must sum to 1")
        n_bins = len(self.frequency_bins)
        if self.stage_band_powers.shape != (self.n_stages, n_bins):
            raise ValueError("stage_band_powers shape mismatch")
        if self.stage_bin_probs.shape != (self.n_stages, n_bins):
            raise ValueError("stage_bin_selection_probability shape mismatch")
        if not np.allclose(self.stage_bin_probs.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("Each row of stage_bin_selection_probability must sum to 1")
        if self.components_per_epoch <= 0:
            raise ValueError("components_per_epoch must be > 0")
        if self.jitter_low <= 0 or self.jitter_high <= 0 or self.jitter_low > self.jitter_high:
            raise ValueError("Invalid amplitude jitter range")
        self.samples_per_epoch = self.sampling_rate * self.epoch_length_s
        self.total_samples = self.samples_per_epoch * self.n_epochs
        if self.stage_mean_offsets is None:
            self.stage_mean_offsets = np.zeros(self.n_stages, dtype=float)
        else:
            if len(self.stage_mean_offsets) != self.n_stages:
                raise ValueError("stage_mean_amplitudes length must equal n_stages")
        if self.stage_rms_scales is not None:
            if len(self.stage_rms_scales) != self.n_stages:
                raise ValueError("stage_rms_scales length must equal n_stages")

        # For phase continuity we keep per-component phase and frequency arrays
        self._component_phases = np.zeros(self.components_per_epoch)
        self._component_freqs = np.zeros(self.components_per_epoch)
        self._initialized = False

    # ------------------------- Public API -------------------------
    def generate(self):
        labels = self._sample_stage_sequence()
        eeg = np.zeros(self.total_samples, dtype=float)
        cursor = 0
        # Value-continuity hard shift is disabled to avoid DC drift.
        # Instead, apply a small linear smoothing at boundaries if configured.
        prev_last_val = None
        win_n = int(self.overlap_samples) if self.overlap_samples and self.overlap_samples > 0 else 0
        for epoch_idx, stage in enumerate(labels):
            epoch_sig = self._synthesize_epoch(stage)
            if prev_last_val is not None and win_n > 0:
                # Align first sample towards previous last sample and fade back to original over win_n samples
                # Compute delta after normalization and mean-offset application (already inside _synthesize_epoch)
                delta = float(epoch_sig[0] - prev_last_val)
                # Create linear ramp from delta to 0 over win_n samples (endpoint=False keeps smooth transition to next sample)
                ramp = np.linspace(delta, 0.0, win_n, endpoint=False, dtype=float)
                n_avail = min(win_n, epoch_sig.shape[0])
                epoch_sig[:n_avail] = epoch_sig[:n_avail] - ramp[:n_avail]
            # Write epoch and update trackers
            eeg[cursor:cursor + self.samples_per_epoch] = epoch_sig
            prev_last_val = float(epoch_sig[-1])
            cursor += self.samples_per_epoch
        return eeg, labels

    def save(self, eeg: np.ndarray, labels: np.ndarray, config_path: Path):
        """Save outputs under data/synthetic_data/<name>.

        If legacy results/synthetic_data/<name> exists (from earlier version),
        it is moved to the new location before saving (unless the new one already exists).
        """
        new_base = Path("data") / "synthetic_data"
        out_dir = new_base / self.name
        old_dir = Path("results") / "synthetic_data" / self.name
        # Migrate legacy directory if present
        if old_dir.exists() and not out_dir.exists():
            out_dir.parent.mkdir(parents=True, exist_ok=True)
            try:
                old_dir.rename(out_dir)
            except Exception:
                # Fallback: copy contents if rename fails (e.g., cross-device)
                import shutil
                out_dir.mkdir(parents=True, exist_ok=True)
                for p in old_dir.iterdir():
                    dest = out_dir / p.name
                    if p.is_file():
                        shutil.copy2(p, dest)
                # Do not remove old_dir to avoid accidental data loss
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "eeg.npy", eeg)
        np.save(out_dir / "labels.npy", labels)
        # Copy config
        target_cfg = out_dir / "config_copy.yml"
        if config_path.exists():
            target_cfg.write_text(config_path.read_text())
        print(f"Saved synthetic data to {out_dir}")

    # ------------------------ Internals --------------------------
    def _sample_stage_sequence(self) -> np.ndarray:
        labels = np.zeros(self.n_epochs, dtype=int)
        # First stage uniform
        labels[0] = self.rng.integers(0, self.n_stages)
        for i in range(1, self.n_epochs):
            labels[i] = self.rng.choice(self.n_stages, p=self.transition_matrix[labels[i-1]])
        return labels

    def _synthesize_epoch(self, stage: int) -> np.ndarray:
        n = self.samples_per_epoch
        t = np.arange(n) / self.sampling_rate
        epoch_sig = np.zeros(n, dtype=float)
        # Frequency bin probabilities & band powers for stage
        bin_probs = self.stage_bin_probs[stage]
        band_powers = self.stage_band_powers[stage]
        # Choose bins for each component
        chosen_bins = self.rng.choice(len(self.frequency_bins), size=self.components_per_epoch, p=bin_probs)
        # Initialize persistent arrays first time
        if not self._initialized:
            # Assign initial frequencies for components (could be random placeholder)
            for c in range(self.components_per_epoch):
                b = chosen_bins[c]
                f_low, f_high = self.frequency_bins[b]
                self._component_freqs[c] = self.rng.uniform(f_low, f_high)
                self._component_phases[c] = self.rng.uniform(0, 2*np.pi)
            self._initialized = True
        # Update component frequencies to newly sampled bins but keep phases continuous
        for c in range(self.components_per_epoch):
            b = chosen_bins[c]
            f_low, f_high = self.frequency_bins[b]
            # Keep same frequency with some probability to add temporal cohesion
            if self.rng.random() < 0.3:
                # Small random walk around previous frequency (stay inside bin)
                new_f = np.clip(self._component_freqs[c] + self.rng.normal(0, (f_high - f_low)/10), f_low, f_high)
            else:
                new_f = self.rng.uniform(f_low, f_high)
            self._component_freqs[c] = new_f
            # amplitude proportional to band power of bin divided over expected components
            amp = band_powers[b]
            # mild jitter
            amp *= self.rng.uniform(self.jitter_low, self.jitter_high)
            # component signal; phase evolves from stored phase
            phase0 = self._component_phases[c]
            component = amp * np.sin(2 * np.pi * new_f * t + phase0)
            epoch_sig += component
            # Update stored phase to end phase modulo 2pi
            end_phase = (phase0 + 2 * np.pi * new_f * (n / self.sampling_rate)) % (2 * np.pi)
            self._component_phases[c] = end_phase
        # Normalize epoch roughly (avoid huge amplitudes)
        std = np.std(epoch_sig)
        if self.normalize_epoch and std > 0:
            epoch_sig = epoch_sig / std
        # Apply per-stage scaling (post-normalization)
        if self.stage_rms_scales is not None:
            epoch_sig = epoch_sig * float(self.stage_rms_scales[stage])
        # Add per-stage mean (DC offset) if specified
        epoch_sig = epoch_sig + float(self.stage_mean_offsets[stage])
        return epoch_sig

# --------------------------- YAML Loader -------------------------------------

def load_config(path: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    required = [
        'name','sampling_rate_hz','epoch_length_s','n_epochs','n_stages',
        'transition_matrix','frequency_bins','stage_band_powers',
        'stage_bin_selection_probability','components_per_epoch'
    ]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")
    return data


def build_generator(cfg: dict) -> SimpleSyntheticEEGGenerator:
    name = cfg['name']
    sr = int(cfg['sampling_rate_hz'])
    epoch_len = int(cfg['epoch_length_s'])
    n_epochs = int(cfg['n_epochs'])
    n_stages = int(cfg['n_stages'])
    tm = np.asarray(cfg['transition_matrix'], dtype=float)
    freq_bins = [tuple(b) for b in cfg['frequency_bins']]
    stage_band_powers = np.asarray(cfg['stage_band_powers'], dtype=float)
    stage_bin_probs = np.asarray(cfg['stage_bin_selection_probability'], dtype=float)
    components = int(cfg['components_per_epoch'])
    seed = int(cfg.get('seed')) if 'seed' in cfg and cfg['seed'] is not None else None
    jl, jh = 0.8, 1.2
    if 'amplitude_jitter_range' in cfg:
        rng_vals = cfg['amplitude_jitter_range']
        if isinstance(rng_vals, (list, tuple)) and len(rng_vals) == 2:
            jl, jh = float(rng_vals[0]), float(rng_vals[1])
    mean_offsets = None
    if 'stage_mean_amplitudes' in cfg:
        mean_offsets = np.asarray(cfg['stage_mean_amplitudes'], dtype=float)
    # Optional per-stage post-normalization scales
    rms_scales = None
    if 'stage_rms_scales' in cfg and cfg['stage_rms_scales'] is not None:
        rms_scales = np.asarray(cfg['stage_rms_scales'], dtype=float)
        if rms_scales.shape[0] != n_stages:
            raise ValueError("stage_rms_scales length must equal n_stages")
        if np.any(rms_scales < 0):
            raise ValueError("stage_rms_scales must be non-negative")
    # Normalization flag
    normalize_epoch = bool(cfg.get('normalize_epoch', True))
    # Optional linear fade window in seconds
    overlap_samples = 0
    if 'overlap_s' in cfg:
        try:
            overlap_s = float(cfg['overlap_s'])
            if overlap_s > 0:
                overlap_samples = int(round(overlap_s * sr))
                # Cap to at most half an epoch to avoid excessive warping
                overlap_samples = max(0, min(overlap_samples, (sr * epoch_len) // 2))
        except Exception:
            overlap_samples = 0
    return SimpleSyntheticEEGGenerator(
        name=name,
        sampling_rate=sr,
        epoch_length_s=epoch_len,
        n_epochs=n_epochs,
        n_stages=n_stages,
        transition_matrix=tm,
        frequency_bins=freq_bins,
        stage_band_powers=stage_band_powers,
        stage_bin_probs=stage_bin_probs,
        components_per_epoch=components,
        seed=seed,
        jitter_low=jl,
        jitter_high=jh,
    stage_mean_offsets=mean_offsets,
    stage_rms_scales=rms_scales,
    normalize_epoch=normalize_epoch,
        overlap_samples=overlap_samples,
    )

# ----------------------------- CLI Entry -------------------------------------

def main(argv: Sequence[str] | None = None):
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) != 1:
        print("Usage: python -m scr.synthetic.synthetic_data_generator <config.yml>")
        return 1
    cfg_path = Path(argv[0])
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}")
        return 1
    cfg = load_config(cfg_path)
    gen = build_generator(cfg)
    eeg, labels = gen.generate()
    gen.save(eeg, labels, cfg_path)
    print(f"Generated EEG shape: {eeg.shape}, labels shape: {labels.shape}")
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
