# lab_2 REM ablations (VAE input path)

**Added 2026-06-07.** Target lab_2 in-cohort REM separability while keeping the
**conv cGMVAE as the feature learner** (thesis: subject-conditioned VAE latent
beats parallel expert handcrafted features; we do not add PAC/ratio transforms).

## Problem

Winner prepro (`no_postnorm_widebp`, NMI ~0.568) improves Awake/NREM geometry
but **REM still overlaps NREM** in latent space and in GMM predictions (~6% true
REM, heavy Awake↔REM confusion). Architecture sweeps did not help.

## Design

| Principle | Implementation |
|-----------|----------------|
| VAE stays the encoder | Same `cvae_marhmm` conv encoder + GMM prior; no legacy transform stack |
| Only input / pre-FFT knobs | Bandpass, paper robust scaling, optional RMS bin on FFT tensor |
| New diagnostics | Input-space EEG/EMG band plots on every run with `summary_statistics: true` |

### Variants (`scripts/cv4fold/generate_ablation_lab2_rem.py`)

| Variant | cvae knobs | Hypothesis |
|---------|------------|------------|
| `rem_winner` | (none — same as `no_postnorm_widebp`) | Baseline under new results dir |
| `rem_paper_robust` | `robust_normalize`, paper EEG 0.3–35 Hz | Lab-agnostic amplitude scaling |
| `rem_emg_wide` | EMG band 3–100 Hz | Preserve tonic REM muscle tone |
| `rem_emg_low` | EMG band 1–40 Hz | Emphasise low-frequency atonia envelope |
| `rem_paper_robust_emg_wide` | paper robust + wide EMG | Combined |
| `rem_append_rms` | `append_channel_rms: true` | Log-RMS per channel as extra FFT bin → still through conv VAE |
| `rem_winner_append_rms` | winner + RMS bin | Winner + atonia amplitude cue |
| `rem_paper_robust_append_rms` | paper robust + RMS bin | Paper front-end + RMS |

Config path: `src/config/run/cvaemarhmm/cv4fold/ablation_rem/lab_2/<variant>.yaml`  
Results: `results/cv4fold/ablation_rem/lab_2/abl_lab_2_<variant>_*`

## Input-space diagnostics (all future runs)

When `validator.summary_statistics: true` and data are VAE spectra `(N,S,C,F)`:

| Plot | Path | Meaning |
|------|------|---------|
| `input_channel_total_power_per_state.png` | `plots/` | Total log-power per EEG/EMG channel × state |
| `input_eeg_band_power_per_state.png` | `plots/` | Delta/theta/alpha/beta (EEG channels averaged) |
| `input_emg_band_power_per_state.png` | `plots/` | EMG low/mid/total bands — **REM atonia before encoder** |
| `input_separation_gaps.png` | `plots/` | REM−NREM and Awake−REM gaps (numeric) |

JSON: `data_validations.json` → `input_channel_statistics`.

**Interpretation:** If EMG bands do not separate REM from NREM here, fixing the
latent (arch/prior) alone is unlikely to work. If input separates but latent
does not, focus on training/capacity.

Latent plots (`feature_amplitude_per_state.png`) remain post-encoder diagnostics.

## Submit

```bash
source .venv/bin/activate
PYTHONPATH=. python3 scripts/cv4fold/generate_ablation_lab2_rem.py
bash hpc/submit/cv4fold/submit_ablation_lab2_rem.sh
```

Logs: `hpc/output/cv4fold/ablation_rem/lab_2_rem_<variant>_%J.out`

**Queues** (8 jobs): `gpuv100` ×5, `gpua10` ×1, `gpua100` ×1, `gpul40s` ×1.
`gpul40s` / `gpua100` may PEND longer than `gpuv100` — check `bqueues` before submit.
Walltime `4:00`.

## Success criteria

Primary: GMM prior NMI (same as other ablations).  
REM-specific: improved REM recall in confusion matrix; `input_emg_band_power_per_state`
shows REM EMG < NREM EMG; latent tripanel less Awake↔REM smear.
