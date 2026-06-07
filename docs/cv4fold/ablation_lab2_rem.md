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

Primary: **best-of-3** GMM prior NMI ([README.md](README.md#selection-metric-final-cv4fold)).  
REM-specific: improved REM recall in confusion matrix; `input_emg_band_power_per_state`
shows REM EMG < NREM EMG; latent tripanel less Awake↔REM smear.

## Results (best-of-3, 2026-06-07)

All on **`no_postnorm_widebp`** unless noted. EEG3 montage unless `*_eeg4`.

| Variant | Best | seeds | Stable? | Notes |
|---------|------|-------|---------|-------|
| `rem_winner` (EEG3) | **0.575** | [0.575, 0.539, 0.522] | yes | Baseline REM dir |
| `rem_append_rms` | **0.568** | [0.392, 0.568, 0.511] | mostly | Ties prepro best on one seed |
| `rem_winner_append_rms` | 0.560 | [0.560, 0.555, 0.537] | yes | Below winner |
| `rem_emg_wide` (EEG3) | 0.549 | [0.523, 0.549, **0.097**] | **no** | s3 collapse — discard |
| `rem_emg_low` (EEG3) | 0.523 | [0.520, 0.476, 0.523] | yes | No gain vs winner |
| `rem_paper_robust` | 0.512 | [0.181, 0.512, 0.397] | no | Paper front-end worse |
| `rem_paper_robust_append_rms` | 0.485 | [0.439, 0.485, 0.399] | yes | Worse |
| `rem_paper_robust_emg_wide` | — | — | — | Completed elsewhere; poor |
| **`rem_emg_low_eeg4`** | **0.566** | [0.566, 0.555, 0.565] | **yes** | DONE — stable, ≈ prepro |
| **`rem_emg_wide_eeg4`** | **0.593** | [0.593] (1/3) | TBD | **RUN** — leading candidate |

**Takeaway:** Wide EMG on EEG3 **collapses** one seed; on **EEG4**, `rem_emg_wide` s1 **0.593** exceeds prepro **0.568** — wait for 3 seeds before lock. `rem_emg_low_eeg4` matches prepro but does not beat **`rem_emg_wide_eeg4`** on best seed.

Training curves / queue: [ablation_lab2_findings_20260607.md](ablation_lab2_findings_20260607.md).

## Interim findings (superseded table — see Results above)

Prior one-seed notes kept for W&B curve context only; use **Results** table for decisions.
