# lab_2 locked montage & recipe

**Locked 2026-06-07** after `rem_emg_wide_eeg4` (job 28607461).

## Winner

| Field | Value |
|-------|--------|
| **Montage** | **EEG1, EEG4, EMG** (replaces EEG1, EEG3, EMG) |
| **EMG band** | 3–100 Hz (was 5–60 Hz) |
| **EEG band** | 0–30 Hz, `post_normalize: false` |
| **Best-of-3 NMI** | **0.593** [0.593, 0.578, 0.541] |
| **Canonical YAML** | [`rem_emg_wide_eeg4.yaml`](../../src/config/run/cvaemarhmm/cv4fold/ablation_rem/lab_2/rem_emg_wide_eeg4.yaml) |

## Where it is applied

| Location | Path |
|----------|------|
| HQ manifest | [`data/manifests/cv_quality_cohort_v1.yaml`](../../data/manifests/cv_quality_cohort_v1.yaml) → `lab_signals.lab_2` |
| Locked recipe loader | [`scripts/cv4fold/locked_recipes.py`](../../scripts/cv4fold/locked_recipes.py) |
| Per-lab holdout / incohort | `generate_configs.py` (manifest `signals` on dataset entries) |
| New REM ablations | `generate_ablation_lab2_rem.py` (default `--montage eeg4`) |

Historical ablation YAMLs under `ablation_prepro/`, `ablation_rem/` (EEG3 variants), and `ablation_signals/` are **unchanged archives** — do not use for new runs.

## Regenerate downstream configs

```bash
source .venv/bin/activate
PYTHONPATH=. python3 scripts/cv4fold/generate_configs.py --phase per_lab_incohort --labs lab_2 lab_3 lab_5
PYTHONPATH=. python3 scripts/cv4fold/generate_configs.py --phase per_lab_holdout --fold 4 --labs lab_2 lab_3 lab_5 --models cgmvae chmmgmvae
```
