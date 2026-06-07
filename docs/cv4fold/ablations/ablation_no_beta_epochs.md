# Per-lab `no_beta_epochs` ablation

**Added 2026-06-07.** KL-onset sweep on each lab's **locked best-NMI** incohort recipe.

## Question

On each lab's winner config, does delaying KL (`no_beta_epochs: 10`) beat `no_beta_epochs: 0` (beta from epoch 1)?

## Locked bases (control = `no_beta_epochs: 10`)

| Lab | Base YAML | Control best NMI | Notes |
|-----|-----------|------------------|-------|
| **lab_2** | `ablation_rem/lab_2/rem_emg_wide_eeg4.yaml` | **0.593** | EEG1+EEG4+EMG, EMG 3–100 Hz, no postnorm (28607461) |
| **lab_3** | `ablation_arch/lab_3/wide_mlp.yaml` | **0.737** | postnorm, EEG 0–20 Hz, wide MLP |
| **lab_5** | `ablation_arch/lab_5/wide_mlp.yaml` | **0.534** | postnorm, EEG 0–20 Hz, 200 ep, wide MLP |

Everything else unchanged (200 ep, 3 seeds, scratch, GMM prior).

**Selection:** best-of-3 prior NMI ([README.md](README.md#selection-metric-final-cv4fold)).

## Generate & submit

Default generates **`no_beta_epochs: 0` only** — use existing winner runs as control **10**.

```bash
source .venv/bin/activate
PYTHONPATH=. python3 scripts/cv4fold/generate_ablation_no_beta_epochs.py
bash hpc/submit/cv4fold/submit_ablation_no_beta_epochs.sh
```

Single lab:

```bash
PYTHONPATH=. python3 scripts/cv4fold/generate_ablation_no_beta_epochs.py --labs lab_3
```

Paired re-run in `ablation_beta/` (optional):

```bash
PYTHONPATH=. python3 scripts/cv4fold/generate_ablation_no_beta_epochs.py --values 0 10
bash hpc/submit/cv4fold/submit_ablation_no_beta_epochs.sh
```

Logs: `hpc/output/cv4fold/ablation_beta/<lab>_no_beta_epochs_*_%J.out`  
Results: `results/cv4fold/ablation_beta/<lab>/abl_<lab>_no_beta_epochs_*`

## Decision rule (per lab)

| Outcome | Action |
|---------|--------|
| **0** best-of-3 **>** control by ≥ ~0.01, stable seeds | Set `no_beta_epochs: 0` in that lab's locked template |
| **10** wins or tie | Keep default **10** |
| **0** collapses (≥2 bad seeds) | Keep **10** regardless of one lucky seed |

Compare per lab — do not assume one result transfers across labs.

Related: [ablation_lab2_findings_20260607.md](ablation_lab2_findings_20260607.md), [ablation_prepro_lab2_lab5.md](ablation_prepro_lab2_lab5.md).
