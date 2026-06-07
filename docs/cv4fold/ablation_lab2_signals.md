# lab_2 signal montage ablation

**Added 2026-06-07.** Updated with **3-seed best-of-3** results.

Compare **EEG1+EEG3+EMG** (HQ montage) vs **EEG1+EEG4+EMG** on the **`no_postnorm_widebp`** winner (200 ep, 3 seeds, scratch).

**Selection metric:** best GMM prior NMI among 3 seeds per YAML ([README.md](README.md#selection-metric-final-cv4fold)).

| Variant | signals | Config dir |
|---------|---------|------------|
| `baseline_eeg1_eeg3` | EEG1, EEG3, EMG | `ablation_signals/lab_2/` |
| `eeg1_eeg4` | EEG1, EEG4, EMG | same |

Generator: `scripts/cv4fold/generate_ablation_lab2_signals.py`  
Results: `results/cv4fold/ablation_signals/lab_2/abl_lab_2_*`

Reference prepro winner: `no_postnorm_widebp` **best 0.568** (prepro ablation dir).

## Results (3 seeds complete — 2026-06-07)

| Variant | Best-of-3 | seeds | Mean | Notes |
|---------|-----------|-------|------|-------|
| `baseline_eeg1_eeg3` | **0.517** | [0.515, 0.370, 0.517] | 0.467 | Paired re-run in `ablation_signals/` |
| `eeg1_eeg4` | **0.551** | [0.537, 0.538, 0.551] | 0.542 | All seeds stable |
| `rem_winner` (REM dir, EEG3) | **0.575** | [0.575, 0.539, 0.522] | 0.545 | Same prepro, different results path |
| prepro `no_postnorm_widebp` | **0.568** | [0.567, 0.541, 0.568] | 0.559 | Historical prepro winner |

**Best-of-3:** EEG4 (**0.551**) beats paired EEG3 baseline (**0.517**) by **+0.034**, but **`rem_winner` (0.575)** still edges EEG4 on best seed alone. EEG4 has **better seed stability** (no 0.37 collapse).

**W&B fold-1 detail:** EEG4 still climbing at ep 200 (0.584); EEG3 peaked ep 71 then drifted — do not early-stop on prior NMI. Full curves: [ablation_lab2_findings_20260607.md](ablation_lab2_findings_20260607.md).

**Provisional lock:** adopt **EEG1+EEG4+EMG** for downstream REM ablations if **`rem_emg_wide_eeg4`** best-of-3 beats prepro winner **0.568** when complete (see REM doc).

## Submit

```bash
source .venv/bin/activate
PYTHONPATH=. python3 scripts/cv4fold/generate_ablation_lab2_signals.py
bash hpc/submit/cv4fold/submit_ablation_lab2_signals.sh
```

Logs: `hpc/output/cv4fold/ablation_signals/lab_2_<variant>_%J.out`
