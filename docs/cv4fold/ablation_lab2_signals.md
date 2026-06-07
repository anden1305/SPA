# lab_2 signal montage ablation

**Added 2026-06-07.** Compare **EEG1+EEG3+EMG** (current HQ montage, P+F) vs
**EEG1+EEG4+EMG** (P + second frontal) on the **`no_postnorm_widebp`** winner
(same prepro, arch, 200 ep, 3 seeds). Motivation: sub-072 has bad EEG2 only;
EEG4 is slightly lower low-freq dominance than EEG3 on that mouse (0.13 vs 0.19).

| Variant | signals | Baseline |
|---------|---------|----------|
| `baseline_eeg1_eeg3` | EEG1, EEG3, EMG | Re-run for apples-to-apples in `ablation_signals/` |
| `eeg1_eeg4` | EEG1, EEG4, EMG | Test second frontal |

Generator: `scripts/cv4fold/generate_ablation_lab2_signals.py`  
Configs: `src/config/run/cvaemarhmm/cv4fold/ablation_signals/lab_2/`  
Results: `results/cv4fold/ablation_signals/lab_2/abl_lab_2_*`

Compare GMM prior NMI to historical winner
`results/cv4fold/ablation_prepro/lab_2/abl_lab_2_no_postnorm_widebp_*` (~0.568).

## Submit

```bash
source .venv/bin/activate
PYTHONPATH=. python3 scripts/cv4fold/generate_ablation_lab2_signals.py
bash hpc/submit/cv4fold/submit_ablation_lab2_signals.sh
```

Logs: `hpc/output/cv4fold/ablation_signals/lab_2_<variant>_%J.out`
