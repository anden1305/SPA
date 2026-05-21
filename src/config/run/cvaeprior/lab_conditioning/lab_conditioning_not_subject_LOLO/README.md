# Lab conditioning, leave-one-lab-out (LOLO)

Trains on two labs, validates on the held-out lab. **`conditioning_source: lab`**.

Cohort IDs match successful **decoder-only subject conditioning** runs per lab
(see `results/decoder_only/reliability/` and `decoder_only_lab_{2,5}_subject_conditioning/`).

| Lab | Subjects (`quality_filter`) | Channels (`signals`) |
|-----|-----------------------------|----------------------|
| lab_2 | sub-071, 072, 076, 077, 080, 081 | EEG1, EEG3, EMG |
| lab_3 | sub-038, 039, 041, 043, 048, 054, 056, 059, 060, 069 | EEG1, EEG2, EMG |
| lab_5 | sub-087, 088, 089, 092 | EEG1, EEG2, EMG |

## Submit

```bash
bash hpc/submit/lab_conditioning/run_lab_conditioning_lab_only_LOLO.sh
```

Uses `train_vae` then `validate_cvae_gmm` (not `--method train`).
