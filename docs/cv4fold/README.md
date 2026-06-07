# cv4fold experiments

Cross-lab validation for cGMVAE / cHMM-GMVAE (Phase 2 of the [decoder conditioning roadmap](../decoder_conditioning_roadmap.md)).

**HQ cohort only:** 20 mice in [`data/manifests/cv_quality_cohort_v1.yaml`](../../data/manifests/cv_quality_cohort_v1.yaml) (labs 2, 3, 5). All configs are generated from this manifest.

| Doc | Contents |
|-----|----------|
| [cross_lab_cv4fold.md](cross_lab_cv4fold.md) | Story, folds, conditioning ablations, fold-4 results, next steps |
| [ablation_prepro_lab2_lab5.md](ablation_prepro_lab2_lab5.md) | Preprocessing ablations (rounds 1–3) |
| [ablation_lab2_rem.md](ablation_lab2_rem.md) | lab_2 REM input ablations + input-space EEG/EMG diagnostics |
| [ablation_lab2_signals.md](ablation_lab2_signals.md) | lab_2 EEG1+EEG4 vs EEG1+EEG3 montage (winner prepro) |
| [latent_separability_guide.md](latent_separability_guide.md) | Which plots/metrics matter; input vs latent; what to skip |
| [overnight_experiments_20260606.md](overnight_experiments_20260606.md) | **Overnight plan:** arch sweep + job tracker |
| [local_profiling.md](../local_profiling.md) | Laptop GPU profiling (`--profile` + smoke configs) |
| Phase 1 (lab_3) | [decoder_only_lab3_chmm_experiments.md](../decoder_only_lab3_chmm_experiments.md) |

## Per-lab in-cohort (baby step)

```bash
cd /work3/s204070/SPA && source .venv/bin/activate
PYTHONPATH=. python3 scripts/cv4fold/generate_configs.py --phase per_lab_incohort --labs lab_5 lab_2
bash hpc/submit/cv4fold/submit_per_lab_incohort.sh
```

LSF logs: `hpc/output/cv4fold/per_lab_incohort/{lab_5,lab_2}_<JOBID>.{out,err}`  
Results: `results/cv4fold/per_lab_incohort/<lab>/`

**Results:** `results/cv4fold/`  
**Fold-4 pilot table:** `results/cv4fold/decoder_only_fold4/comparison.md`
