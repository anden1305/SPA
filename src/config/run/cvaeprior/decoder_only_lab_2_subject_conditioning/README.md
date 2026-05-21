# Decoder-only cGMVAE — Lab 2 (subject conditioning)

Mirrors `decoder_only_lab_5_subject_conditioning/` for the Lab 2 quality cohort used in LOLO configs:

| Subject | Runs |
|---------|------|
| sub-071, sub-072, sub-076, sub-077, sub-080, sub-081 | 1–2 each |

## Layout

| Path | Purpose |
|------|---------|
| `subjectwise/sub{NNN}_cgmvae_decoder_only.yaml` | Single-mouse training (3 seeds) |
| `reliability/reliability_cgmvae_decoder_only_subject_conditioning.yaml` | All 6 subjects (10 seeds) |
| `generalization/generalization_*_sub{holdout}.yaml` | Leave-one-mouse-out: train on 5 mice, val on holdout (all 6 subjects) |
| `generalization_subject/generalization_subject_*_sub{holdout}.yaml` | Same 6 holdouts; train run 1, val run 2 on holdout only |

Hyperparameters match Lab 5 / Lab 3 decoder-only (`prior: gmm`, 80 epochs, same architecture).

## Submit (login node)

```bash
bash hpc/submit/decoder_only_lab_2/run_all_decoder_only_lab_2.sh
```

Or individual stages under `hpc/submit/decoder_only_lab_2/`.

Results: `results/decoder_only/{subjectwise,reliability,generalization,generalization_subject}/cgmvae_decoder_only_lab_2/`

**Tip:** Run subjectwise before generalization if you want to load per-subject checkpoints; generalization yaml paths mirror Lab 5 (`generalization/.../sub{holdout}/cvae_final_model.pth`).
