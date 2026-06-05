# cv4fold cross-validation experiment

4-fold CV on quality cohort (labs 2, 3, 5): per-lab + joint scopes, four model families.

## Models

| Key | Command | `runs` |
|-----|---------|--------|
| `cgmvae` | `python3 main.py --method train_vae -c <yaml>` | 3 (best-of-3) |
| `hmmgmvae` | same | 3 |
| `chmmgmvae` | same | 3 |
| `hmm_raw` | `python3 main.py --method train -c <yaml>` | 1 |

## Folder structure

```
results/cv4fold/
├── manifest_summary.csv
├── selection/best_runs.json
├── {cgmvae,hmmgmvae,chmmgmvae,hmm_raw}/
│   └── {joint,per_lab/lab_2,per_lab/lab_3,per_lab/lab_5}/
│       └── fold_{1..4}/
│           └── <run_name_timestamp>/
│               ├── config.json
│               ├── 1/  2/  3/          # VAE: three runs; HMM raw: 1/ only
│               │   ├── validations.json
│               │   ├── losses.json
│               │   ├── plots/
│               │   │   ├── losses.png
│               │   │   ├── hmm_tripanel_pc1_pc2.png
│               │   │   ├── confusion_matrices.png
│               │   │   ├── metrics.txt
│               │   │   └── results.npz
│               │   └── checkpoints/
│               │       ├── cvae_final_model.pth
│               │       └── cvae_best_kmeans_nmi.pth
│               └── per_mouse/
│                   ├── run_1/sub-038/
│                   │   ├── hmm_tripanel_pc1_pc2.png   # same style as run-level
│                   │   ├── predictions.npz
│                   │   └── metrics.json
│                   ├── run_2/...
│                   └── run_3/...
```

Configs: `src/config/run/cvaeprior/cv4fold/`, `src/config/run/hmm/cv4fold/`.

Manifest: `data/manifests/cv_quality_cohort_v1.yaml`.

## Validation frequency

- VAE: `validate_per_epoch: 10`
- HMM raw: `validate_per_epoch: 10`

## W&B metrics (cgmvae / chmmgmvae)

Per-epoch collapse metrics, thesis checkpoint score \(S(\varepsilon)\), and optional `cvae_best_checkpoint_score.pth` are documented in **[cvae_wandb_checkpoint_metrics.md](../cvae_wandb_checkpoint_metrics.md)**.

Post-run analysis for the subject_lab tune-winner batch: **[subject_lab_followup.md](subject_lab_followup.md)**.

Decoder-only thesis → cv4fold fold-4 A/B/C (seq1 vs seq64 vs subject_lab): **[decoder_only_roadmap.md](decoder_only_roadmap.md)**.

## Workflow

1. `python -m scripts.data_exploration.build_cv_fold_manifest`
2. `python -m scripts.cv4fold.generate_configs --phase full`
3. **Phase 0 smoke (80 ep, fold 4 sanity):** `bash hpc/submit/cv4fold/submit_smoke_80.sh`
4. **Phase 1 tune (Bayesian, all mice in-sample):**
   - `PYTHONPATH=. python3 scripts/cv4fold/generate_configs.py --phase tune_sweep_base --all-mice --models cgmvae chmmgmvae`
   - `bash hpc/submit/cv4fold/submit_tune_sweep.sh` (50 W&B trials per model; mem/queue in script)
5. **Phase 2 full CV (all folds / scopes):** lock hyperparams in manifest → `generate_configs --phase full` → `bash hpc/submit/cv4fold/submit_all_cv4fold.sh`
6. **subject_lab + tune winners (joint, 4 folds):** `generate_configs --phase subject_lab_tune_winners` → `bash hpc/submit/cv4fold/submit_subject_lab_tune_winners.sh` → `summarize_subject_lab_cv.py` after jobs finish. Per-lab val NMI: `val_nmi_by_lab.csv` (via `postprocess_fold.py`).

### HPC wall times (`gpuv100`, max **24:00**)

| Phase | Script | `-W` | Config workload |
|-------|--------|------|-----------------|
| Smoke80 VAE/HMM | `submit_smoke_80.sh` | 8:00 | 80 epochs (VAE), HMM raw smoke80 |
| Tune sweep | `submit_tune_sweep.sh` → `launch_wandb_sweep.sh` | 8:00 / 12:00 | 50 bayes trials, all mice, `runs: 1` |
| Full VAE | `run_cv4fold_vae.sh` | 16:00 | 80 epochs, `runs: 3` |
| Full HMM | `run_cv4fold_hmm_raw.sh` | 24:00 | 30k epochs (thesis); may need resume |

All jobs: `-n 4 -R rusage[mem=5GB]` (~20GB with 4 slots), `-gpu num=1:exclusive_process`.

**Queue spread** (`hpc/submit/cv4fold/cv4fold_queue.sh`, same idea as `subjectwise_raw`):

| Queue | Typical parallel | cv4fold routing |
|-------|------------------|-----------------|
| `gpuv100` | many | **all smoke** (4 jobs); default VAE/HMM full grid |
| `gpua100` | 1–2 | all `per_lab/lab_3` VAE; joint chmmgmvae fold 2/4; joint HMM fold 2/4; 2 tune jobs |
| `gpul40s` | 1 | lab_5 chmmgmvae / hmmgmvae fold 4; per-lab lab_3/5 HMM |
| `gpua40` | 1 | `per_lab/lab_5/fold_4` HMM raw only |

Submit scripts use **heredoc bsub** (`cv4fold_bsub.sh`) so `#BSUB -W`, mem, and GPU flags apply. Do not use `bsub script.sh arg` — LSF treats that as a plain command and defaults to 15 min / 1 GB.
6. `python -m scripts.cv4fold.select_best_vae_run --fold-dir results/cv4fold/.../<timestamp>`

## Smoke / sanity checklist

| Artifact | Healthy | Problem |
|----------|---------|---------|
| `losses.png` | Smooth decrease in recon phase | NaN / flat / spike at HMM warmup |
| `validations.json` | `cvae_latent_kmeans_nmi` rises | Stuck ~0.33 |
| `hmm_tripanel_pc1_pc2.png` | Stage clusters visible; titles use model name (cHMM-GMVAE, etc.) | One blob |
| `feature_amplitude_per_state.png` | Wake/NREM/REM separability when tuning `latent_dim` | Flat / overlapping bands |
| `confusion_matrices.png` | Structured off-diagonals | Single class |
| `metrics.txt` | NMI reasonable for lab | <0.25 on lab 3 smoke |
| `results.npz` | `sub_ids` matches holdout mice | Wrong shape/IDs |
| `hmmgmm_trajectory.png` | Present for HMM-GMM priors | Missing |

## Best-of-3 (VAE)

Compare `metrics.txt` NMI in runs `1/`, `2/`, `3/`; record winner in `results/cv4fold/selection/best_runs.json`.

## Ablation comparisons

- **Prior:** cgmvae vs chmmgmvae (same subject conditioning)
- **Conditioning:** hmmgmvae vs chmmgmvae (same warm_hmm_gmm prior)
- **Thesis uplift:** hmm_raw vs VAE variants on identical folds

## Your next steps (on HPC)

```bash
cd /work3/s204070/SPA
bash hpc/submit/cv4fold/submit_smoke_80.sh       # Phase 0

# after smoke OK:

PYTHONPATH=. python3 scripts/cv4fold/generate_configs.py --phase tune_sweep_base --all-mice --models cgmvae chmmgmvae
bash hpc/submit/cv4fold/submit_tune_sweep.sh     # Phase 1 (both models; NUM_AGENTS=50 default)

# After tune: pick best trial in W&B, edit data/manifests/cv_quality_cohort_v1.yaml:
#   hyperparams:
#     learning_rate: 0.0003   # winner from tune
#     epochs: 120
# Then regenerate all full YAMLs (folds/mice unchanged):
PYTHONPATH=. python3 scripts/cv4fold/generate_configs.py --phase full

bash hpc/submit/cv4fold/submit_all_cv4fold.sh   # Phase 2
```

**Lock hyperparams** = write the tune winner into the manifest so every full-grid config gets the same `trainer.learning_rate` and `trainer.epochs`. Only rerun `build_cv_fold_manifest` if you changed cohort or fold splits—not for LR/epochs alone.