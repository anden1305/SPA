# Unified holdout paper line (3 models × 2 scopes)

**Added 2026-06-08.** Primary thesis experiments: **cgmvae** vs **hmmgmvae** vs **chmmgmvae**, each with **one unified recipe**, tested in **within-lab** and **joint** train scopes. Per-lab **preprocessing** from incohort locks via `cvae_overrides`; model/training hyperparams identical across scopes per model family.

Incohort ablation NMI = **ceiling reference** (supplementary), not the train config for this line.

---

## Design

| | Within-lab | Joint |
|--|------------|-------|
| Train pool | Mice in one lab (minus fold holdout) | All labs' train mice |
| Val mice | Same manifest holdout per lab | Same |
| Recipe | `UNIFIED_*` per model | Same YAML |

**Holdout targets:** lab_2 & lab_5 ≥ **0.50**, lab_3 ≥ **0.63** prior NMI (best-of-3).

### Zero-shot inference (paper Methods)

Holdout **prior NMI** uses encoder latents + mixture prior only — **no subject embeddings** at evaluation for decoder-only models (`cgmvae_locked`, `chmmgmvae_locked`). Subject `emb` is training-only (decoder reconstruction). See [`docs/paper/holdout_experiments.md`](../paper/holdout_experiments.md) and `src/models/vae.py` (`predict_gmm_labels`).

---

## Unified recipes ([`locked_recipes.py`](../../scripts/cv4fold/locked_recipes.py))

Shared: `latent_dim=6`, wide_mlp `[512,256,128]` / `[128,256,512]`, `epochs=200`, `batch_size=128`, `runs=3`, scratch.

| Model | T | Prior | emb | lr |
|-------|---|-------|-----|-----|
| cgmvae | 1 | gmm | 4 | 3e-4 |
| hmmgmvae | 64 | hmm_gmm | 0 | 1.3e-3 |
| chmmgmvae | 64 | hmm_gmm | 4 | 1.3e-3 |

Per-mouse `cvae_overrides` from `LOCKED_SOURCES` prepro YAML per lab.

### Locked fair-comparison joint line (2026-06-08)

After fold-4 ablations, joint cv4fold for **cgmvae vs chmmgmvae** with matched capacity:

| Model | emb | T | Prior | lr |
|-------|-----|---|-------|-----|
| cgmvae_locked | 8 | 1 | gmm | 1.3e-3 |
| hmmgmvae_locked | 0 | 64 | hmm_gmm | 1.3e-3 |
| chmmgmvae_locked | 8 | 64 | warm_hmm_gmm, κ=0.92 | 1.3e-3 |

Both: `latent_dim=6`, wide_mlp, per-lab `cvae_overrides`, `runs: 3`.

```bash
PYTHONPATH=. python3 scripts/cv4fold/generate_joint_locked_holdout.py
bash hpc/submit/cv4fold/submit_joint_locked_holdout.sh   # 12 jobs: 3 models × 4 folds
```

YAML: `joint_holdout/fold_{k}/{cgmvae,chmmgmvae}_locked.yaml`  
Results: `results/cv4fold/joint_holdout/fold_{k}/{cgmvae,chmmgmvae}_locked/`

**Within-lab scope** (same locks, train/val single lab per YAML):

```bash
PYTHONPATH=. python3 scripts/cv4fold/generate_within_lab_locked_holdout.py
bash hpc/submit/cv4fold/submit_within_lab_locked_holdout.sh
```

YAML: `unified_holdout/{lab}/fold_{k}/{cgmvae,chmmgmvae}_locked.yaml` (24 configs)  
Results: `results/cv4fold/unified_holdout/{lab}/fold_{k}/{cgmvae,chmmgmvae}_locked/`

---

## Generate configs

```bash
source .venv/bin/activate
# Phase 0 pilot (joint fold 4 cHMM only):
PYTHONPATH=. python3 scripts/cv4fold/generate_configs.py \
  --phase joint_holdout --fold 4 --models chmmgmvae

# Full paper line:
PYTHONPATH=. python3 scripts/cv4fold/generate_configs.py \
  --phase within_lab_holdout --fold all
PYTHONPATH=. python3 scripts/cv4fold/generate_configs.py \
  --phase joint_holdout --fold all
```

Paths:
- `src/config/run/cvaemarhmm/cv4fold/unified_holdout/{lab}/fold_{k}/{model}.yaml`
- `src/config/run/cvaemarhmm/cv4fold/joint_holdout/fold_{k}/{model}.yaml`

---

## Submit

**Phase 0 gate** (1 job before 48-job wave):

```bash
bash hpc/submit/cv4fold/submit_joint_holdout.sh --fold 4 --models chmmgmvae
```

`hpc/output/cv4fold/joint_holdout/f4_chmmgmvae_%J.out`

Pass: job completes; ≥2/3 seeds NMI > 0.1; `val_nmi_by_lab.csv` sane after postprocess.

**Full wave** (after pilot):

```bash
bash hpc/submit/cv4fold/submit_unified_within_lab_holdout.sh --fold all
bash hpc/submit/cv4fold/submit_joint_holdout.sh --fold all \
  --skip-fold 4 --skip-model chmmgmvae
```

36 within-lab + 11 joint (skip fold-4 chmmgmvae if pilot done).

Walltime: cgmvae **4h**, HMM models **6h**, queue `gpuv100`.

---

## Results

- Within-lab: `results/cv4fold/unified_holdout/{lab}/fold_{k}/{model}/`
- Joint: `results/cv4fold/joint_holdout/fold_{k}/{model}/`

Postprocess (per run):

```bash
PYTHONPATH=. python3 scripts/cv4fold/postprocess_fold.py \
  --result-root results/cv4fold/joint_holdout/fold_4/chmmgmvae/<run_ts>/ \
  --per-mouse-metrics
```

---

## Phase 0 pilot

| Field | Value |
|-------|--------|
| Config | `joint_holdout/fold_4/chmmgmvae.yaml` |
| Status | TBD after first submit |

---

## Code pointers

- `DatasetConfig.cvae_overrides` — [`config.py`](../../src/config/config.py)
- Merge at load — [`data_loader.py`](../../src/data/data_loader.py) `process_data()`
- Generator — [`generate_configs.py`](../../scripts/cv4fold/generate_configs.py) phases `within_lab_holdout`, `joint_holdout`
- Tests — [`tests/test_per_lab_cvae_overrides.py`](../../tests/test_per_lab_cvae_overrides.py)
