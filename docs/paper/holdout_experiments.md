# Paper holdout experiments (3-model ladder)

**Added 2026-06-09.** P0 experiments for main Fig 2: joint + within-lab locked holdout across **cGMVAE**, **HMMGMVAE**, **cHMMGMVAE**.

Canonical design: [unified_holdout_paper_line.md](../cv4fold/unified_holdout_paper_line.md).

---

## Zero-shot inference protocol (holdout evaluation)

**Paper claim:** population-trained on training folds; evaluated on held-out mice with **no per-subject calibration** (contrast AccuSleep mixture z-scoring).

| Phase | Subject embeddings | Path |
|-------|-------------------|------|
| **Training** | Yes, decoder-only (`emb=8`, `decoder_only_conditioning: true`) | signal → encoder → latent → decoder(+subject emb) for reconstruction; prior loss on latent |
| **Holdout staging** | **No** | signal → encoder → latent μ → GMM or sticky HMM–GMM prior → Wake/NREM/REM (prior-prediction NMI) |

Code: [`src/models/vae.py`](../../src/models/vae.py) — `use_encoder_conditioning` is false when `decoder_only_conditioning` is true; [`predict_gmm_labels`](../../src/models/vae.py) never calls the decoder. Validator passes `sub_ids` but they are ignored at encode for decoder-only models.

**Honest limit:** per-lab spectral front-end locks (`cvae_overrides`) are fixed from incohort tuning — not per-mouse manual scoring.

| Model | emb at train | Subject ID at staging |
|-------|--------------|----------------------|
| `hmmgmvae_locked` | 0 | N/A |
| `cgmvae_locked` / `chmmgmvae_locked` | 8 (decoder only) | **Not used** |

---

## Locked recipes (fair comparison)

| Model | emb | T | Prior | lr |
|-------|-----|---|-------|-----|
| `cgmvae_locked` | 8 | 1 | gmm | 1.3e-3 |
| `hmmgmvae_locked` | 0 | 64 | hmm_gmm | 1.3e-3 |
| `chmmgmvae_locked` | 8 | 64 | warm_hmm_gmm, κ=0.92 | 1.3e-3 |

Thesis HMM/MAR-HMM feature baseline (~0.51 population NMI) — supplement / bottom ladder rung; no retrain unless refreshing one fold.

---

## Generate configs

```bash
source .venv/bin/activate
PYTHONPATH=. python3 scripts/cv4fold/generate_joint_locked_holdout.py
PYTHONPATH=. python3 scripts/cv4fold/generate_within_lab_locked_holdout.py
```

YAML paths:
- Joint: `src/config/run/cvaemarhmm/cv4fold/joint_holdout/fold_{k}/{cgmvae,hmmgmvae,chmmgmvae}_locked.yaml`
- Within-lab: `src/config/run/cvaemarhmm/cv4fold/unified_holdout/{lab}/fold_{k}/...`

---

## Submit (user confirms before `bsub`)

| Script | Jobs | Queue mix | Wall |
|--------|------|-----------|------|
| `hpc/submit/cv4fold/submit_joint_locked_holdout.sh` | 12 (3×4 folds) | `_queue_mix.sh` | cGMVAE 1:30; HMM 2:00 |
| `hpc/submit/cv4fold/submit_within_lab_locked_holdout.sh` | 36 (3×3 labs×4) | same | same |

```bash
bash hpc/submit/cv4fold/submit_joint_locked_holdout.sh
```

`hpc/output/cv4fold/joint_holdout/f<FOLD>_<MODEL>_locked_%J.out`

```bash
bash hpc/submit/cv4fold/submit_within_lab_locked_holdout.sh
```

`hpc/output/cv4fold/unified_holdout/f<FOLD>_<LAB>_<MODEL>_locked_%J.out`

---

## Postprocess + aggregate

Per finished run:

```bash
PYTHONPATH=. python3 scripts/cv4fold/postprocess_fold.py \
  --result-root results/cv4fold/joint_holdout/fold_4/chmmgmvae_locked/<run_ts>/ \
  --per-mouse-metrics
```

Paper ladder table (all scopes):

```bash
PYTHONPATH=. python3 scripts/paper/summarize_holdout_ladder.py \
  --out paper/overleaf/tables/holdout_ladder.csv
PYTHONPATH=. python3 scripts/paper/plot_ladder_figure.py
```

---

## Results roots

| Scope | Path |
|-------|------|
| Joint | `results/cv4fold/joint_holdout/fold_{k}/{model}_locked/` |
| Within-lab | `results/cv4fold/unified_holdout/{lab}/fold_{k}/{model}_locked/` |

**Do not** compare absolute NMI between within-lab vs joint — different train pools.

**Partial status (2026-06-09):** joint cHMM 3/4 folds; joint cGMVAE 4/4; hmmgmvae_locked and some chmmgmvae folds still pending.
