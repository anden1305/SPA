# Joint holdout fold-4 cHMM-GMVAE ablations

**Added 2026-06-08.** Nine targeted deltas on the **unified paper-line** joint fold-4 config, aiming for **best-of-3 prior NMI ≥ 0.52** (baseline interactive run peaked at **0.407**; cGMVAE on the same fold reached **0.575** on seed 1).

Base: per-lab `cvae_overrides`, `wide_mlp`, `latent_dim=6`, `emb_dim=4`, `hmm_gmm`, `T=64`, `lr=1.3e-3`, `decoder_only_conditioning=true`. **Not** re-run here (already in `joint_holdout/fold_4/chmmgmvae`).

| Ablation | Change | Rationale |
|----------|--------|-----------|
| `warm_prior` | `warm_hmm_gmm` | lab_3 incohort lock (+0.04 vs simple) |
| `seq32` | `sequence_length: 32` | lab_5 incohort best T; less collapse |
| `warm_seq32` | warm + T=32 | combine lab_3 + lab_5 locks |
| `lr8e4` | `learning_rate: 8e-4` | stabler HMM / transition learning |
| `sticky92` | `hmm_sticky_kappa: 0.92` | anti single-state collapse (seed-1 hmmgmvae pattern) |
| `latent8` | `latent_dim: 8` | more capacity toward cGMVAE-like separation |
| `no_beta20` | `no_beta_epochs: 20` | longer recon-first phase |
| `emb8` | `emb_dim: 8` | richer subject conditioning cross-lab |
| `warm_latent8` | warm + `latent_dim: 8` | highest-upside combo |

Excluded (prior evidence): enc+dec conditioning (worse incohort), `T=128` (collapse), per-lab prior tiers (paper line uses one recipe).

**Generate:**

```bash
PYTHONPATH=. python3 scripts/cv4fold/generate_ablation_joint_chmm_fold4.py
```

**Submit** (~3h wall, `runs: 3` each):

```bash
bash hpc/submit/cv4fold/submit_ablation_joint_chmm_fold4.sh
```

| Queue | Jobs |
|-------|------|
| gpuv100 | warm_prior, seq32, warm_seq32, lr8e4, sticky92 |
| gpua100 | latent8 |
| gpua10 | no_beta20 |
| gpul40s | emb8 |
| gpua40 | warm_latent8 |

Logs: `hpc/output/cv4fold/ablation_joint_chmm/f4_<id>_%J.out`  
Results: `results/cv4fold/ablation_joint_chmm/fold_4/<id>/abl_joint_chmm_f4_<id>_*`

**Postprocess** (after jobs finish):

```bash
PYTHONPATH=. python3 scripts/cv4fold/postprocess_fold.py \
  --result-root results/cv4fold/ablation_joint_chmm/fold_4/<id>/<run_ts>/ \
  --per-mouse-metrics
```

Compare best-of-3 to baseline `joint_ho_f4_chmmgmvae_20260608-160204` (0.407 / 0.399 / 0.206).

## Round 2 (combo follow-ups)

Early round-1 signals: **emb8** 0.625 best-of-3; **seq32** seed1 0.649 (incomplete).

| Ablation | Change | Queue |
|----------|--------|-------|
| `emb8_seq32` | emb8 + T=32 | gpuv100 |
| `warm_emb8` | warm + emb8 | gpuv100 |
| `emb8_sticky92` | emb8 + κ=0.92 | gpuv100 |
| `warm_emb8_seq32` | warm + emb8 + T=32 | gpul40s |

```bash
PYTHONPATH=. python3 scripts/cv4fold/generate_ablation_joint_chmm_fold4.py --round 2
bash hpc/submit/cv4fold/submit_ablation_joint_chmm_fold4_round2.sh
```

## Round 3 — lock candidate

`warm_emb8_sticky92`: warm prior + emb8 + `hmm_sticky_kappa=0.92` + T=64 (fold 4 confirmatory).

```bash
PYTHONPATH=. python3 scripts/cv4fold/generate_ablation_joint_chmm_fold4.py --round 3
bash hpc/submit/cv4fold/submit_ablation_joint_chmm_fold4_round3.sh
```

## Generalize winner to folds 1–3

**Lock (pending round 3):** `warm_emb8_sticky92`. Earlier cross-fold run used plain `emb8` (0.625 fold 4; uneven folds 1–2).

```bash
PYTHONPATH=. python3 scripts/cv4fold/generate_joint_chmm_emb8_holdout.py
bash hpc/submit/cv4fold/submit_joint_chmm_emb8_holdout.sh
```

Configs: `joint_holdout/fold_{1,2,3}/chmmgmvae_emb8.yaml`  
Results: `results/cv4fold/joint_holdout/fold_{k}/chmmgmvae_emb8/`

## Fair cGMVAE comparison (fold 4)

Architecture-aligned cGMVAE: **latent 6, emb 8**, T=1, `gmm`, lr=3e-4 (vs unified emb4 baseline 0.575 on 1 seed).

```bash
PYTHONPATH=. python3 scripts/cv4fold/generate_joint_cgmvae_emb8_holdout.py --fold 4
bash hpc/submit/cv4fold/submit_joint_cgmvae_emb8_holdout.sh
```

Results: `results/cv4fold/joint_holdout/fold_4/cgmvae_emb8/`

## LR swap sensitivity (fold 4)

| Model | Recipe | LR |
|-------|--------|-----|
| cHMM | warm_emb8_sticky92, T=64 | **3e-4** (cGMVAE default) |
| cGMVAE | emb8, T=1, gmm | **1.3e-3** (cHMM default) |

```bash
PYTHONPATH=. python3 scripts/cv4fold/generate_joint_lr_swap_fold4.py
bash hpc/submit/cv4fold/submit_joint_lr_swap_fold4.sh
```
