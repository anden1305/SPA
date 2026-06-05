# CVAE / cHMM-GMVAE — W&B metrics and thesis checkpoint score

**Added:** 2026-06-03  
**Scope:** `train_vae` for `cgmvae` and `chmmgmvae` (cv4fold and other YAML configs using the CVAE training pipeline).

This document describes per-epoch Weights & Biases logging and the optional **collapse-aware checkpoint** aligned with the thesis composite score \(S(\varepsilon)\). Implementation: [`src/validation/validator.py`](../src/validation/validator.py), [`src/validation/hmmgmm_metrics.py`](../src/validation/hmmgmm_metrics.py), [`src/training/trainer.py`](../src/training/trainer.py), [`src/orchestrator/orchestrator.py`](../src/orchestrator/orchestrator.py).

Related: [cv4fold README](cv4fold/README.md), [cgmvae_training.md](cgmvae_training.md), [tune sweep results](cv4fold/tune_sweep_bayes50_results.md).

---

## 1. Motivation

During training we need to:

1. **Monitor collapse** — prior predictions concentrating on one sleep state while loss still improves.
2. **Track the CV metric** — prior-prediction NMI (GMM marginal or HMM Viterbi), not only latent KMeans NMI.
3. **Select checkpoints** — thesis criterion balances validation log-likelihood and diverse state usage.

MARHMM training already logs `perplexity` and `nlpp` in `validate_epoch`. CVAE training used only `cvae_latent_kmeans_nmi` per epoch; post-train `metrics.txt` NMI was computed once at the end.

---

## 2. Thesis checkpoint score (Eq. 3.49–3.50)

Let \(\tilde{\omega}(\varepsilon)\) be validation **log-likelihood** (same sign convention as training: \(-\mathrm{NLL}\) on the val set).

Let \(H(\hat{Y})\) be Shannon entropy of the empirical predicted state distribution over \(K_{\mathrm{pred}}\) states. **Normalized entropy:**

\[
H_{\mathrm{norm}}(\hat{Y}) = \frac{H(\hat{Y})}{\log K_{\mathrm{pred}}} \in [0, 1]
\]

- \(0\) = all mass on one state (collapse)  
- \(1\) = uniform usage over \(K_{\mathrm{pred}}\) states  

**Composite score:**

\[
S(\varepsilon) = \beta \, \tilde{\omega}(\varepsilon) + (1 - \beta) \, H_{\mathrm{norm}}(\hat{Y})
\]

Default \(\beta = 0.9\). The epoch maximizing \(S\) after an initial warmup fraction (default 5% of training) may be saved as `cvae_best_checkpoint_score.pth`.

**Note:** This is **not** the older MARHMM `nlpp` metric (`0.9 * log(likelihood) + 0.1 * log(perplexity/K)`), which mixes log-likelihood with log effective state count in a different way.

---

## 3. W&B metrics (per validation epoch)

Logged when `wandb.enabled` is true (default for cv4fold YAMLs without an explicit `wandb: enabled: false`). Keys appear under the `val/` namespace via [`ExperimentLogger`](../src/training/experiment_logger.py).

| W&B key | Meaning |
|---------|---------|
| `val/prior_pred_nmi` | NMI of prior labels vs scored stages (GMM marginal or HMM Viterbi) |
| `val/cvae_latent_kmeans_nmi` | KMeans on latent means vs stages (unsupervised probe; can diverge from prior NMI) |
| `val/log_likelihood` | Validation log-likelihood \(\tilde{\omega}\) |
| `val/entropy_norm` | \(H_{\mathrm{norm}}(\hat{Y})\) from prior predictions |
| `val/checkpoint_score` | \(S(\varepsilon)\) |
| `val/n_unique_pred_states` | Number of unique prior-predicted states (expect 3 for wake/NREM/REM) |
| `val/prior_switch_rate_per100` | State transitions per 100 steps on prior label sequence |
| `val/latent_autocorr_lag1` | (unchanged) Latent temporal autocorrelation when `sequence_length > 1` |
| `train/total_loss`, `train/reg_loss`, `train/data_loss` | Optimization health (β / HMM warmup) |

**Not logged per epoch:** silhouette, state distinctness (still computed post-train in `validate_cvae()` when enabled).

### Run summary (`wandb.summary`)

| Key | When set |
|-----|----------|
| `val/best_checkpoint_score` | Best \(S\) during training (if any validation epoch ran) |
| `val/best_checkpoint_score_epoch` | 1-based epoch index |
| `val/best_kmeans_nmi` | Best latent KMeans NMI |
| `val/best_kmeans_nmi_epoch` | 1-based epoch index |
| `val/prior_pred_nmi` | After post-train eval on the selected checkpoint |

---

## 4. Checkpoint selection

### Default (`checkpoint_score.enabled: false`)

1. After training, load **`cvae_best_kmeans_nmi.pth`** if it exists (best `val/cvae_latent_kmeans_nmi` during training).
2. Else **`cvae_final_model.pth`**.
3. Run `predict_cvae()` → `plots/metrics.txt` NMI.

`validation_checkpoint.txt` in the run’s `checkpoints/` folder records which file was used.

### Thesis-aligned (`checkpoint_score.enabled: true`)

```yaml
trainer:
  checkpoint_score:
    enabled: true
    beta: 0.9
    warmup_frac: 0.05
```

1. During training (after warmup), save **`cvae_best_checkpoint_score.pth`** when `val/checkpoint_score` improves.
2. Post-train priority: **score** → **KMeans** → **final**.
3. Same `predict_cvae()` and `metrics.txt` path.

KMeans-best checkpoint is **still saved** for comparison on W&B.

---

## 5. Code map

| Component | Path |
|-----------|------|
| \(H_{\mathrm{norm}}\), \(S(\varepsilon)\) | `src/validation/hmmgmm_metrics.py` — `entropy_norm`, `checkpoint_score` |
| Per-epoch CVAE validation | `Validator.validate_cvae_epoch` → `_validate_cvae_prior_metrics` |
| KMeans checkpoint | `Trainer._maybe_save_best_kmeans_checkpoint` |
| Score checkpoint | `Trainer._maybe_save_best_checkpoint_score` |
| W&B defer finish until post-train | `Trainer.train` (CVAE pipeline); `Trainer.finalize_wandb` from orchestrator |
| Post-train checkpoint choice | `Orchestrator.train_cvae` |
| Config | `CheckpointScoreConfig` in `src/config/config.py` under `trainer.checkpoint_score` |

Unit tests: [`tests/test_hmmgmm_checkpoint_score.py`](../tests/test_hmmgmm_checkpoint_score.py).

---

## 6. Interpreting curves

| Pattern | Likely meaning |
|---------|----------------|
| `prior_pred_nmi` ↑, `entropy_norm` stable ~0.8–1 | Healthy learning |
| `log_likelihood` ↑, `entropy_norm` → 0, `n_unique_pred_states` → 1 | Collapse — do not trust peak KMeans NMI alone |
| `cvae_latent_kmeans_nmi` high, `prior_pred_nmi` low | Latent separability without good prior / scoring alignment |
| `prior_switch_rate_per100` → 0 (chmm) | Sticky or collapsed HMM path |
| `checkpoint_score` peaks before final epoch | Use `enabled: true` or inspect KMeans-best epoch |

Rank experiments and sweeps by **`metrics.txt` / `val/prior_pred_nmi`**, not latent KMeans alone ([tune sweep caveat](cv4fold/tune_sweep_bayes50_results.md)).

---

## 7. Cost

Each validation epoch adds one full val `forward` and one prior predict (GMM or HMM). With `validate_per_epoch: 5` and ~250 epochs, this is modest relative to training.

---

## 8. Enabling on new jobs

Add to the training YAML (or template) before submit:

```yaml
trainer:
  checkpoint_score:
    enabled: true
```

No change to `generate_configs.py` is required unless you want this on by default for a phase.

Jobs **already running** on code before 2026-06-03 do not log the new per-epoch metrics until restarted on updated code.
