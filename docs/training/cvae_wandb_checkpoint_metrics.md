# CVAE / cHMM-GMVAE — W&B metrics and thesis checkpoint score

**Added:** 2026-06-07 · **Updated:** 2026-06-08 (`val/entropy_norm` interpretation)  
**Scope:** `train_vae` for cGMVAE and cHMM-GMVAE (cv4fold, decoder-only reliability, ablations).

Per-epoch Weights & Biases logging and optional **collapse-aware checkpoint** aligned with thesis composite score \(S(\varepsilon)\). Implementation: [`src/validation/validator.py`](../src/validation/validator.py), [`src/validation/hmmgmm_metrics.py`](../src/validation/hmmgmm_metrics.py), [`src/training/trainer.py`](../src/training/trainer.py), [`src/orchestrator/orchestrator.py`](../src/orchestrator/orchestrator.py).

Related: [cvae_checkpointing.md](cvae_checkpointing.md), [cv4fold README](../cv4fold/README.md).

---

## 1. Motivation

During training we need to:

1. **Monitor collapse** — prior predictions concentrating on one sleep state while loss still improves.
2. **Track the CV metric** — prior-prediction NMI (GMM marginal or HMM Viterbi), not only latent KMeans NMI.
3. **Select checkpoints** — thesis criterion balances validation log-likelihood and diverse state usage.

MARHMM training already logs `perplexity` and `nlpp` in `validate_epoch`. CVAE training logs `cvae_latent_kmeans_nmi` each validation epoch; post-train `metrics.txt` NMI uses the best saved checkpoint.

---

## 2. Thesis checkpoint score (Eq. 3.49–3.50)

Let \(\tilde{\omega}(\varepsilon)\) be validation **log-likelihood** (same sign convention as training: \(-\mathrm{NLL}\) on the val set).

Let \(H(\hat{Y})\) be Shannon entropy of the empirical predicted state distribution over \(K_{\mathrm{pred}}\) states. **Normalized entropy:**

\[
H_{\mathrm{norm}}(\hat{Y}) = \frac{H(\hat{Y})}{\log K_{\mathrm{pred}}} \in [0, 1]
\]

- \(0\) = all mass on one state (collapse)
- \(1\) = **uniform** usage over \(K_{\mathrm{pred}}\) states (~33% each for \(K=3\)) — **not** the ideal target for sleep staging (real hypnograms are skewed)

### 2.1 What is a good `val/entropy_norm`?

`val/entropy_norm` is \(H_{\mathrm{norm}}(\hat{Y})\) on **prior predictions** (GMM marginal or HMM Viterbi labels) over the validation set — see [`entropy_norm`](../../src/validation/hmmgmm_metrics.py).

**Target = validation label baseline.** Compute normalized entropy on **scored val stages** once per fold/lab; a healthy prior should stay near that value while `val/prior_pred_nmi` rises. Do **not** expect values near 1.0 unless predictions are artificially uniform.

```python
from src.validation.hmmgmm_metrics import entropy_norm
# y_val: flattened scored stage labels on the val split; k_pred = 3
baseline = entropy_norm(y_val.astype(int), k_pred=3)
```

**Typical ranges (REM ≈ 6%, wake + NREM split the rest):**

| Wake | NREM | REM | Baseline \(H_{\mathrm{norm}}\) |
|------|------|-----|--------------------------------|
| 5% | 89% | 6% | ~0.38 (very sleep-heavy val) |
| 10% | 84% | 6% | ~0.50 |
| 15% | 79% | 6% | ~0.58 |
| 20% | 74% | 6% | ~0.65 |
| 48% | 46% | 6% | ~0.80 (HQ-like mix; see [lab preprocessing review](../cv4fold/lab_preprocessing_review_20260607.md)) |

**How to read curves:**

| `entropy_norm` vs baseline | Likely meaning |
|--------------------------|----------------|
| Stable within ~±0.05–0.10, `prior_pred_nmi` ↑ | Healthy marginal calibration |
| → 0, `n_unique_pred_states` → 1 | Collapse |
| → 1.0 while true REM ~6% | Over-uniform predictions (e.g. too much REM/wake) |
| Plausible entropy but low REM recall | REM merged into NREM — check confusion matrix; entropy alone is not enough |

Pair with `val/n_unique_pred_states` (expect 3) and confusion-matrix REM recall.

**Composite score:**

\[
S(\varepsilon) = \beta \, \tilde{\omega}(\varepsilon) + (1 - \beta) \, H_{\mathrm{norm}}(\hat{Y})
\]

Default \(\beta = 0.9\). The epoch maximizing \(S\) after warmup (default 5% of training) may be saved as `cvae_best_checkpoint_score.pth`.

**Note:** This is **not** the MARHMM `nlpp` metric (`0.9 * log(likelihood) + 0.1 * log(perplexity/K)`).

---

## 3. W&B metrics (per validation epoch)

Logged when `wandb.enabled` is true. Keys appear under `val/` via [`ExperimentLogger`](../src/training/experiment_logger.py).

| W&B key | Meaning |
|---------|---------|
| `val/prior_pred_nmi` | NMI of prior labels vs scored stages (GMM marginal or HMM Viterbi) |
| `val/cvae_latent_kmeans_nmi` | KMeans on latent means vs stages (unsupervised probe; can diverge from prior NMI) |
| `val/log_likelihood` | Validation log-likelihood \(\tilde{\omega}\) |
| `val/entropy_norm` | \(H_{\mathrm{norm}}(\hat{Y})\) from prior predictions |
| `val/checkpoint_score` | \(S(\varepsilon)\) |
| `val/n_unique_pred_states` | Unique prior-predicted states (expect 3 for wake/NREM/REM) |
| `val/prior_switch_rate_per100` | State transitions per 100 steps on prior label sequence |
| `train/total_loss`, `train/reg_loss` | Optimization health (β / HMM warmup) |

**Not logged per epoch:** silhouette, state distinctness (still computed post-train when enabled).

### Run summary (`wandb.summary`)

| Key | When set |
|-----|----------|
| `val/best_checkpoint_score` | Best \(S\) during training (if any validation epoch ran) |
| `val/best_checkpoint_score_epoch` | 1-based epoch index |
| `val/best_prior_pred_nmi` | Best prior-prediction NMI during training |
| `val/best_prior_pred_nmi_epoch` | 1-based epoch index |
| `val/prior_pred_nmi` | After post-train eval on the selected checkpoint |

---

## 4. Checkpoint selection (this branch)

See [cvae_checkpointing.md](cvae_checkpointing.md) for full load/save semantics.

### Default (`checkpoint_score.enabled: false`)

1. During training, save **`cvae_best_prior_pred_nmi.pth`** when `val/prior_pred_nmi` improves.
2. Post-train priority: **prior-pred NMI best** → **`cvae_final_model.pth`**.
3. `validation_checkpoint.txt` records which file was used.

Latent KMeans NMI is logged but **not** used for checkpoint selection on this branch.

### Thesis-aligned (`checkpoint_score.enabled: true`)

```yaml
trainer:
  checkpoint_score:
    enabled: true
    beta: 0.9
    warmup_frac: 0.05
```

Post-train priority: **score** → **prior-pred NMI best** → **final**.

---

## 5. Optional config toggles (2026-06-07)

Defaults preserve existing job behaviour; omit from YAML to keep current settings.

| YAML key | Default | Use |
|----------|---------|-----|
| `validator.validate_train` | `true` | Set `false` to skip train-set NMI during sweeps (MARHMM `validate_epoch` only) |
| `dataloader.max_batches_per_epoch` | `null` | Cap batches per epoch for smoke/debug |
| `visualizer.save_results_npz` | `true` | Set `false` to skip large `plots/results.npz` on sweeps |

Example sweep-friendly snippet:

```yaml
validator:
  validate_train: false
visualizer:
  save_results_npz: false
dataloader:
  max_batches_per_epoch: 20   # smoke only
```

---

## 6. Code map

| Component | Path |
|-----------|------|
| \(H_{\mathrm{norm}}\), \(S(\varepsilon)\) | `src/validation/hmmgmm_metrics.py` |
| Per-epoch CVAE validation | `Validator.validate_cvae_epoch` |
| Prior-pred checkpoint | `Trainer._maybe_save_best_prior_checkpoint` |
| Score checkpoint | `Trainer._maybe_save_best_checkpoint_score` |
| Post-train checkpoint choice | `Orchestrator.train_cvae` |
| Config | `CheckpointScoreConfig`, `ValidatorConfig`, `DataLoaderConfig`, `VisualizerConfig` in `src/config/config.py` |

---

## 7. Interpreting curves

| Pattern | Likely meaning |
|---------|----------------|
| `prior_pred_nmi` ↑, `entropy_norm` stable near val-label baseline (~0.5–0.8 for typical mixes) | Healthy learning |
| `log_likelihood` ↑, `entropy_norm` → 0, `n_unique_pred_states` → 1 | Collapse — do not trust peak KMeans NMI alone |
| `entropy_norm` → 1.0 while scored REM ~6% | Over-uniform prior marginals — inspect confusion matrix |
| `cvae_latent_kmeans_nmi` high, `prior_pred_nmi` low | Latent separability without good prior alignment |
| `prior_switch_rate_per100` → 0 (chmm) | Sticky or collapsed HMM path |
| `checkpoint_score` peaks before final epoch | Enable `checkpoint_score.enabled: true` or inspect prior-pred-best epoch |

See [§2.1](#21-what-is-a-good-valentropy_norm) for baseline tables and the one-off `entropy_norm(y_val, 3)` recipe.

Rank experiments by **`metrics.txt` / prior-pred NMI**, not latent KMeans alone.

---

## 8. Jobs already in queue

These changes use **Pydantic defaults** only — existing YAML without the new keys behaves identically to before:

- Train-set NMI still runs for MARHMM (`validate_train` default `true`).
- Full dataset per epoch (`max_batches_per_epoch` default `null`).
- `results.npz` still written (`save_results_npz` default `true`).
- Non-finite batch guards only skip bad steps; clean runs are unchanged.

Restarted jobs pick up the new code; **running jobs** keep the code snapshot from submit time.
