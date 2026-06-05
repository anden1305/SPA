# cv4fold Phase 1 tune (`bayes50`) — condensed results

**Period:** May 25–27, 2026 · **Setup:** joint, all 20 mice, in-sample train=val, fold-4 cohort configs  
**Sweeps:** chmmgmvae `1gbsbad1` (50 trials), cgmvae `8e8jftet` (50 trials)

> **Metric caveat:** These runs optimized W&B `val/cvae_latent_kmeans_nmi` (latent KMeans during training), **not** prior-prediction NMI (`metrics.txt`). Rank trials by **`metrics.txt` NMI** until you re-sweep with `val/prior_pred_nmi` (wired in code + sweep YAML after May 2026 fix).

---

## Headline numbers (prior NMI from `metrics.txt`)

| Model | Trials | Best pooled | Median pooled | Collapsed (NMI &lt; 0.05) |
|--------|--------|-------------|---------------|---------------------------|
| **chmmgmvae** | 53 | **0.560** (`5bkceef8`) | 0.335 | 16 (30%) |
| **cgmvae** | 53 | **0.513** (`8duimqhc`) | 0.086 | 24 (45%) |

**Takeaway:** chmmgmvae is stronger and more stable; cgmvae has a narrow good region and many failed trials.

---

## Best trials to lock for Phase 2

### chmmgmvae — `5bkceef8` (best on disk)

| | |
|--|--|
| Pooled NMI | 0.560 |
| Macro NMI (equal per mouse) | 0.617 |
| Lab means (mouse avg) | lab_2 **0.48**, lab_3 **0.72**, lab_5 **0.55** |

**Hyperparams (rounded):** `lr=0.0014`, `epochs=257`, `validate_per_epoch=5`, `batch_size=32`, `num_batches=32`, `latent_dim=4`, `emb_dim=8`, `beta_schedule=anneal`, `min_beta≈0.016`, `gmm_warmup=18`, `hmm_warmup=37`, `hmm_ramp=18`, `hmm_sticky_kappa≈0.86`, `ridge≈0.18`, `var_reg≈0.016`

**Weak mice:** sub-072 (0.34). **Strong:** sub-056/060 (~0.78–0.80).

*W&B latent leader `0z8ird8h` (0.52 pooled) is worse on sub-072 and lab_5 — do not pick from W&B alone.*

### cgmvae — `8duimqhc`

| | |
|--|--|
| Pooled NMI | 0.513 |
| Macro NMI | 0.520 |
| Lab means | lab_2 **0.40**, lab_3 **0.62**, lab_5 **0.46** |

**Hyperparams:** `lr≈1.2e-5`, `epochs=240`, `num_batches=256`, `latent_dim=16`, `gmm_warmup=0`, `beta_schedule=anneal`, `batch_size=64`

---

## What worked / what failed

**Worked**

- chmmgmvae, `latent_dim=4`, lr ~**1e-3**, long runs (185–257 ep), moderate HMM stickiness
- Enough β/HMM warmup; post-train HMM switch ~3–4/100 on best chmm trial (3 states)

**Failed**

- ~30–45% collapsed trials (bad β schedule, lr too low, &lt; ~80 ep, harsh HMM warmup)
- Optimizing latent KMeans ≠ good Viterbi/GMM predictions
- cgmvae: most draws useless except very low lr + large `num_batches`

**Collapsed chmm pattern (example `ynb7ame7`):** 82 ep, lr ~1.4e-4, `latent_dim=8`, `hmm_warmup=74` vs winner 257 ep, lr ~1.4e-3, `latent_dim=4`, `hmm_warmup=37`

---

## Why pooled NMI is not “good everywhere”

1. **Token weighting** — validation mass: lab_5 **51%** (sub-087 alone **38%**), lab_3 **36%**, lab_2 **13%**. Pooled NMI favours lab_3 + sub-087.
2. **Channel mismatch** — lab_2 uses EEG1+**EEG3**; lab_3/5 use EEG1+**EEG2**. Joint subject conditioning across montages is hard.
3. **Problem mice** — **sub-072** often weak; **sub-059** unstable (smoke80 holdout ~0.002, best tune ~0.64). Not one bad lab: **lab_2 often ~0 in failed trials**, lab_3 stays high.
4. **In-sample tune** — high here does not imply fold holdout generalization (smoke80 holdout: sub-080 ~0.66, sub-059/060 ~0).

**Reference:** per-lab lab_3-only cgmvae tune (3 trials) → **0.61–0.63** (comparable to joint lab_3 mice when tuning works).

---

## Manifest starting point (chmmgmvae)

```yaml
hyperparams:
  learning_rate: 0.0014
  epochs: 260
```

Full `model.params` (β, HMM warmup, `latent_dim: 4`, etc.) should be copied from `5bkceef8` sweep config (`wandb/sweep-1gbsbad1/config-5bkceef8.yaml`) when running `generate_configs.py --phase full`.

---

## Next steps

1. Re-submit tune with **`val/prior_pred_nmi`** (`submit_tune_sweep.sh` after regenerating sweep YAMLs).
2. Phase 2: lock **`5bkceef8`** (chmm) hyperparams; enable per-mouse exports for CV.
3. Paper cross-lab narrative: report **macro (per-mouse) NMI** and/or **per-lab scopes**, not pooled alone.

**Artifacts:** `results/cv4fold/tune_sweep/{chmmgmvae,cgmvae}/joint/all_mice/<trial_id>/`
