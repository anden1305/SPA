# Latent separability — what to use on this branch

**Added 2026-06-07.** Distilled from the `chmmgmm_fold` latent-separability notes, trimmed for the **per-lab → holdout → cross-lab** path on `vae_decoder_chmm_cv4`. No cyclical β, no full 20-mouse Bayes sweep as a prerequisite.

Related: [decoder_conditioning_roadmap.md](../decoder_conditioning_roadmap.md), [ablation_lab2_rem.md](ablations/ablation_lab2_rem.md), [overnight_experiments_20260606.md](overnight_experiments_20260606.md).

---

## One rule

**Ship decisions on GMM prior NMI** (`plots/<seed>/metrics.txt`). Separability plots explain *why* NMI is high or low; they do not replace NMI.

**Final cv4fold selection:** each YAML runs **3 seeds**; take the **best** prior NMI among seeds (not the mean). When comparing ablations, also check that **≥2 of 3 seeds** are healthy — one lucky seed is not enough if the other two collapse. See [README.md](README.md#selection-metric-final-cv4fold).

---

## Two spaces — do not mix them

| Plot / metric | Space | Level | When to use |
|---------------|-------|-------|-------------|
| `input_*_per_state.png`, `input_channel_statistics` in `data_validations.json` | **Pre-VAE** | **`plots/` once** | REM/atonia before encoder (lab_2) |
| `input_pairwise_ed_network.png`, `input_feature_separability_ranking.png` | **Pre-VAE** | **`plots/` once** | Input separability summary |
| `latent_pairwise_ed_network.png`, `latent_dim_separability_ranking.png`, `separability_input_vs_latent.png` | **Post-prior μ** | **`plots/` root** (best NMI seed) + **`plots/<seed>/`** (every seed) | Latent geometry vs input |
| `plots/<seed>/tripanel_pc1_pc2.png` | Latent PCA | **per seed** | Prior pred vs True |
| `plots/<seed>/feature_amplitude_per_state.png` | Post-prior μ | **per seed** | Which latent dims encode stages |
| `plots/<seed>/losses.png` | Training | **per seed** | Loss curve (from trainer even if epoch predictions missing) |

**Plot pipeline (2026-06-08):** `ensure_cv4fold_diagnostics()` in [`locked_recipes.py`](../../scripts/cv4fold/locked_recipes.py) turns on `validate_data` + separability flags. After each seed, encoder + prior-μ diagnostics land under `plots/<seed>/`; root `plots/` copies from the **best prior-NMI** seed.

If **input** EMG/EEG bands do not separate REM from NREM, fixing **arch or prior alone** rarely helps (lab_2 lesson). If **input** separates but **latent** does not, look at capacity / training / collapse.

Code: [`src/helpers/input_channel_statistics.py`](../../src/helpers/input_channel_statistics.py), [`src/helpers/frequency_statistics.py`](../../src/helpers/frequency_statistics.py), [`src/helpers/state_distinctness.py`](../../src/helpers/state_distinctness.py).

---

## How to read `feature_amplitude_per_state.png`

Built after training from **prior posterior means** μ (HMM-GMM path; encoder μ before prior validation is overwritten — see plot pipeline note below). Not samples, not decoder, not raw EMG.

- Each line = mean value of **latent dimension *k*** for Awake / NREM / REM; band = ±1 std **within that stage**.
- **Good enough for staging:** several dimensions with state-specific means; at least one axis where REM band does not sit on top of both others (e.g. lab_5 seq32 seed 2 **Feature 7** for REM, **Feature 4** for NREM).
- **Red flags:** most dimensions ≈ 0 for all states (**dead latents**); only 1–2 “hero” dims while REM overlaps everywhere else; pretty Awake/NREM split but REM smeared (typical lab_2 ~0.56 NMI).
- **Seed-specific structure:** a strong REM axis on one seed may be flat on others — check all three seeds before locking (lab_5 seq32: 0.640 vs 0.488/0.507).

**Do not** treat non-overlapping bands on a single dimension as “solved” — 3-way classification uses the **full** vector. Prefer `state_distinctness` numbers or the tripanel over one feature line.

Case study: [ablation_findings_20260608.md](ablations/ablation_findings_20260608.md#lab_5-seq32-seed-2-nmi-0640--feature-7--rem).

---

## Metric priority (this branch)

### 1. Primary — always

| Metric | Where | Use |
|--------|-------|-----|
| **GMM prior NMI** | `plots/<seed>/metrics.txt` | Per-lab gate (>0.45 incohort; lab_3 ~0.72 target), compare ablations |
| **Confusion matrix** | `plots/<seed>/` | REM recall, collapse (one predicted state ~50% while true REM ~6%) |

### 2. Secondary — same run, no extra jobs

From `data_validations.json` after `validate_data` + `validate_cvae`:

| Key | Meaning | Better = |
|-----|---------|----------|
| `weighted_mean_pairwise_energy` | Multivariate distance between stage clouds in latent space | Higher |
| `fisher_trace` | Between/within scatter ratio | Higher |
| `pairwise_energy` | Awake–NREM, Awake–REM, NREM–REM | All pairs large; **NREM–REM** matters for lab_2 |
| `input_*` / `latent_*` prefixed keys | Same metrics in **pre-VAE** vs **latent** space | Compare with `separability_input_vs_latent.png` |

Use these to **rank tie-breakers** among similar NMI runs, not to override NMI.

**W&B during training:** `val/entropy_norm` measures prior-prediction spread (0 = collapse, 1 = uniform ~33% per state). Target the **val-label baseline** (~0.80 for ~48/46/6% awake/NREM/REM; lower if val is sleep-heavy) — not 1.0. See [cvae_wandb_checkpoint_metrics.md §2.1](../training/cvae_wandb_checkpoint_metrics.md#21-what-is-a-good-valentropy_norm).

### 3. Ignore for selection (exploratory only)

- W&B / epoch **KMeans-on-latent NMI** — can disagree with prior NMI (Phase 1 tune lesson).
- `feature_amplitude` alone — interpretability only.

### 4. Later — cross-lab / folds

When you move to **joint or holdout** training:

- **Macro per-mouse NMI** (do not trust pooled NMI if one lab dominates).
- **Per-lab NMI** on val (`val_nmi_by_lab.csv` pattern from postprocess).
- **Holdout fold** mice only — in-sample all-20 metrics lie.

---

## Simple “is this run healthy?” checklist

**Good enough to lock a per-lab recipe:**

- [ ] Prior NMI meets your gate (lab_2 ~0.55+, lab_3 ~0.72, lab_5 TBD).
- [ ] All **3 predicted states** used (pred distribution not 2-state collapse).
- [ ] Tripanel **True** panel shows three blobs (not all smeared); **Pred** panel reasonably aligned.
- [ ] For REM-sensitive labs: check **input** `input_emg_band_power_per_state.png` *or* latent REM row on confusion matrix.

**Not good enough** (even if one plot looks OK):

- REM overlaps both others on **all** active latent dims + bad REM recall.
- Most latent dims dead (flat amplitude plot) unless NMI is already high (lab_3 wide_mlp is the exception — verify with NMI).

---

## What actually moved separability here (keep doing)

| Lever | Status on branch |
|-------|------------------|
| **Per-lab preprocessing** | lab_2: drop `post_normalize`, EEG 0–30 Hz; lab_3: postnorm + wide_mlp; lab_5: long epochs |
| **Per-lab arch** | lab_3 `wide_mlp`; lab_2 arch ≈ tie — bottleneck is input/REM not width |
| **Input diagnostics** | Always on with `summary_statistics: true` |
| **HQ cohort + signals** | Manifest mice only; **lab_2 EEG1+EEG4+EMG** (locked 2026-06-07) |
| **decoder_only_conditioning** | Encoder lab-agnostic; subject in decoder only |

---

## What to skip or defer (from chmmgmm_fold notes)

| Idea | Why skip *for now* |
|------|---------------------|
| Cyclical / exotic β schedules | Default β warmup in your YAMLs is enough; adds tuning surface |
| Phase 1 Bayes50 as gate before per-lab work | You already have per-lab winners from ablations |
| Optimizing **latent KMeans NMI** | Misaligns with prior NMI |
| **`sequence_length: 64`** joint recipe | Different setting than per-lab incohort (`seq1`); compare only when you deliberately switch to joint cv4fold |
| Extreme LR / 256-batch tune winners | Bayes50 cgmvae winner (~1.2e-5 LR) — only revisit if joint fold undertrains |
| **`subject_lab` / joint 20-mouse** | After per-lab incohort + holdout pass |

---

## Minimal workflow per experiment

Configs already set:

```yaml
validator:
  state_distinctness: true
  summary_statistics: true
visualizer:
  state_distinctness: true
  summary_statistics: true
validate_data: true
```

After a run:

```
results/cv4fold/<experiment>/lab_2/<run>/
  data_validations.json                 # input_* + latent_* (input kept across seeds)
  plots/                                # experiment-level (once)
    input_emg_band_power_per_state.png
    input_separation_gaps_bar.png
    input_pairwise_ed_network.png
    latent_pairwise_ed_network.png
    separability_input_vs_latent.png
    latent_dim_separability_ranking.png
  plots/<seed>/                         # per seed
    tripanel_pc1_pc2.png
    feature_amplitude_per_state.png
    metrics.txt                         # ← decision metric
```

Compare ablations with the same seed count (3) and epochs (200 for lab_2 sweeps). **Lock recipes on best-of-3 NMI**; use mean or all-seed lists only as a stability sanity check.

---

## How this connects to cross-lab (later)

1. **Per-lab incohort** — NMI + REM/confusion + input EMG (you are here).
2. **Per-lab fold holdout** — same metrics on **held-out mice only**.
3. **Joint training** — add macro/per-lab NMI; amplitude plot on **mixed** data will show lab batch effects — expect lab clusters before stage clusters until recipe is lab-aware.

Do not interpret joint pooled NMI or a single amplitude plot across labs until holdout per lab passes.

---

## Changelog

- **2026-06-07 (pm)** — Document **best-of-3** selection metric for final cv4fold; stability check on collapsed seeds.
- **2026-06-07** — Initial doc; trimmed from chmmgmm_fold draft; aligned with lab_2 REM + input diagnostics work.
