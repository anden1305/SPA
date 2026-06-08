# cGMVAE incohort ablation analysis — latent separability & preprocessing

**Added 2026-06-08.** Synthesis of all cGMVAE incohort ablation runs (metrics, plots, training curves, `data_validations.json`) with actionable ideas to push in-cohort NMI higher.

Related: [latent_separability_guide.md](../latent_separability_guide.md), [ablation_lab2_findings_20260607.md](ablation_lab2_findings_20260607.md), [ablation_prepro_lab2_lab5.md](ablation_prepro_lab2_lab5.md), [incohort_to_cross_lab_strategy.md](../incohort_to_cross_lab_strategy.md), [ablation_findings_20260608.md](ablation_findings_20260608.md) (cHMM + seq-length + enc+dec synthesis), [`locked_recipes.py`](../../../scripts/cv4fold/locked_recipes.py).

**Data source:** 51 runs scraped from `results/cv4fold/ablation_*` (metrics.txt + separability JSON + LSF checkpoint logs).

---

## Executive summary

The locked recipes are real wins, but **each lab hits a different ceiling for different reasons**:

| Lab | Locked NMI | Main bottleneck | Encoder behavior |
|-----|------------|-----------------|------------------|
| **lab_2** | **0.593** | **NREM–REM** (input almost identical; latent barely expands) | **Compresses** separability (input → latent gap shrinks) |
| **lab_3** | **0.737** | Mostly solved for scratch; wide MLP is the lever | **Expands** usable latent geometry (multi-hero dims) |
| **lab_5** | **0.534** | Weak input + needs lab-specific EMG/notch | **Can** expand NREM–REM strongly (EMG ablation) |

**Headline:** lab_2 is an **REM-recall / NREM–REM geometry** problem, not a “more preprocessing” problem. lab_3 is an **architecture/capacity** problem (already fixed). lab_5 is a **lab-specific front-end** problem with a promising untested combo (`emg_wide + notch50`).

---

## What the numbers actually say (51 runs scraped)

### Locked winners (best-of-3, scratch)

| Lab | Recipe | Best | Seeds | Stability |
|-----|--------|------|-------|-----------|
| lab_2 | `rem_emg_wide_eeg4` | **0.593** | 0.593, 0.578, 0.541 | ✅ all healthy |
| lab_3 | `baseline_long` + **`wide_mlp`** | **0.737** | 0.737, 0.728, 0.695 | ✅ |
| lab_5 | `long` + **`wide_mlp`** | **0.534** | 0.534, 0.438, 0.404 | ⚠️ seed spread |

### Preprocessing — per-lab is mandatory (not optional)

- **lab_2:** `post_normalize: false` + EEG **0–30 Hz** → +0.19 mean NMI vs baseline. `postnorm` on lab_2 is dead.
- **lab_3:** **`postnorm` helps** (0.713 vs 0.628 no_postnorm). Opposite of lab_2.
- **lab_5:** `postnorm` + 200 ep `long` beats no_postnorm variants.
- **Shared “paper_robust” front-end failed all labs** (lab_2 best 0.514 with collapsed seeds).
- **bp25** trending worse everywhere; do not pursue.

### Architecture — lab-specific

| Lab | wide_mlp vs prepro-only | Notes |
|-----|-------------------------|-------|
| lab_2 | **0.530** vs **0.568** prepro | Architecture **does not beat** prepro on old montage; **`wide_mlp` never tested on locked `rem_emg_wide_eeg4`** |
| lab_3 | **0.737** vs **0.713** prepro | **`wide_mlp` is essential** (+0.024) |
| lab_5 | **0.534** vs **0.508** prepro | **`wide_mlp` helps** (+0.026) |

**Collapse warning:** `lat16` on lab_2 (0.486, 2/3 seeds ~0), `no_beta_epochs: 0` on lab_3/lab_5 (all seeds 0.0). **`no_beta_epochs: 10` is non-negotiable** except lab_2 where β-from-0 is merely slightly worse (0.563 vs 0.593).

### lab_2 montage + EMG (biggest incohort gain)

Progression on same prepro base:

```
no_postnorm_widebp  0.568
eeg1_eeg4           0.551 (stable)
rem_winner (EEG3)   0.575
rem_emg_wide_eeg4   0.593  ← LOCK
```

- Wide EMG on **EEG3 collapses** seed 3 (0.097) → discard.
- On **EEG4**, wide EMG is stable and wins.
- **`notch50` on lab_2 hurts** (0.531) despite decent input stats.

---

## Latent separability — the new read

### Rule: two spaces, two stories

From `data_validations.json` + plots under `results/cv4fold/.../plots/`:

| Space | Plots | Question |
|-------|-------|----------|
| **Pre-VAE (input)** | `input_emg_band_power_per_state.png`, `input_separation_gaps_bar.png` | Is the problem fixable before the encoder? |
| **Post-VAE (latent μ)** | `separability_input_vs_latent.png`, `feature_amplitude_per_state.png`, `tripanel_pc1_pc2.png` | Did the encoder preserve or destroy stage geometry? |

See [latent_separability_guide.md](../latent_separability_guide.md) for the full checklist.

---

### lab_2 — input is *good enough* for Awake; encoder + prior lose REM

**Input EMG (`rem_emg_wide_eeg4`):** clear atonia pattern — Awake EMG ≈ −10.3, NREM ≈ −14.3, REM ≈ −15.1 (log-power). Awake–REM gap ≈ **4.83**.

**But NREM–REM input pairwise energy ≈ 0.27** — almost overlapping in raw FFT space.

**Latent pairwise (locked run):**

| Pair | Distance |
|------|----------|
| Awake–NREM | 2.96 |
| Awake–REM | 3.88 |
| **NREM–REM** | **0.80** ← ceiling |

The separability plot confirms the encoder **shrinks** Awake–NREM (4.17→2.0) and Awake–REM (5.9→3.1) while NREM–REM stays tiny. This is the **opposite** of lab_5’s EMG ablation.

**Tripanel (seed 1, best NMI 0.593):** PC1 separates Awake; true labels show REM along PC2, but **predictions collapse REM into NREM** (almost no green in pred panel).

**Tripanel (seed 3, NMI 0.541):** visually **better 3-way separation** on PC2 — yet lower NMI. Suggests prior/NMI is rewarding Awake vs sleep more than REM recall. **Optimize for REM confusion matrix, not peak NMI alone.**

**Feature amplitude:** Feature 7 separates Awake from sleep; Features 4/7 partially split NREM/REM, but bands overlap. Most dims are dead (~0 for all states).

**Diagnosis:** lab_2 bottleneck is **not** “EMG doesn’t work in input” — it’s **NREM–REM indistinguishability in input + encoder regularization that doesn’t expand that pair + GMM prior that merges REM into NREM**.

**Result path:** `results/cv4fold/ablation_rem/lab_2/abl_lab_2_rem_emg_wide_eeg4_20260607-155225/`

---

### lab_3 — wide MLP creates multi-axis latent code

At **0.737**, latent Fisher ≈ **3.07+** (similar magnitude to lab_2 but with better stage geometry overall).

**Feature amplitude:** Feature 4 → Awake vs sleep; Feature 7 → NREM vs non-NREM. **Multiple active dims** — why `wide_mlp` wins.

Scratch **0.737 already beats** the ~0.72 hotstart reference on this HQ subset. Further incohort gains are incremental (epochs, fine capacity), not a new prepro axis.

**Result path:** `results/cv4fold/ablation_arch/lab_3/arch_lab_3_wide_mlp_20260607-060104/`

---

### lab_5 — encoder *can* fix weak input (if front-end is right)

**Locked `wide_mlp` (0.534):** moderate, unstable seeds.

**`emg_wide + notch50` (partial, 1 seed complete):** NMI **0.540** with striking separability:

| Pair | Input | Latent |
|------|-------|--------|
| Awake–NREM | 0.85 | **3.05** |
| Awake–REM | 0.95 | **3.44** |
| **NREM–REM** | **0.09** | **2.35** |

Latent/input energy ratio ≈ **3.88** (lab_2 locked ≈ **0.72**). Same VAE family, **opposite encoder behavior** — driven by preprocessing + postnorm scale (lab_5 input is postnorm-normalized ~0 mean, lab_2 raw log-spectrum ~−18).

**This is the strongest lab_5 incohort lead:** finish 3-seed `emg_wide_notch50` before holdout.

**Result path:** `results/cv4fold/ablation_emg/lab_5/abl_lab_5_wide_mlp_emg_wide_notch50_20260607-195216/`

---

## Training curves — small details that matter

From checkpoint traces in LSF logs (`Saved best prior-pred NMI checkpoint`):

### lab_2 locked (job 28607461)

- Seed 1: still climbing at ep **196 → 0.593** (peak). **Do not early-stop at 200** without testing longer.
- Flat prior NMI until ep ~10 (`no_beta_epochs: 10`) — expected, keep it.
- EEG4 montage still improving at ep 200; EEG3 peaked ~ep 71 then drifted.

Log: `hpc/output/cv4fold/ablation_rem/lab_2_rem_emg_wide_eeg4_28607461.out`

### lab_3 wide_mlp (job 28606515)

- Rapid rise ep 45–80 (0.59 → 0.70+), then slow polish to ~0.737.
- **`no_beta_epochs: 0` → complete collapse** (latent Fisher 0).

Log: `hpc/output/cv4fold/ablation_arch/lab_3_wide_mlp_28606515.out`

### lab_5 wide_mlp

- Slow climb (81 checkpoint updates in log); weaker lab needs **more epochs** (300 ep already used for some lab_5 jobs).

Log: `hpc/output/cv4fold/ablation_arch/lab_5_wide_mlp_28606519.out`

---

## Prioritized ideas to push incohort higher

### P0 — Highest expected return

1. **lab_5: complete `wide_mlp + emg_wide + notch50` (3 seeds, 300 ep)**  
   Only 1/3 seeds in results; separability plot is the best in the entire sweep. If best-of-3 beats 0.540 → update locked recipe.

2. **lab_2: `rem_emg_wide_eeg4` + `wide_mlp` (never tested together)**  
   Arch sweep used old montage/prepro. lab_3/lab_5 show wide_mlp helps when latent geometry matters. Cheap 3-seed job.

3. **lab_2: 300 epochs on locked recipe**  
   Best seed still climbing at 200; lowest-risk change.

### P1 — Target NREM–REM specifically (lab_2)

4. **REM-weighted prior / confusion-aware selection**  
   Best-NMI seed collapses REM visually; seed 3 looks better but scores lower. Consider selecting checkpoints on **REM recall** or macro-F1, not prior NMI alone.

5. **`append_channel_rms` on EEG4 recipe**  
   Hit 0.568 on one seed, unstable — worth one controlled rerun if you want an input-side atonia cue without leaving VAE path.

6. **Slightly wider latent on lab_2 only** (`wide_lat16` got 0.564 on old prepro — unstable but untested on locked montage).

**Skip:** lab_2 notch50, paper_robust, postnorm, β-from-epoch-0 (lab_3/5), shared front-ends.

### P2 — lab_3 (already strong)

7. **Marginal:** 250–300 epochs, or `wide_lat16` if you want ~0.736 with fewer params — not worth much sweep time.  
8. **Do not disable `no_beta_epochs: 10`** — catastrophic collapse.

### P3 — Cross-cutting diagnostics for next ablations

When comparing runs, track these **together** (not NMI alone):

| Diagnostic | Good sign | lab_2 locked |
|------------|-----------|--------------|
| `input_emg_band_power_per_state.png` | REM < NREM < Awake | ✅ |
| NREM–REM input pairwise | > ~0.5 | ❌ (~0.27) |
| NREM–REM **latent** pairwise | > ~1.5 | ❌ (~0.80) |
| latent/input energy ratio | > 1 if input weak | ❌ (0.72) |
| Tripanel pred | 3 colors in sleep cluster | ❌ seed1 REM collapse |
| 3 seed health | no seed < 0.15 | ✅ |

---

## Suggested next experiments (minimal matrix)

| Priority | Lab | Change | Hypothesis | Gate |
|----------|-----|--------|------------|------|
| 1 | lab_5 | EMG 3–100 Hz + notch50 + wide_mlp, 300ep, 3 seeds | Encoder expands NREM–REM (seen in separability) | best-of-3 > **0.540** |
| 2 | lab_2 | locked prepro/montage + **wide_mlp** | Capacity for NREM–REM without hurting stability | best-of-3 > **0.593** |
| 3 | lab_2 | locked recipe, **300 epochs** | Still climbing at 200 | best-of-3 > **0.60** |
| 4 | lab_2 | same + checkpoint on **REM recall** | Fix NMI/geometry mismatch | REM recall ↑ at similar NMI |

**Submitted 2026-06-08** — generator + submit:

```bash
PYTHONPATH=. python3 scripts/cv4fold/generate_ablation_incohort_followup.py
bash hpc/submit/cv4fold/submit_ablation_incohort_followup.sh
```

Configs: `src/config/run/cvaemarhmm/cv4fold/ablation_followup/`  
Logs: `hpc/output/cv4fold/ablation_followup/<lab>/followup_*_%J.out`  
Results: `results/cv4fold/ablation_followup/<lab>/abl_followup_*`

### Inspect key artifacts

```bash
# lab_2 locked — separability + tripanel
ls results/cv4fold/ablation_rem/lab_2/abl_lab_2_rem_emg_wide_eeg4_20260607-155225/plots/

# lab_3 wide_mlp — feature amplitude + metrics
cat results/cv4fold/ablation_arch/lab_3/arch_lab_3_wide_mlp_20260607-060104/plots/1/metrics.txt

# lab_5 EMG lead — input vs latent bar chart
ls results/cv4fold/ablation_emg/lab_5/abl_lab_5_wide_mlp_emg_wide_notch50_20260607-195216/plots/
```

---

## Bottom line

Most **preprocessing** juice is already extracted (per-lab postnorm, lab_2 montage/EMG, lab_3 wide_mlp). Further incohort gains are about:

1. **lab_2:** Stop treating global NMI as sufficient — **REM is the missing state** in the best run’s predictions. Fix NREM–REM in latent space (capacity + training length + checkpoint criterion), not another shared prepro sweep.

2. **lab_5:** Follow the separability signal — **`emg_wide + notch50` turns the encoder into a separator**; finish validating it.

3. **lab_3:** Lock and move on — scratch **0.737** is the ceiling of this ablation program; effort better spent on holdout.

---

## Changelog

- **2026-06-08** — Initial synthesis from 51 ablation runs (metrics, separability JSON, plots, LSF curves).
