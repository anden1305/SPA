# Preprocessing & epochs ablations (labs 2, 3, 5)

**Added 2026-06-06.** Canonical log for preprocessing rounds 1–3. Architecture sweep:
[`overnight_experiments_20260606.md`](overnight_experiments_20260606.md).

## Baselines (80 ep, per-lab in-cohort, from scratch)

| Lab | Best GMM NMI | Job |
|-----|-------------|-----|
| lab_5 | 0.474 (s2) | 28605498 |
| lab_2 | 0.334 (s3) | 28605499 |

## Round 1 — DONE (~18:10–20:40)

Generator: `scripts/cv4fold/generate_ablation_prepro.py`  
Submit: `bash hpc/submit/cv4fold/submit_ablation_prepro.sh`

| Variant | JOBID | seed1 | seed2 | seed3 | Best | Mean |
|---------|-------|-------|-------|-------|------|------|
| lab_5_long | 28605764 | 0.417 | 0.508 | 0.436 | **0.508** | 0.454 |
| lab_2_baseline_long | 28605765 | 0.336 | 0.309 | 0.385 | 0.385 | 0.343 |
| lab_2_widebp | 28605767 | 0.321 | 0.323 | 0.337 | 0.337 | 0.327 |
| lab_2_no_postnorm | 28605766 | 0.521 | 0.527 | 0.520 | **0.527** | **0.523** |
| lab_2_no_postnorm_widebp | 28605768 | 0.567 | 0.541 | 0.568 | **0.568** | **0.559** |

**Findings:** lab_2 gain is **`post_normalize: false`** (+0.19 mean); wide EEG band helps only combined with no postnorm. lab_5 modest gain from 200 epochs.

## Round 2 — running / partial (~23:12+)

Submit: `bash hpc/submit/cv4fold/submit_ablation_prepro_round2.sh`

| Variant | JOBID | Status | Best (partial) |
|---------|-------|--------|----------------|
| lab_3_baseline_long | 28606252 | running | 0.595 (s1) |
| lab_3_no_postnorm | 28606253 | running | 0.517 (s1) |
| lab_3_no_postnorm_widebp | 28606254 | running | 0.570 (s1) |
| lab_5_no_postnorm | 28606255 | running | 0.470 (s1–2) |
| lab_5_no_postnorm_widebp | 28606256 | PEND | — |

**Interim:** `post_normalize` **helps lab_3**, **hurts lab_2** → per-lab preprocessing required.

## Round 3 — paper front-end (PEND/RUN)

`robust_normalize: true`, `post/pre_normalize: false`, EEG 0.3–35 Hz (see [`lab_measurement_preprocessing_2_3_5.md`](../lab_measurement_preprocessing_2_3_5.md)).

Submit: `bash hpc/submit/cv4fold/submit_ablation_paper_robust.sh`

| Lab | JOBID |
|-----|-------|
| lab_2 | 28606501 |
| lab_3 | 28606502 |
| lab_5 | 28606503 |

## Latent separability & diagnostic plots

After each run, inspect under `results/cv4fold/ablation_prepro/<lab>/<run_name>/plots/`:

| Plot | What to look for |
|------|------------------|
| `hmm_tripanel_pc*_pc*.png` | **True** panel: overlapping classes → poor ceiling; **HMM** panel: sharp but wrong boundaries → preprocessing/arch issue not prior |
| `feature_amplitude_per_state.png` | State-wise EEG/EMG amplitude; Awake vs REM EMG separation |
| `feature_variance_per_state.png` | Spectral variance by state |
| `plots/<seed>/results.npz` | `x_latent`, `y_true`, `y_hat` for silhouette / confusion |

**Documented patterns (baseline 80 ep):**

- **lab_2 baseline** (`per_lab_incohort/.../162334/plots/`): True labels smeared in PC1–PC2; Awake↔REM confusion; silhouette ~0.22.
- **lab_5 baseline** (`.../161924/plots/`): True labels more clustered; silhouette ~0.31; NMI still climbing at ep80.
- **lab_2 no_postnorm_widebp** (`ablation_prepro/lab_2/abl_lab_2_no_postnorm_widebp_*/plots/`): tripanels show cleaner True-class structure vs baseline_long.

Scrape NMI + plot paths:

```bash
source .venv/bin/activate && python3 scripts/cv4fold/scrape_experiment_results.py
```

## Per-lab preprocessing winners (locked for arch sweep)

| Lab | `post_normalize` | EEG bandpass |
|-----|------------------|--------------|
| lab_2 | **false** | 0–30 Hz |
| lab_3 | **true** | 0–20 Hz |
| lab_5 | **true** | 0–20 Hz (+ 200 ep) |
