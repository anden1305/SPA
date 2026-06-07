# Preprocessing & epochs ablations (labs 2, 3, 5)

**Added 2026-06-06.** Updated **2026-06-07** with R2–R4 results and **best-of-3** locked winners.

**Selection metric:** for each YAML, **`runs: 3`** → report **best** GMM prior NMI across seeds (`scrape_experiment_results.py`). Mean shown for stability only. See [README.md](README.md#selection-metric-final-cv4fold).

Architecture sweep: [`overnight_experiments_20260606.md`](../overnight_experiments_20260606.md).

## Baselines (80 ep, per-lab in-cohort, from scratch)

| Lab | Best-of-3 NMI | Job |
|-----|---------------|-----|
| lab_5 | 0.474 (s2) | 28605498 |
| lab_2 | 0.334 (s3) | 28605499 |

## Round 1 — DONE

Generator: `scripts/cv4fold/generate_ablation_prepro.py`  
Submit: `bash hpc/submit/cv4fold/submit_ablation_prepro.sh`

| Variant | Best | Mean | seeds | Winner? |
|---------|------|------|-------|---------|
| lab_5_long | **0.508** | 0.454 | [0.417, 0.508, 0.436] | **lab_5 prepro** |
| lab_2_baseline_long | 0.385 | 0.343 | [0.336, 0.309, 0.385] | no |
| lab_2_widebp | 0.337 | 0.327 | [0.321, 0.323, 0.337] | no |
| lab_2_no_postnorm | 0.527 | 0.523 | [0.521, 0.527, 0.520] | partial |
| lab_2_no_postnorm_widebp | **0.568** | 0.559 | [0.567, 0.541, 0.568] | **lab_2 prepro** |

**Finding:** lab_2 gain is **`post_normalize: false`** (+0.19 mean vs baseline); wide EEG band (0–30 Hz) helps only with no postnorm.

## Round 2 — DONE

Submit: `bash hpc/submit/cv4fold/submit_ablation_prepro_round2.sh`

| Variant | Best | Mean | seeds | Winner? |
|---------|------|------|-------|---------|
| lab_3_baseline_long | **0.713** | 0.649 | [0.595, 0.639, 0.713] | **lab_3 prepro** |
| lab_3_no_postnorm | 0.628 | 0.563 | [0.517, 0.628, 0.543] | no |
| lab_3_no_postnorm_widebp | 0.616 | 0.540 | [0.570, 0.616, 0.433] | no |
| lab_5_no_postnorm | 0.471 | 0.386 | [0.470, 0.318, 0.471] | no |
| lab_5_no_postnorm_widebp | 0.471 | 0.402 | [0.382, 0.354, 0.471] | no |

**Finding:** `post_normalize` **helps lab_3**, **hurts lab_2** → per-lab preprocessing required.

## Round 3 — paper front-end — DONE

`robust_normalize: true`, `post/pre_normalize: false`, EEG 0.3–35 Hz.

| Lab | Best | Mean | seeds | Winner? |
|-----|------|------|-------|---------|
| lab_2 | 0.514 | 0.377 | [0.234, 0.514, 0.382] | no |
| lab_3 | 0.616 | 0.604 | [0.616, 0.600, 0.595] | no |
| lab_5 | 0.365 | 0.284 | [0.365, 0.284, 0.202] | no |

**Finding:** shared paper front-end does **not** beat per-lab tuned prepro on any lab.

## Round 4 — EEG 0.3–25 Hz (lab_3, lab_5) — partial

Mid band + paper-style high-pass 0.3 Hz. Compare to locked winners above.

| Variant | Status | Best | seeds | vs winner |
|---------|--------|------|-------|-----------|
| lab_3 `no_postnorm_bp25` | RUN | 0.551 | [0.551] (1/3) | below lab_3 baseline_long **0.713** |
| lab_5 `no_postnorm_bp25` | RUN | 0.357 | [0.333, 0.357] (2/3) | below lab_5 long **0.508** |

**Interim:** bp25 unlikely to replace locked recipes; let jobs finish then archive.

## Locked per-lab recipes (best-of-3, from-scratch HQ incohort)

| Lab | Config source | Montage / prepro | Best NMI |
|-----|---------------|------------------|----------|
| **lab_2** | `ablation_rem/.../rem_emg_wide_eeg4.yaml` | **EEG1+EEG4+EMG**, EMG 3–100 Hz, no postnorm, EEG 0–30 Hz | **0.593** |
| **lab_3** | `baseline_long` + `wide_mlp` arch | EEG1+EEG2+EMG, postnorm, EEG 0–20 Hz | **0.737** |
| **lab_5** | `long` + `wide_mlp` arch | EEG1+EEG2+EMG, postnorm, EEG 0–20 Hz | **0.534** |

lab_5 remains the **weakest lab** (best ~0.53 with arch); gate >0.45 passed but no ~0.72-class ceiling yet.

## Latent separability & diagnostic plots

After each run, inspect under `results/cv4fold/ablation_prepro/<lab>/<run_name>/plots/`:

| Plot | What to look for |
|------|------------------|
| `hmm_tripanel_pc*_pc*.png` | True panel smearing → prepro issue |
| `input_emg_band_power_per_state.png` | REM atonia before encoder (lab_2) |
| `plots/<seed>/metrics.txt` | **Best-of-3 prior NMI** |

Scrape:

```bash
source .venv/bin/activate && PYTHONPATH=. python3 scripts/cv4fold/scrape_experiment_results.py
```

## Changelog

- **2026-06-07 (eve)** — lab_2 montage locked **EEG1+EEG4+EMG** (`rem_emg_wide_eeg4`, 0.593).
- **2026-06-07** — R2–R3 complete; R4 bp25 partial; locked winners table (best-of-3); paper_robust rejected.
- **2026-06-06** — Initial R1 log.
