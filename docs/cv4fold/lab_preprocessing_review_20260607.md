# Lab-difference review & preprocessing ablation roadmap

**Added 2026-06-07.** Data-driven synthesis of [`results/data_exploration/`](../../results/data_exploration) (HQ review under [`hq_review/`](../../results/data_exploration/hq_review/)), [`results/noise/`](../../results/noise), acquisition notes in [`lab_measurement_preprocessing_2_3_5.md`](../preprocessing/lab_measurement_preprocessing_2_3_5.md), and completed cv4fold ablations.

HQ plots: `PYTHONPATH=. python3 scripts/data_exploration/hq_lab_review.py`

---

## Locked recipes (best-of-3 incohort scratch)

| Lab | Prepro | Arch | Montage / EMG | Best NMI |
|-----|--------|------|---------------|----------|
| **lab_2** | no postnorm, EEG **0–30 Hz** | default | **EEG1+EEG4+EMG**, EMG **3–100 Hz** | **0.593** |
| **lab_3** | postnorm, EEG **0–20 Hz**, 200 ep | **`wide_mlp`** | EEG1+EEG2+EMG | **0.737** |
| **lab_5** | postnorm, EEG **0–20 Hz**, 200 ep | **`wide_mlp`** | EEG1+EEG2+EMG | **0.534** |

Manifest [`cv_quality_cohort_v1.yaml`](../../data/manifests/cv_quality_cohort_v1.yaml) and [`locked_recipes.py`](../../scripts/cv4fold/locked_recipes.py) now use **lab_2 EEG1+EEG4**.

---

## Data evidence (HQ cohort)

| Signal | lab_2 | lab_3 | lab_5 | Implication |
|--------|-------|-------|-------|-------------|
| 50 Hz interference | **9/12** recordings with outlier; ratios up to **1548×** (sub-072) | **0/18** | **2/12** (sub-089 spike) | **P0:** notch lab_2 winner; optional for lab_5 |
| EEG1 low-freq fraction (0.5–2 / 0.5–30 Hz) | HQ median **0.017** | **0.112** | **0.176** | lab_2 is high-freq heavy → wider EEG BP helps |
| EMG awake/NREM RMS ratio | HQ median **3.73** | **2.69** | **2.58** | Usable atonia cue; wide EMG helped lab_2 |
| Amplitude CV (raw) | EEG CV high, frontal noisy | moderate | moderate | postnorm helps lab_3/5, **hurts lab_2** |
| Stage mix | ~48% Awake, ~46% NREM, ~6% REM | similar | similar | NMI gaps not class-imbalance |
| Label churn (changes/epochs) | lower | lowest | **highest** | lab_5 5 s scoring blur — acquisition limit |

Artifacts: [`results/data_exploration/hq_review/`](../data_exploration/hq_review/) (`hq_50hz_ratio_boxplot.png`, `hq_eeg1_lowfreq_fraction.png`, etc.).

---

## What ablations already settled

| Idea | lab_2 | lab_3 | lab_5 |
|------|-------|-------|-------|
| `post_normalize: false` | **win** | lose | lose |
| EEG 0–30 Hz | **win** | not needed | — |
| EEG 0–25 Hz (bp25) | worse | below baseline | below baseline |
| paper_robust (median/IQR + 0.3–35) | **rejected** | rejected | rejected |
| EMG 3–100 Hz | **win** | not tested | **P1 candidate** |
| EEG1+EEG4 montage | **win** | N/A | N/A |
| `wide_mlp` | marginal | **essential** | small gain |
| `append_channel_rms` | no gain | — | — |

Details: [ablation_lab2_findings_20260607.md](ablations/ablation_lab2_findings_20260607.md), [ablation_prepro_lab2_lab5.md](ablations/ablation_prepro_lab2_lab5.md), [ablation_lab2_rem.md](ablations/ablation_lab2_rem.md).

---

## Prioritized ablation backlog

### P0 — lab_2: 50 Hz notch (implemented, ready to submit)

- **Config knob:** `cvae.notch_freqs: [50]` → zeros **49.5–50.5 Hz** in rFFT ([`vae_preprocessing.py`](../../src/preprocessing/vae_preprocessing.py)).
- **Variant:** `rem_emg_wide_eeg4_notch50` — generator [`generate_ablation_lab2_notch.py`](../../scripts/cv4fold/generate_ablation_lab2_notch.py).
- **Gate:** best-of-3 NMI **> 0.593**, stable seeds.

```bash
PYTHONPATH=. python3 scripts/cv4fold/generate_ablation_lab2_notch.py
bash hpc/submit/cv4fold/submit_ablation_lab2_notch.sh
```

Log: `hpc/output/cv4fold/ablation_rem/lab_2/notch50_%J.out`  
Results: `results/cv4fold/ablation_rem/lab_2/abl_lab_2_rem_emg_wide_eeg4_notch50_*`

### P1 — lab_5: EMG wide + optional notch

- **`wide_mlp_notch50`:** locked `wide_mlp` + `notch_freqs: [50]` only (EMG **5–60 Hz** unchanged).
- **`wide_mlp_emg_wide`:** EMG **3–100 Hz**, keep postnorm + EEG 0–20 Hz.
- **`wide_mlp_emg_wide_notch50`:** above + `notch_freqs: [50]`.
- Generator: [`generate_ablation_lab5_emg.py`](../../scripts/cv4fold/generate_ablation_lab5_emg.py).

```bash
PYTHONPATH=. python3 scripts/cv4fold/generate_ablation_lab5_emg.py
bash hpc/submit/cv4fold/submit_ablation_lab5_emg.sh
```

Log: `hpc/output/cv4fold/ablation_emg/lab_5/lab_5_wide_mlp_emg_wide_%J.out`

### P2 — lab_3: EMG wide (low urgency)

One YAML on `wide_mlp` if P0/P1 complete and holdout not time-critical. Skip if lab_3 holdout already passes.

### P3 — Secondary (only if P0–P2 plateau)

| Knob | When |
|------|------|
| `band_pass_filter_type: time_domain` | Notch + freq BP insufficient for lab_2 broadband noise |
| `percentile_clip_channels: true` | Spike artifacts after notch |
| `perform_hanning_window: true` | Low priority |
| Exclude sub-072 from HQ | **Diagnostic** only — tests bad-mouse ceiling |

### Do not repeat

- paper_robust, bp25 / no_postnorm_bp25, global single recipe
- LR / β / `no_beta_epochs` retune (unless `no_beta_epochs_0` ablation surprises)
- Architecture sweeps on lab_2

---

## Not fixable by preprocessing alone

- **lab_5** 5 s sliding-window scoring → fuzzier transitions; holdout NMI is the honest test.
- **lab_2** GCaMP/fiber hardware → partial mitigation (EEG4, notch), not removal.
- **lab_3** mixed sex → report stratified holdout metrics; do not filter by default.
- **lab_5** small N (6 mice) → high variance; ceiling may stay ~0.55–0.60.

---

## Suggested execution order

1. Submit **P0** notch ablation (user confirms `bsub`).
2. **Per-lab holdout pilot** with locked recipes ([per_lab_holdout_pilot.md](per_lab_holdout_pilot.md)).
3. **P1** lab_5 EMG if holdout shows lab_5 still weakest.
4. Joint cross-lab cv4fold only after holdout gate ([cross_lab_cv4fold.md](cross_lab_cv4fold.md)).

---

## Changelog

- **2026-06-07** — Initial review; HQ plots; `notch_freqs` knob; lab_2/lab_5 ablation generators; manifest EEG4 lock.
