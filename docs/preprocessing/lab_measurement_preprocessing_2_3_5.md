# How labs 2, 3, 5 are measured & preprocessed

**Added 2026-06-06.** Acquisition and preprocessing differences between the labs we
train on, pulled from the data paper, the OpenNeuro dataset files, and our own
metadata. Motivation: explain why per-lab cGMVAE behaves differently (esp. the
lab_2 `post_normalize` finding in [`cv4fold/ablations/ablation_prepro_lab2_lab5.md`](../cv4fold/ablations/ablation_prepro_lab2_lab5.md)).

## Sources

- **Paper:** Rose et al., *Probability estimation of narcolepsy type 1 in DTA mice using unlabeled EEG and EMG data*, Sleep Advances 2025, [zpaf025](https://doi.org/10.1093/sleepadvances/zpaf025) — Table 1 + per-cohort acquisition + "Data preprocessing".
- **Dataset:** OpenNeuro [ds006366 v1.0.1](https://openneuro.org/datasets/ds006366/versions/1.0.1) (MSSV). Local mirror: `data/ds006366/README.md`, `dataset_description.json`, `data/ds006366_processed/metadata.csv`.
- The MSSV OpenNeuro release = the **healthy (WT) training cohorts A–E** of the paper = `lab_1`…`lab_5` (92 mice). Cohorts F–H (NT1/DTA, labs 3/6/7) are **not** in this dataset.

## Cohort ↔ lab mapping (our data)

| Lab | Paper cohort | Site | Mice (our metadata) | EEG montage | EMG |
|-----|--------------|------|---------------------|-------------|-----|
| lab_1 | A | Bologna, IT | 10 | 1× ipsilateral fronto-parietal **differential** (`EEG IFPD`) | neck |
| **lab_2** | **B** | **Bern, CH** | **17** | **2 parietal + 2 frontal** (EEG1/2=P, EEG3/4=F) | neck |
| **lab_3** | **C** | **Copenhagen, DK** | **32** | **1 parietal + 1 frontal** (EEG1=P, EEG2=F) | neck |
| lab_4 | D | Copenhagen, DK | 27 | 1× prefrontal-cortex-frontal (`EEG PFCF`) / cerebellum variants | neck |
| **lab_5** | **E** | **Lyon, FR** | **6** | **1 parietal + 1 frontal** (EEG1=P, EEG2=F) | neck |

(Electrode `*_TYPE` strings are from `metadata.csv`: `EEG P` = parietal, `EEG F` = frontal.)

## The three labs we train on

### lab_2 (cohort B — Bern)
- **17 WT male** mice, C57BL/6JRj (Janvier Labs), 6–15 weeks.
- **4 EEG electrodes (2 parietal + 2 frontal)** + neck EMG — the richest montage of the three.
- Mice carried a **GCaMP6 viral injection + optical fiber** above the lateral hypothalamus for calcium imaging (unused here, but extra implanted hardware can add impedance/movement artifact to the EEG).
- **Scored in 1-second epochs** with custom MATLAB scripts → later re-binned to 4 s.
- **No "Artifact" stage** scored (3 classes: Wake/NREM/REM). In our loader `n_stages=3` for lab_2.

### lab_3 (cohort C — Copenhagen) — our "home"/reference lab
- **23 WT** mice in the WT cohort (12 female); `metadata.csv` shows 32 mice/53 runs under `lab_3`.
- **1 parietal + 1 frontal** EEG + neck EMG.
- **Scored natively in 4-second epochs** by standard criteria — matches the model's epoch length with no resampling of labels.
- **"Artifact" stage scored** (4 classes), which we drop via `remove_artifact: True`.
- Longest recordings (mean ≈ 63 k s/run). This is the lab our baseline preprocessing was implicitly tuned on.

### lab_5 (cohort E — Lyon)
- **5–6 WT male** mice (paper says 5; our metadata has 6), C57BL/6J.
- **1 parietal + 1 frontal** EEG + neck EMG (same montage as lab_3).
- **Scored with a 5-second sliding window**, then upsampled to 1 s and re-binned to 4 s — labels pass through the most resampling, which can blur transitions.
- **"Artifact" stage scored** (4 classes), dropped via `remove_artifact: True`.

## Shared preprocessing in the paper (uniform across all labs)

The paper applied **one** pipeline to every lab (no per-lab tuning):

1. Resample all EEG/EMG to **128 Hz** (polyphase). *(Already done in our `*_processed` data.)*
2. **Per-channel robust scaling**: median 0, IQR 1.
3. **Artifact clip**: samples > 20× IQR from the channel median are clipped.
4. **Bandpass 0.3–35 Hz**.
5. Relabel to 3 classes (Wake/NREM/REM); convert all scorings to **4-second epochs** (majority vote for finer scorings; up- then down-sample for 5 s).

## What differs across labs (and why it matters for us)

| Axis | lab_2 | lab_3 | lab_5 | Modeling implication |
|------|-------|-------|-------|----------------------|
| EEG electrodes | 2P + 2F | 1P + 1F | 1P + 1F | lab_2 has redundant pairs; our manifest picks **EEG1(P)+EEG3(F)** to match the P+F montage of lab_3/5 |
| Scoring resolution | 1 s | 4 s (native) | 5 s sliding | lab_5 labels most resampled → fuzzier transitions |
| Artifact stage | none (3) | yes→removed | yes→removed | different label provenance per lab |
| Extra hardware | GCaMP6 + fiber | none | none | possible lab_2 EEG artifact |
| Sex | all male | mixed (12 F) | all male | population shift |
| Recording length | ~21 k s | ~63 k s | ~43 k s | data volume per mouse |

### Key takeaways for our cGMVAE work

- **Our pipeline ≠ the paper's pipeline.** We use EEG lowpass `[null,20]` + EMG `[5,60]` and a `post_normalize` step, vs the paper's uniform `[0.3–35 Hz]` + per-channel median/IQR scaling. There is no per-lab normalization in either the paper or our defaults.
- The **lab_2 channel montage is fine** — `EEG1(parietal)+EEG3(frontal)+EMG` is the correct analogue of lab_3/5's `EEG1(parietal)+EEG2(frontal)+EMG`. The earlier 5-channel failure was a config bug, not a montage problem.
- The strongest measurable lab_2 differences (1 s scoring, extra implanted hardware, no artifact stage, all-male, shorter recordings) are **acquisition-side**, not preprocessing. This is consistent with the ablation result: lab_2's gain came from **dropping `post_normalize`** (a normalization tuned on lab_3), not from changing channels — i.e. lab_2's amplitude statistics differ enough from lab_3 that lab_3-tuned normalization washed out its structure.
- **Next idea worth testing:** adopt the paper's **median/IQR per-channel scaling + 0.3–35 Hz bandpass** as a lab-agnostic front end, since it was explicitly designed to work uniformly across these labs.
