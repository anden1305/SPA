---
name: Biological insights markdown
overview: Create a standalone thesis-oriented markdown in docs/ synthesizing biological interpretations from the preprocessing audit plots (raw + model-input), separate from the training execution plan.
todos:
  - id: draft-biology-md
    content: Write docs/preprocessing_audit_biology.md with sections 1–9 and figure index
    status: completed
  - id: cross-link-docs
    content: Add links from findings.md and recommendations.md to biology doc
    status: completed
  - id: optional-rerun-plots
    content: After phase2 PCA fix, re-run audit phase 2 and update biology doc figure refs
    status: completed
isProject: false
---

# Biological insights documentation plan

## Goal

Add **[docs/preprocessing_audit_biology.md](docs/preprocessing_audit_biology.md)** — a readable synthesis of **what the EEG/EMG and sleep labels imply biologically**, grounded in `results/preprocessing_audit/` plots. This complements:

- [docs/preprocessing_audit_findings.md](docs/preprocessing_audit_findings.md) — numbers and go/no-go
- [docs/preprocessing_audit_recommendations.md](docs/preprocessing_audit_recommendations.md) — what to do next technically

No new analysis runs required unless you want updated figures after the phase2 PCA path fix.

---

## Document structure

### 1. Scope and caveats (short)

- Cohort: quality mice labs 2, 3, 5 ([cv_quality_cohort_v1.yaml](data/manifests/cv_quality_cohort_v1.yaml))
- Raw exploration plots use loader bandpass 0.5–30 Hz + per-run z-score ([helpers.py](scripts/data_exploration/helpers.py)) — **not identical** to VAE FFT pipeline; label clearly in doc
- lab_2: no Artifact class; lab_3/5: Artifact label exists but ~0% in this cohort

### 2. Sleep architecture across labs

Source: `raw/*/label_composition.png`, `tables/per_mouse_summary.csv`

| Finding | Biology / scoring |
|---------|-------------------|
| REM ~6–7%, NREM ~40–50% | Typical mouse overnight mix; REM rare → hard for unsupervised clustering |
| lab_3 lowest %NREM | More wake or fragmentation in labels, not necessarily less sleep |
| Similar REM across labs | Stage balance alone does not explain lab NMI gap |

### 3. Spectral physiology (raw EEG1)

Source: `raw/*/band_power_by_lab.png`, `raw/*/psd_by_stage.png`

| Finding | Interpretation |
|---------|----------------|
| NREM theta > Awake | Expected **hippocampal/cortical theta** in mouse NREM |
| lab_2 highest theta (Awake/NREM) | **Montage**: two parietal channels (EEG1+EEG3) vs P+F in lab_3/5 — not a pure “better sleep” signal |
| lab_5 elevated REM theta | REM scoring + **frontal theta**; compare with EMG before blaming EEG preprocessing |
| PSD roll-off ~25–30 Hz on raw plots | Loader filter artifact in exploration path; do not over-interpret high-frequency cut-off for VAE path |

### 4. Montage and EMG (cross-lab comparability)

| Lab | EEG montage | Implication |
|-----|-------------|-------------|
| lab_2 | EEG1 + EEG3 (P + P) | Duplicate spatial sampling; same band-pass index hits parietal twice |
| lab_3, lab_5 | EEG1 (P) + EEG2 (F) | Classic P/F dissociation for NREM vs REM |

**EMG** is shared as channel index 2 with 5–60 Hz band-pass — primary **REM vs Wake** discriminator in mice. Cross-lab REM disagreement may reflect **scoring** or **muscle tone differences**, not only VAE failure.

### 5. Model-input features (after preprocessing)

Source: `after/lab_*/fft_logpower_by_stage.png`, `after/lab_3/model_input_lab_contrast.png`, misplaced cross-lab PCA (see below)

| Finding | Interpretation |
|---------|----------------|
| Low stage separability (silhouette &lt; 0.15 all labs) | Hand-crafted FFT features are **weakly separable** for 3-class sleep; VAE + temporal prior must extract structure |
| lab_5 weakest separability | Less information per sequence in feature space — aligns with worse val NMI risk |
| Cross-lab PCA: labs form separate manifolds | **Batch/lab signal** in inputs; motivates `subject_lab` conditioning (training plan) |
| Heatmap flat high bands | Zeros from **frequency-domain band-pass** (masked bins), not biological silence |

**Figure note to include:** `after/lab_3/feature_pca_triple.png` currently shows **all labs unlabeled** (script bug); per-lab stage PCA is in the same folder only for lab_2/5 until re-run — cite `fft_logpower_by_stage` as primary per-lab biology figure.

### 6. Normalization (partner question, biological angle)

Source: `after/normalization_scope/postnorm_scale_by_run.png`

- Normalization is **per recording run**, not per lab — biology-relevant because run 1 vs run 2 can differ in depth of anesthesia, electrode impedance, or activity level
- Does not remove lab identity from spectra — explains persistent lab clusters in feature space

### 7. Problem mice called out in tune doc

Tie audit to [docs/cv4fold/tune_sweep_bayes50_results.md](docs/cv4fold/tune_sweep_bayes50_results.md):

- **sub-072** (lab_2): often weak — check scoring / signal quality, not only model
- **sub-087** (lab_5): dominates token mass in pooled metrics — pooled NMI can look “OK” while lab_5 biology is underrepresented in optimization

### 8. Thesis-ready bullet summary

End with 5–7 bullets suitable for methods/discussion:

1. Cross-lab montage differences limit fair spectral comparison by channel index alone.
2. Theta–NREM relationships hold within lab but absolute band power differs across sites.
3. REM is sparse and EMG-driven; lab_5 REM theta patterns differ from lab_3.
4. FFT model inputs show limited linear separability of stages in all labs.
5. Lab structure in feature space is as large as stage structure → conditioning or per-lab reporting required.
6. Preprocessing choices (run-level post-norm) preserve run-specific amplitude regimes.

### 9. Figure index table

| Figure path | Use in thesis for |
|-------------|-------------------|
| `raw/lab_*/label_composition.png` | Cohort stage balance |
| `raw/lab_*/band_power_by_lab.png` | Cross-lab theta by stage (EEG1) |
| `after/lab_*/fft_logpower_by_stage.png` | Post-pipeline spectral structure |
| `after/lab_3/model_input_lab_contrast.png` | Visual cross-lab input mismatch |
| `after/normalization_scope/postnorm_scale_by_run.png` | Run-level scaling (methods) |

---

## Implementation effort

- **~1–2 hours writing** from existing summaries + plot review (no code required for v1)
- Optional: re-run `python -m scripts.preprocessing_audit.run --phases 2` after PCA path fix to refresh `feature_pca_triple.png` per lab and add `after/cross_lab/pca_all_labs.png`

---

## Link from existing docs

Add one line at the top of [docs/preprocessing_audit_findings.md](docs/preprocessing_audit_findings.md) and [docs/preprocessing_audit_recommendations.md](docs/preprocessing_audit_recommendations.md):

> Biological interpretation: see [preprocessing_audit_biology.md](preprocessing_audit_biology.md).
