# Cross-lab preprocessing audit — biological interpretation

Thesis-oriented readout of what the MSSV sleep recordings and labels imply **biologically**, grounded in plots under `results/preprocessing_audit/`. For numeric gates and go/no-go, see [preprocessing_audit_findings.md](preprocessing_audit_findings.md). For next technical steps, see [preprocessing_audit_recommendations.md](preprocessing_audit_recommendations.md).

**Regenerate audit:** `PYTHONPATH=. python -m scripts.preprocessing_audit.run --manifest data/manifests/cv_quality_cohort_v1.yaml --config src/config/run/cvaeprior/cv4fold/templates/cgmvae_base.yaml --out results/preprocessing_audit`

---

## 1. Scope and caveats

- **Cohort:** quality mice in **lab_2, lab_3, lab_5** ([`data/manifests/cv_quality_cohort_v1.yaml`](../data/manifests/cv_quality_cohort_v1.yaml)); lab_1 excluded.
- **Raw plots** (`results/preprocessing_audit/raw/`) load `.npy` via [`scripts/data_exploration/helpers.py`](../scripts/data_exploration/helpers.py) with **0.5–30 Hz bandpass + per-run z-score**. That path is **not identical** to the cv4fold VAE pipeline (FFT, freq-domain band-pass, `post_normalize` on sequences). Use raw figures for **cohort-level physiology and labels**; use `after/` for **what the VAE actually sees**.
- **Artifacts:** lab_3 and lab_5 have an Artifact label in metadata; this cohort shows **~0% Artifact** after epoch aggregation. **lab_2 has no Artifact class** — `remove_artifact: true` in training is a no-op there, while it strips labeled Artifact epochs in lab_3/5.
- **Silhouette** (below) is a linear separability proxy on FFT features, not validation NMI.

---

## 2. Sleep architecture across labs

**Sources:** `raw/lab_*/label_composition.png`, `tables/per_mouse_summary.csv`

| Lab | Median %NREM | Median %REM | Median %Artifact |
|-----|--------------|-------------|------------------|
| lab_2 | 47.8 | 6.2 | (no label) |
| lab_3 | 39.5 | 7.1 | ~0.04 |
| lab_5 | 50.1 | 7.3 | ~0.0 |

**Interpretation**

- **REM is sparse (~6–7%)** across labs. Unsupervised models see mostly NREM vs Wake with few REM examples — expect REM to be the hardest stage regardless of lab.
- **lab_3 has the lowest %NREM** (~40% vs ~48–50% elsewhere). That usually means **more Wake or more fragmented scoring** in that site’s protocol, not necessarily “less sleep” in a physiological sense.
- **Similar REM fractions** across labs mean **stage balance alone does not explain** why validation NMI is often best in lab_3 and weaker in lab_2/5.

---

## 3. Spectral physiology (raw EEG1)

**Sources:** `raw/lab_*/band_power_by_lab.png` (same cross-lab bar chart in each lab folder), `raw/lab_*/psd_by_stage.png` (example mice per lab)

| Observation | Biological reading |
|-------------|-------------------|
| **NREM theta > Awake** on EEG1 | Consistent with **mouse NREM**: elevated hippocampal/cortical theta (4–8 Hz) during NREM vs quiet wake. |
| **lab_2 highest theta** in Awake and NREM | Likely **montage**, not “deeper sleep”: lab_2 uses **EEG1 + EEG3 (both parietal)**; lab_3/5 use **parietal + frontal**. Parietal theta power can read higher than frontal depending on state. |
| **lab_5 elevated REM theta** vs lab_3 | REM in mice combines **frontal theta** and **EMG/atonia** cues. Higher REM theta at lab_5 may reflect **scoring**, **frontal electrode placement**, or real REM physiology — compare **EMG channel** before attributing to preprocessing. |
| **Sharp PSD roll-off near 25–30 Hz** in raw PSD plots | Dominated by the **exploration loader’s 30 Hz low-pass**, not the VAE’s 20 Hz EEG band-pass in frequency domain. Do not treat that cut-off as evidence about the trained model’s band limits. |

Within lab_3, example PSDs overlap strongly across mice — **within-lab acquisition is consistent**; cross-lab differences are the main concern.

---

## 4. Montage and EMG (cross-lab comparability)

| Lab | Channels (cv4fold) | Anatomy | Band-pass (by index) |
|-----|-------------------|---------|---------------------|
| lab_2 | EEG1, EEG3, EMG | P + P + EMG | ≤20 Hz, ≤20 Hz, 5–60 Hz |
| lab_3 | EEG1, EEG2, EMG | P + F + EMG | ≤20 Hz, ≤20 Hz, 5–60 Hz |
| lab_5 | EEG1, EEG2, EMG | P + F + EMG | ≤20 Hz, ≤20 Hz, 5–60 Hz |

**EEG:** Applying the **same numeric filters to channel indices** means lab_2 applies **parietal-style limits twice**; lab_3/5 apply parietal + frontal limits. That is appropriate only if you treat indices as “EEG vs EMG”, not as “P vs F”.

**EMG:** Channel 2 is the main **Wake vs REM** separator in mice (muscle tone). Cross-lab differences in REM scoring or EMG contamination in Wake will show up in model inputs even when EEG spectra look plausible.

**Takeaway:** Cross-lab comparison is a **montage + scoring** problem as much as a machine-learning problem. Harmonizing artifact policy (lab_2 has no Artifact epoch class) is a **protocol/science** decision, not a trivial YAML tweak.

---

## 5. Model-input features (after preprocessing)

**Sources:** `after/lab_*/fft_logpower_by_stage.png`, `after/lab_3/model_input_lab_contrast.png`, `after/lab_*/feature_pca_triple.png` (stage-colored, per lab), `after/cross_lab/pca_all_labs_unlabeled.png` (lab-colored mix)

| Observation | Biological / modelling reading |
|-------------|----------------------------------|
| **Median silhouette &lt; 0.15** (all labs) | Wake / NREM / REM are **not linearly separable** in hand-crafted log-power FFT features alone. The VAE and temporal prior must **learn** discriminative representations; good NMI is not guaranteed from spectra alone. |
| **lab_5 lowest silhouette** (~0.07 vs ~0.11–0.13) | Less stage structure in the **exact tensors fed to the encoder** — consistent with risk of worse holdout NMI at lab_5 even when lab_3 looks acceptable. |
| **Cross-lab PCA: separate manifolds per lab** | **Lab batch effects** (montage, amplitude, scoring) remain in model inputs after run-level post-normalization. Supports **lab + subject conditioning** and reporting **per-lab NMI**, not only pooled metrics. |
| **Flat high bands in input heatmaps** | Mostly **zeroed frequency bins** after band-pass in the FFT pipeline, not extended high-frequency biology. |
| **Heatmap contrast (lab_3 vs lab_2/5)** | Same sequence layout, different log-power texture and baseline level on channel 0 — visual confirmation that **“lab 3 preprocessing” is not producing lab-3-shaped inputs for other labs**. |

**Per-lab stage PCA** (`after/lab_*/feature_pca_triple.png`): points colored by majority stage (Awake / NREM / REM) within that lab. Overlap between stages is large everywhere — biology does not present three clean clusters in FFT space.

---

## 6. Normalization (biological angle)

**Source:** `after/normalization_scope/postnorm_scale_by_run.png`

Training uses **`post_normalize` per mouse × run** (statistics over all sequences in that recording), **not per lab**.

Biologically, **run 1 vs run 2** can differ in electrode impedance, animal activity before sleep, or depth/quality of the night — so run-level scaling can preserve **real within-subject run differences**. It does **not** remove **between-lab** differences in spectrum shape or montage, which helps explain **persistent lab clusters** in feature-space PCA.

The audit plot shows **pre–post-normalization std** used for z-scoring (not std after norm, which should be ~1 within each run).

---

## 7. Problem mice (link to tuning)

From [cv4fold tune sweep results](cv4fold/tune_sweep_bayes50_results.md) and the audit cohort:

| Mouse | Lab | Note |
|-------|-----|------|
| **sub-072** | lab_2 | Often weak NMI in tune; check signal quality and scoring, not only model capacity. |
| **sub-087** | lab_5 | Very long recordings; **dominates token count** in joint training/validation. Pooled NMI can look acceptable while **lab_5 per-mouse performance** stays poor. |
| **sub-059** | lab_3 | Historically unstable (high tune NMI on some runs, very low on holdout smoke) — biology + split luck, not only lab identity. |

When interpreting “lab 3 is best”, separate **site effects** from **which mice** are in the validation fold.

---

## 8. Thesis-ready summary bullets

1. **Montage differs by lab** (double parietal vs parietal+frontal); identical band-pass indices do not mean identical physiology across sites.
2. **Theta–NREM relationships** match mouse sleep biology within labs, but **absolute band power offsets** differ across labs.
3. **REM is rare and EMG-relevant**; lab_5 shows distinct REM theta patterns vs lab_3 — scoring and frontal EEG matter.
4. **FFT model inputs** show **limited linear stage separability** in every lab; the VAE must do most discriminative work.
5. **Lab structure in feature space** rivals stage structure → report **per-lab / per-mouse NMI** and consider **subject + lab conditioning**.
6. **Run-level normalization** preserves run-specific amplitude regimes when pooling runs for training.
7. **Artifact handling is asymmetric** across labs (label availability), so cross-lab “fairness” of training data should be discussed explicitly in methods.

---

## 9. Figure index

| Figure | Path | Use in thesis |
|--------|------|----------------|
| Stage balance per mouse | `raw/lab_*/label_composition.png` | Cohort label distribution |
| Theta by stage & lab (EEG1) | `raw/lab_*/band_power_by_lab.png` | Cross-lab spectral offset |
| Example PSD | `raw/lab_*/psd_by_stage.png` | Within-lab spectral consistency |
| Log-power vs stage (model path) | `after/lab_*/fft_logpower_by_stage.png` | Post-pipeline spectral biology |
| Cross-lab input texture | `after/lab_3/model_input_lab_contrast.png` | Visual lab mismatch on VAE inputs |
| Stage PCA (per lab) | `after/lab_*/feature_pca_triple.png` | Separability without VAE |
| Lab PCA (all labs) | `after/cross_lab/pca_all_labs_unlabeled.png` | Batch / lab manifold |
| Run-level norm stats | `after/normalization_scope/postnorm_scale_by_run.png` | Methods: normalization scope |
| Counterfactual lab pool norm | `after/normalization_scope/counterfactual_lab_vs_run_norm.png` | Sensitivity to norm scope |
| Sequence counts | `after/lab_*/window_count_diagnostics.png` | Data volume fairness |

---

*Audit date referenced in manifest: 2026-06-03. Silhouette and PCA are exploratory; validation NMI from training remains the primary performance metric.*
