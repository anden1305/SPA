# Cross-lab preprocessing audit — main findings

> **Biological interpretation:** see [preprocessing_audit_biology.md](preprocessing_audit_biology.md).

Condensed readout of `results/preprocessing_audit/` (cv4fold quality cohort: labs 2, 3, 5; baseline = `cgmvae_base.yaml` FFT pipeline). Full artifacts: [results/preprocessing_audit/README.md](../results/preprocessing_audit/README.md).

**Regenerate:** `PYTHONPATH=. python -m scripts.preprocessing_audit.run --manifest data/manifests/cv_quality_cohort_v1.yaml --config src/config/run/cvaeprior/cv4fold/templates/cgmvae_base.yaml --out results/preprocessing_audit`

---

## Headline

**Current cv4fold preprocessing does not yield clearly stage-separated VAE inputs in any lab** (median silhouette &lt; 0.15 everywhere). **Lab 5 is weakest**; lab 3 is **not** uniquely “good” on features alone—so worse validation NMI on labs 2/5 is **not fully explained** by a simple “lab 3 preprocessing works, others don’t” story. Next levers: **(1)** targeted preprocessing experiments (especially lab 5), **(2)** lab/subject conditioning, **(3)** re-link to NMI once cv4fold `per_mouse` metrics exist.

---

## 1. What the model actually sees (Phase 2)

After `VAEPreprocessing` + `sequence_length=64`, inputs are `(n_seq, 64, 3, F_fft)` log-power spectra per run.

| Lab | Median silhouette† | Median sequences/run | Generalize from lab-3-like inputs? |
|-----|-------------------|----------------------|-----------------------------------|
| lab_2 | **0.110** | 84 | Uncertain / no |
| lab_3 | **0.125** | 84 | Uncertain / no |
| lab_5 | **0.069** | **168** | **No** |

†Silhouette on flattened features, majority stage label per sequence (proxy for “can k-means see stages without VAE?”). Success bar from plan: **≥ 0.15** — **no lab passes**.

**Per-mouse spread:** lab_5 silhouettes stay low (median **0.08**, max **~0.12**). lab_3 has the best single runs (up to **~0.21**) but cohort median is still modest. lab_2 per-mouse median is slightly **above** lab_3 (**0.13** vs **0.13**) when aggregated by mouse—so good lab-3 **NMI** is unlikely to be explained only by “better FFT features in lab 3.”

**Plots to open first:** `after/lab_3/model_input_lab_contrast.png`, `after/lab_*/feature_pca_triple.png`, `after/lab_*/fft_logpower_by_stage.png`.

---

## 2. Raw data before preprocessing (Phase 1)

| Lab | Median %NREM | Median %REM | %Artifact (labeled) |
|-----|--------------|-------------|---------------------|
| lab_2 | 47.8 | 6.2 | **N/A** (3-class scoring) |
| lab_3 | 39.5 | 7.1 | ~0.04 |
| lab_5 | 50.1 | 7.3 | ~0 |

- **REM/NREM balance** is broadly similar; lab_3 has somewhat **less NREM** (more wake/fragmentation in labels).
- **Artifact handling is asymmetric:** `remove_artifact: true` in the manifest **strips Artifact only on lab_3/5**; on lab_2 it is a **no-op** (no Artifact class). Same YAML ≠ same effective data.
- **Montage:** lab_2 uses **EEG1 + EEG3** (both parietal); lab_3/5 use **EEG1 (P) + EEG2 (F)**. Band-pass is applied by **channel index** `[[≤20 Hz], [≤20 Hz], [5–60 Hz EMG]]`, so the **same numbers hit different anatomy** in lab_2 vs 3/5.

**Plots:** `raw/lab_*/label_composition.png`, `raw/lab_*/psd_by_stage.png`, `raw/lab_*/band_power_by_lab.png`.

---

## 3. Normalization scope (partner question)

**Answer: not lab-level.**

| Step | Scope |
|------|--------|
| Windowing, FFT, band-pass | Per **mouse × run** |
| `post_normalize` | Per **mouse × run** (mean/std over all sequences in that recording) |
| `normalize_global` | Off in cv4fold |

**Run 1 vs run 2:** post-norm scale (`postnorm_std_mean`) differs between runs on the same mouse (e.g. lab_2 ~0.95 vs ~0.90; lab_5 run 3 lower ~0.76). Training pools both runs—**run-level norm can inject run-specific scaling**.

**Counterfactual:** lab-pooled vs run-level post-norm shifts features non-trivially; see `after/normalization_scope/counterfactual_lab_vs_run_norm.png` and `tables/normalization_scope.csv`.

---

## 4. Preprocessing ablations (Phase 3, 2 mice/lab, run 1)

Median silhouette — **subset only**, not full cohort:

| Variant | lab_2 | lab_3 | lab_5 |
|---------|-------|-------|-------|
| baseline | 0.056 | 0.109 | **−0.018** |
| no_post_normalize | 0.059 | **0.121** | −0.040 |
| +percentile_clip | 0.057 | 0.112 | −0.005 |
| bp_time_domain | 0.031 | 0.076 | −0.006 |
| bp_placement_aware | 0.056 | 0.109 | −0.018 |

**Takeaways:**

- **`no_post_normalize`** helps lab_3 on the subset but **hurts lab_5** — not a free global fix.
- **`+percentile_clip`** is the least bad for lab_5 on the subset (still negative).
- **Placement-aware / global_norm / artifact_harmonized** — no meaningful gain on this subset.
- **Do not change production YAML** from ablations alone; confirm on full cohort if you adopt a variant.

Detail: `deltas/SUMMARY_deltas.md`.

---

## 5. Linkage to validation NMI (Phase 4)

**No `results/cv4fold/.../per_mouse/metrics.json` found** at audit time — scatter of silhouette vs NMI was **not** computed. Re-run phase 4 after cv4fold training:

```bash
PYTHONPATH=. python -m scripts.preprocessing_audit.run --phases 4 5 --out results/preprocessing_audit
```

Until then: if lab-3 NMI is good but lab-5 NMI is poor, and lab-5 silhouette is worst here, suspect **both** weaker features **and** model/conditioning—not preprocessing alone.

---

## 6. Recommended actions (priority order)

1. **Re-run linkage** after cv4fold jobs finish (`linkage/nmi_vs_silhouette_scatter.png`).
2. **Smoke80 on lab_5** (worst feature separability): `bash hpc/submit/cv4fold/submit_smoke_80.sh` with per-lab scope.
3. **If NMI gap remains with similar silhouettes:** prioritize **`chmmgmvae`** with `conditioning_source: subject_lab` ([conditioning_story_map.md](conditioning_story_map.md)) before another large preprocessing sweep.
4. **If adopting preprocessing changes:** trial **`+percentile_clip`** for lab_5 and/or **`no_post_normalize`** for lab_3 on **full cohort**; document artifact policy for lab_2.
5. **Report to partner:** preprocessing is **run-level**; plots of model inputs are under `after/`; lab_3 advantage is **not** obvious in pre-VAE separability alone.

---

## 7. Where things live

| Question | Location |
|----------|----------|
| Executive go/no-go | `results/preprocessing_audit/00_INDEX.md` |
| Raw vs FFT narrative | `raw/SUMMARY_raw.md`, `after/SUMMARY_after.md` |
| Ablation table | `deltas/SUMMARY_deltas.md` |
| Per-run metrics | `tables/feature_stats_by_lab.csv`, `tables/per_mouse_summary.csv` |
| Locked config snapshot | `config/cv4fold_fft_baseline.yaml` |
| Manifest notes | `data/manifests/cv_quality_cohort_v1.yaml` → `preprocessing_audit:` block |

---

*Generated from audit run on cv_quality_cohort_v1; silhouettes are diagnostics, not validation NMI.*
