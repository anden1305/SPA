# Report improvement plan — `plos_my_paper_report.pdf`

**Added 2026-06-10.** Multi-angle review of the current special-course report (`plos_my_paper.tex` + embedded figures), compared to QSLP-AE (colleague MSSV paper), Rose 2026 framing, mcRBM/Somnotate/SPINDLE precedents, and PLOS Comp Biol norms.

**Current snapshot:** ~2,440 main-text words (`texcount`); K-sweep **complete** (K=3–15); main Figs 1–5 + SI S1–S4 generated via `scripts/paper/build_curated_figures.py`.

---

## 1. Executive assessment

### What already works (keep)


| Strength                       | Evidence                                                                          |
| ------------------------------ | --------------------------------------------------------------------------------- |
| **Clear zero-shot claim**      | Methods `\methodpara{Holdout inference}` + Fig 1B; contrast with AccuSleep stated |
| **QSLP-style structure**       | Data → Methods (Fig 1 inside) → Results and discussion → Appendix                 |
| **Quantitative ladder result** | cHMM 0.56 vs cGMVAE 0.33, Δ=+0.22 on every fold (Table 1)                         |
| **Honest heterogeneity**       | Lab 2 ≈ parity; labs 3/5 gain from temporal prior (Fig 3)                         |
| **K-sweep with dual metrics**  | NMI + log p(z₁:T) in `k_sweep_dual_axis.pdf`; seed ranges in CSV                  |
| **Reproducibility hooks**      | Locked recipes, manifest paths, `build_curated_figures.py` pipeline               |
| **Limitations inline**         | Label noise, seed fragility, thesis LOSO contrast — not buried                    |


### What blocks “high quality” today


| Risk                             | Severity     | Why reviewers / examiners will push back                                                    |
| -------------------------------- | ------------ | ------------------------------------------------------------------------------------------- |
| **“92 mice” vs 20-mouse cohort** | **Critical** | Intro/Data imply full MSSV training; actual cv4fold uses **20 HQ mice** only                |
| **K=4 substage story is thin**   | **High**     | Best NMI at K=4 (0.643) ≈ macro+1 transition state; not thesis-scale substaging             |
| **Incomplete model ladder**      | **High**     | hmm_raw / HMMGMVAE missing from Fig 2/S1; abstract promises ladder but Results show 2 rungs |
| **Appendix placeholders**        | **Medium**   | S1 Table, S3 Table, S6 Fig, S2/S3 appendices are labels without content                     |
| **No expert substage names**     | **Medium**   | Fig 4 “provisional” names; Birgitte interview not reflected                                 |
| **Single metric (NMI)**          | **Medium**   | Field reports κ / accuracy; Rose 2026 discusses agreement, not NMI                          |
| **Word budget under target**     | **Low**      | ~2,440 vs 3,000–3,500 target — room to add *interpretation*, not Methods                    |


---

## 2. Literature: what strong papers do that we should copy

### Rose / MSSV ecosystem

- **Separate dataset scale from analysis cohort** — “MSSV comprises 92 mice…” then “we analyse a quality subset of *n* recordings…”
- **Per-site and per-mouse breakdown** — never trust pooled NMI alone (we have S4; need one sentence per difficult mouse in main text)
- **Supervised ceiling in one paragraph** — SPINDLE/REST achieve high κ when retrained multi-lab; unsupervised NMI is not comparable but frames expectations

### AccuSleep / PLOS One 2024 architecture comparison

- **Two validation frameworks named explicitly:** within-subject vs cross-subject (maps to our within-lab vs joint)
- **Report κ alongside accuracy** — add macro-averaged κ or per-class F1 for fold-4 exemplar in SI (even if NMI stays primary)

### mcRBM (Katsageorgiou, *PLOS Biology*)

- **Rich physiology grid in main or one SI figure readers actually open** — 7-column panel (PSD, θ/δ, bout, subject mix) is the substage payoff
- **Name states from biology** — not “S1 (42%)”; use Wake-active / NREM-deep / transition / REM after expert pass

### Somnotate (*PLOS Comp Biol*)

- **Transition states as first-class results** — our Fig 5 is correct placement; main text should cite one concrete transition probability (e.g. Wake→transition→NREM) with number from CSV

### SPINDLE (*PLOS Comp Biol*)

- **End-to-end diagram of generalisation** — Fig 1 already good; add one sentence: “training mice never appear in validation folds” (leakage check)

### Switching state-space (*PLOS Comp Biol* 2023)

- **Synthetic sanity check before real data** — one sentence in main + full SI appendix (thesis Ch. 3) proves HMM prior behaves

### QSLP-AE (Ciudad et al., colleague)

- **Contrast in Discussion, don’t conflate** — cross-species alignment vs our cross-lab zero-shot; cite as complementary MSSV work
- **Numbers-first Data section** — we do this; fix the 20 vs 92 count

---

## 3. Scientific narrative fixes (highest impact)

### 3.1 Reframe the cohort (do first)

**Problem:** Line 280 says “decoder-only conditioning on **92 mice**”; Data § says 92 mice across five labs but only analyses labs 2/3/5.

**Fix (template):**

> MSSV comprises 92 mice across five laboratories. Following quality screening (thesis §2.2.8 extended to labs 2 and 5), we analyse a **20-mouse cohort** (6+10+4 mice in labs 2, 3, and 5) with four-fold leave-mice-out cross-validation (`cv_quality_cohort_v1.yaml`).

Also update `methods_writing_guide.md` table (“20 HQ mice, not 92”).

### 3.2 Reframe K=4 vs substage programme

**Data fact:** K=3 mean NMI 0.616; K=4 mean 0.617; K=5 0.581; sharp drop at K≥6 for mean NMI. K=4 best seed uses **4 predicted states** — essentially macro + one transition bucket.

**Options (pick one for main text):**


| Strategy                                    | Main claim                                                       | Fig 4–5 K  | SI                                              |
| ------------------------------------------- | ---------------------------------------------------------------- | ---------- | ----------------------------------------------- |
| **A. Honest macro+ (recommended for exam)** | “Resolution sweep peaks at K=4; finer K over-segments holdout”   | K=4        | K=7, K=12 biology folders for thesis comparison |
| **B. Biology-first**                        | “We show substages at K=7 (thesis-aligned) with modest NMI cost” | K=7        | K=4 in SI as selection curve peak               |
| **C. Two-tier**                             | Macro ladder K=3; substage illustration at K=7                   | K=7 panels | K-sweep explains why K=7 not K=15               |


**Recommended:** **A for Results text**, bring `professor_meeting/K07/` panels to SI as “finer resolution under lower holdout NMI”.

### 3.3 Complete the ladder story

Minimum for credibility:

1. **HMMGMVAE** locked holdout — 4 joint folds (even if weak)
2. **hmm_raw** fold-4 smoke or SI-only — show ~0 NMI population (thesis Table 9 narrative with a number)
3. **Fig 2 / S1** — four bars when data exist; until then, **remove “four-rung ladder” from abstract** or say “primary comparison cGMVAE vs cHMM–GMVAE; extended rungs in SI”

### 3.4 Metrics package


| Metric                         | Where                  | Purpose                             |
| ------------------------------ | ---------------------- | ----------------------------------- |
| Prior NMI (primary)            | Table 1, Fig 2         | Keep                                |
| Cohen’s κ (macro)              | S3 confusion extension | Align with sleep-scoring literature |
| Per-class recall               | S3 Fig caption         | REM collapse visible?               |
| `entropy_norm` / unique states | K-sweep table          | Show K≥6 collapse to 3–4 states     |
| Per-mouse NMI                  | S4 + 2 sentences main  | Rose-style transparency             |


---

## 4. Figure plan

### Main figures (current → target)


| Fig   | Current content             | Improvement                                                                                                               |
| ----- | --------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| **1** | Protocol + ladder schematic | Add “20 mice, 4 folds” label; grey out hmm_raw/HMMGMVAE if not run                                                        |
| **2** | cGMVAE vs cHMM joint        | Add chance baseline (0); optional third bar when HMMGMVAE ready                                                           |
| **3** | Within-lab heatmap          | **Move to SI** if word cap tight; replace with 1 paragraph + S2 heatmap only — *or* keep if heterogeneity is a main claim |
| **4** | K=4 compact physiology      | **Name states** after Birgitte; add EMG panel callout for Wake vs REM                                                     |
| **5** | Transitions + hypnogram     | Annotate one dominant transition edge with probability from CSV                                                           |


### SI figures (fill stubs)


| Item                   | Action                                                   | Script                            |
| ---------------------- | -------------------------------------------------------- | --------------------------------- |
| S1 ladder              | Ensure HMMGMVAE bars or dashed “pending”                 | `plot_ladder_figure.py --by-fold` |
| S2 K-sweep             | Use **dual-axis** (`k_sweep_dual_axis.pdf`) not NMI-only | `plot_k_sweep_dual_axis.py`       |
| S3 confusion           | Done                                                     | `plot_confusion_matrix.py`        |
| S4 per-mouse           | Done                                                     | `plot_per_mouse_nmi.py`           |
| S5 full grid           | `Fig3_substage_panels_full.pdf`                          | already generated                 |
| **S6 thesis taxonomy** | Export thesis Fig 30 or skip claim                       | manual / screenshot               |
| **S7 K=7 biology**     | From `professor_meeting/K07/`                            | `plot_figure27_compact.py`        |


**Regenerate all:**

```bash
source .venv/bin/activate
PYTHONPATH=. python3 scripts/paper/build_curated_figures.py
bash docs/paper/build/compile_report.sh
```

---

## 5. Text expansion plan (+600–900 words to hit 3,100–3,300)

Prioritise **interpretation**, not Methods duplication.

### Introduction (+120 words)

- [ ] One sentence: **20-mouse quality cohort** rationale (not all MSSV)
- [ ] AccuSleep: **~10 min manual calibration** vs zero-shot
- [ ] Explicit **three-way contrast** table sentence (supervised / transductive / static unsupervised) — already started; tighten

### Results — holdout (+150 words)

- [ ] **Fold 2 post-mortem:** which holdout mice (`072`, `076`, `041`, `048`, `069`, `088`) — tie to earlier tier analysis
- [ ] **S4 takeaway:** “gain not driven by sub-056 alone”
- [ ] **Chance / ceiling:** random 3-class NMI ≈ 0; supervised multi-lab κ ~0.6–0.9 (cite REST/SPINDLE) as non-comparable ceiling

### Results — substages (+200 words)

- [ ] **K selection paragraph:** report mean ± SD from `k_sweep_metrics_table.csv`; justify K=4 vs K=7 trade-off
- [ ] **State-by-state physiology:** 3–4 sentences walking Fig 4 panels A–D with θ/δ numbers
- [ ] **Fig 5:** one transition motif with probability; link to Somnotate transition literature

### Discussion (+150 words)

- [ ] **When temporal prior helps:** bout length + lab 5 label churn (cite preprocessing audit)
- [ ] **Decoder-only vs thesis encoder leakage** — one tight paragraph
- [ ] **QSLP-AE contrast** — two sentences (already started)
- [ ] **Future:** CNN end-to-end (one sentence, no scope creep)

### Conclusion (+50 words)

- [ ] Do not repeat numbers; state **population zero-shot** + **resolution limit on holdout** honestly

---

## 6. Appendix / SI completion checklist


| Stub in tex                    | Required content                                                | Priority            |
| ------------------------------ | --------------------------------------------------------------- | ------------------- |
| S1 Table                       | Literature comparison (`related_work_novelty.md` → LaTeX table) | P1                  |
| S2 Table                       | Export `holdout_ladder.csv` within-lab rows                     | P1                  |
| S3 Table                       | Substage summary CSV from K-winner physiology script            | P2 (after Birgitte) |
| S1 Appendix (synthetic)        | 1 page thesis summary + one figure                              | P2                  |
| S2 Appendix (raw HMM)          | Thesis Table 9 numbers + 1 sentence                             | P1                  |
| S3 Appendix (enc/dec ablation) | Point to `ablation_enc_dec_conditioning.md` numbers             | P2                  |
| S4 Appendix (preprocessing)    | Table from `locked_recipes.py` / lab montage doc                | P1                  |
| S6 Fig (thesis taxonomy)       | PDF embed or drop reference                                     | P3                  |
| S5 Appendix (per-seed table)   | Auto-scrape from `plots/*/metrics.txt`                          | P2                  |


---

## 7. Phased implementation order

### Phase 0 — Credibility fixes (1–2 h, no new GPU)

1. Fix **20 vs 92 mice** everywhere (`plos_my_paper.tex`, abstract, `methods_writing_guide.md`)
2. Align **K selection text** with `k_sweep_metrics_table.csv` + dual-axis SI figure
3. Regenerate figures: `build_curated_figures.py` + `compile_report.sh`
4. Scrape **per-seed table** into S5 Appendix (small script or manual CSV)

### Phase 1 — Results depth (2–3 h)

1. Expand Results subsections per §5 (+600 words)
2. Wire **S2 Table** from `paper/overleaf/tables/holdout_ladder.csv`
3. Add **κ / confusion** metrics to S3 (extend `plot_confusion_matrix.py` if needed)

### Phase 2 — Biology & expert (depends on Birgitte)

1. Interview → populate **S3 Table** state names
2. Decide **K=4 vs K=7** for main Fig 4–5 (professor meeting pack: bring K03/K04/K07)
3. Optional: thesis-style **taxonomy schematic** (Fig 30 automation — new script)

### Phase 3 — Ladder completion (GPU)

1. Submit **HMMGMVAE** locked holdout (4 folds) — update S1 Fig
2. **hmm_raw** baseline on same folds — S2 Appendix with number

### Phase 4 — PLOS packaging (post-report)

1. Author summary (150–200 words, first person)
2. Export Fig*.tif at 300 DPI
3. Zenodo tag + DOI in Data availability
4. Expand to 5,000–6,500 words if targeting Comp Biol submission

---

## 8. Reviewer / examiner question prep


| Question                | Prepared answer location                                       |
| ----------------------- | -------------------------------------------------------------- |
| Why NMI not accuracy?   | Limitations + Rose label noise; discovery not mimicry          |
| Why only 20 mice?       | Quality cohort doc + preprocessing audit                       |
| Thesis 0.70 vs 0.56?    | LOSO single-lab vs cross-lab holdout; encoder leakage          |
| Why K=4 not K=13?       | Holdout sweep peaks; higher K over-segments (Fig S2 dual-axis) |
| Incremental over mcRBM? | Zero-shot + VAE + sticky HMM + MSSV protocol                   |
| AccuSleep comparison?   | Transductive calibration vs zero-shot — no 10-min baseline     |
| Lab 2 parity?           | Montage/preprocessing locks; static GMM sufficient             |


---

## 9. Quality bar before hand-in

- [ ] Every numeric claim traceable to CSV, figure, or thesis citation
- [ ] Cohort size **20 mice** stated in Abstract, Data, Fig 1
- [ ] K-sweep **complete** (K8–K9 seeds finished or footnoted as 2/3 seeds)
- [ ] No `\nameref{...}` pointing to empty appendix stubs
- [ ] `texcount` ≥ 3,000 main text
- [ ] Fig 4–5 state names either expert-approved or explicitly “provisional”
- [ ] One explicit **supervised ceiling** sentence (not our result, but context)

---

## 10. Commands reference

```bash
# Full figure regen + report PDF
source .venv/bin/activate
PYTHONPATH=. python3 scripts/paper/build_curated_figures.py
bash docs/paper/build/compile_report.sh

# Word count
cd docs/paper && texcount -inc -1 plos_my_paper.tex

# K-sweep summary
cat docs/paper/figures/k_sweep_metrics_table.csv
```

---

## 11. Decision log (recommended defaults)


| Decision            | Recommendation                    | Rationale                                         |
| ------------------- | --------------------------------- | ------------------------------------------------- |
| Main substage K     | **K=4 for main**, K=7 in SI       | Matches holdout NMI peak; honest about resolution |
| Fig 3 in main       | **Keep**                          | Lab heterogeneity is a publishable sub-result     |
| 92 vs 20 mice       | **20 for all claims**             | Reproducibility; avoid examiner trap              |
| HMM 0.51 baseline   | **Do not cite** in holdout ladder | Wrong protocol; use raw HMM ~0 in SI              |
| Figure 30 schematic | **SI only** after manual layout   | Not blocking report; mcRBM grid more important    |


