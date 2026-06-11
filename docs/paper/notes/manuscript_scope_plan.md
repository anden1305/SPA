# Manuscript scope plan — PLOS Comp Biol + Friday report

**Added 2026-06-09.** Replaces the informal “5-page cap” as a *narrative skeleton* only.  
Sources: [PLOS Comp Biol submission guidelines](https://journals.plos.org/ploscompbiol/s/submission-guidelines), empirical median length from [PLOS cover-paper analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC12385384/) (~14 pages for *PLOS Computational Biology*), current [`plos_my_paper.tex`](plos_my_paper.tex) (`texcount`), cursor plans, thesis extraction report.

---

## 1. What PLOS Computational Biology actually allows

| Article type | Length rule |
|--------------|-------------|
| **Research Article** (yours) | **No word limit, no figure limit, no SI limit** — “manuscripts can be any length” |
| Software Article | **≤3,500 words** (not your format) |
| Education / Perspectives | ~2,000–2,500 words |
| Reviews | ~3,000–6,000 words |

Official guidance for Research Articles:

- Be **concise**, but **Results** should include detail for every experiment needed to support conclusions.
- **Peripheral** experiments → Supporting Information (SI), not omitted.
- **Methods** must be reproducible; long protocols can live in SI.
- **Abstract** ≤300 words; **Author summary** 150–200 words (Comp Biol requirement).
- **≤3 heading levels**; line numbers in submission file.

**Empirical norm:** median published *PLOS Computational Biology* research article ≈ **14 pages** (IQR often ~8 pages wide — method-heavy papers run long).

**Implication:** For the **Friday hand-in**, deliver **3,000–3,500 words** of polished main text (max 4,000) plus integrated figures — QSLP-style clarity at special-course length. PLOS submission may expand word count later; SI holds thesis detail.

---

**PLOS submission (after Friday):** may grow beyond 4k words (journal has no cap). **Friday article:** [`word_budget.md`](word_budget.md) — target **3,000–3,500**, max **4,000** main text, fully written (not outline).

## 2. Where you are now

| Metric | Current `plos_my_paper.tex` | Report target |
|--------|----------------------------|---------------|
| Main text words | **~2,500** (`texcount`) | **3,000–3,500** (max 4,000) |
| Methods | **~1,100** | **adequate** — trim only if over cap |
| Results | **~250** | **650–850** — main expansion |
| Introduction | **~220** | **350–450** |
| Discussion | **~130** | **350–450** |

---

## 3. Two deliverables (do not conflate)

### A. Friday special-course article (hard deadline)

**File:** `plos_my_paper_report.pdf` (`bash docs/paper/build/compile_report.sh`)

| Include | Why |
|---------|-----|
| **Polished** full narrative (Intro/Results/Discussion at word targets in §4) | Hand-in quality, not a status draft |
| **Embedded figures** (Figs 1–4 + S1–S3) with final or best-available data | Readable, self-contained PDF |
| **No Author summary** | DTU format; add for PLOS after Friday |
| **Full-width title page** | Report mode (no PLOS left gutter) |
| **Limitations inline** in Results/Discussion (seed fragility, lab locks, incomplete rungs if any) | Honest without “WIP” banners |
| Short **thesis → paper** mapping (decoder-only novelty vs thesis encoder+decoder; LOSO 0.70 vs holdout 0.56) | Shows special-course delta |
| Pointer to [`source_extraction/extraction_report.md`](source_extraction/extraction_report.md) | Documents what was verified from thesis PDF |

**Target ~15–20 pages** with figures. Same science as PLOS submission; differs only in report layout and missing Author summary.

### B. PLOS Computational Biology submission (later)

**File:** `plos_my_paper.pdf` (caption-only figures; upload `figures/Fig*.tif` separately)

Same scientific content as the report, but:

- **Author summary** included (150–200 words, first person).
- PLOS title-page geometry on page 1.
- No “work in progress” language — only completed experiments in Results.
- Methods reproducible enough for a computational biology reviewer.

---

## 4. Main text — concise expansion (report budget)

See [`word_budget.md`](word_budget.md). Use the **3,000–3,500** report budget for Friday; reserve **5,000–6,500** for PLOS expansion after hand-in.

### Introduction (**350–450 words**, +~150 from now)

| ¶ | Content |
|---|---------|
| 1 | Rose 2026 + MSSV — manual disagreement, supervised generalization failure |
| 2 | One sentence each: AccuSleep (transductive), SegWay (per-mouse), mcRBM (no holdout) |
| 3 | Thesis anchor — HMM 0.51, cGMVAE LOSO 0.70; why that is not zero-shot holdout |
| 4 | Three contributions + scope (labs 2/3/5, SI for ablations) |

### Results (**650–850 words**, +~400–600 from now)

| Subsection | Words | Content |
|------------|------:|---------|
| Holdout ladder | ~250 | Fig 2, Table 1, fold 2 caveat, Δ=+0.22 |
| Within-lab | ~120 | Lab 2 vs 3/5; cite S2 — no full table in main |
| Substages | ~150 | K grid + selection rule; Fig 3–4 (K-sweep when ready) |
| Physiology | ~150 | One paragraph on θ/δ, bout, transitions |

### Discussion (**350–450 words**, +~200 from now)

| Topic | ~words |
|-------|-------:|
| Main finding + zero-shot vs transductive | 100 |
| When HMM prior helps; decoder-only vs thesis | 100 |
| Limits + QSLP colleague contrast (2 sentences) | 100 |
| Next steps (K-sweep, Birgitte, HMMGMVAE) | 80 |

### Methods (**~1,100 words — hold**)

Already adequate. Trim to SI only if total exceeds 4,000.

**Methods checklist** (already covered in draft):

| Subsection | Must include |
|------------|--------------|
| Dataset | 92 mice; labs 2/3/5; epoch length; labels eval-only |
| Preprocessing | Per-lab spectral locks; SI for YAML |
| Holdout inference | No subject ID; prior-only; no decoder |
| Substage sweep | K grid; selection criterion |

Move full hyperparameter tables and ablations to SI — not main text.

---

## 5. Supporting Information — what you **NEED**

SI is not a dump — it must answer reviewer attacks and **deduplicate the thesis**.

### MUST have before submission

| SI item | Content | Why required |
|---------|---------|--------------|
| **S1 Fig** | Per-fold ladder (all 4 rungs × 4 folds) | Complete ladder when HMMGMVAE finishes |
| **S2 Fig** | Within-lab heatmap | Absolute NMI not comparable to joint — keep out of main table footnotes only |
| **S3 Fig** | K-sweep curve (real, not placeholder) | Justifies K for Fig 3–4 |
| **S4 Fig** | Thesis K=13 taxonomy | Shows substage program predates holdout; avoids “only 3 states” criticism |
| **S1 Table** | Extended literature comparison | SegWay, WUCSS, REST, mcRBM, etc. |
| **S2 Table** | Full within-lab NMI (all folds) | Reproducibility |
| **S3 Table** | Substage expert summary | After Birgitte interview |
| **S1 File** | Locked YAML recipes | Reproducibility |
| **S1 Appendix** | **Synthetic HMM validation** (thesis) | Proves temporal prior behaviour |
| **S2 Appendix** | **Raw time-domain HMM failure** | Motivates spectral features |
| **S3 Appendix** | **Encoder+decoder vs decoder-only ablation** | Core novelty vs thesis |
| **S4 Appendix** | **Preprocessing locks** per laboratory | Explains lab 2 vs 3/5 heterogeneity |
| **S5 Appendix** | **Per-seed NMI table** (joint + within) | Seed fragility transparency |

### SHOULD have (strongly recommended)

| SI item | Content |
|---------|---------|
| S6 Appendix | Reviewer Q&A ([`related_work_novelty.md`](related_work_novelty.md) stress-test) |
| S7 Appendix | 2023–2026 landscape paragraph (REST, Rose, WUCSS) |
| S5 Fig | Confusion matrices / per-mouse NMI distributions (fold 4 exemplar) |
| S6 Fig | Latent UMAP / trajectory (if computed) |
| S2 File | Fold manifest CSV paths |

### Do NOT put in SI (common mistakes)

| Avoid | Reason |
|-------|--------|
| Restating entire thesis | Duplicate publication risk — cite thesis + show *delta* only |
| Special-course framing doc verbatim | Outdated goals; use only as future-work sentence |
| Unfinished experiments as if complete | Finish HMMGMVAE / K-sweep for Friday, or state exclusion/limitation once in Discussion — no placeholder figures |
| 190-state mcRBM comparison runs | Out of scope per plan |

---

## 6. Thesis vs paper — what stays where

| Thesis content | Main | SI | Omit |
|----------------|------|-----|------|
| Synthetic HMM benchmarks | 1 sentence | **S1 Appendix** | Full tables |
| Time-domain HMM failure | — | **S2 Appendix** | — |
| Feature HMM 0.51 | Cited as baseline | Per-mouse detail | Re-derive |
| cGMVAE 0.70 LOSO (encoder+decoder) | Contrast only | Ablation | Present as holdout result |
| K=13 substage taxonomy | 1 sentence | **S4 Fig** | Full panel in main until K-sweep winner ready |
| 10-mouse curated subset | — | Footnote | Imply full MSSV cv4fold |
| Decoder-only + zero-shot | **Main claim** | Ablation appendix | — |
| cHMM–GMVAE holdout 0.56 | **Main result** | Per-seed table | — |

---

## 7. Implementation order

### Phase 1 — Polished article (Friday)

1. Finish **HMMGMVAE** holdout + **K-sweep** → update Figs 2–4 and S1–S3 (no placeholders).
2. Expand Intro + Results + Discussion in `plos_my_paper.tex` to word targets (§4).
3. Add **Table 2** (within-lab summary from `holdout_ladder.csv`) if not in main yet.
4. Remove “pending / placeholder” language from captions and `\ifreportmode` WIP blocks.
5. Birgitte interview → substage names in Results if feasible; else SI pointer + limitation sentence.
6. `bash docs/paper/build/compile_report.sh` → hand in `plos_my_paper_report.pdf`.

### Phase 2 — PLOS submission package (after Friday)

1. Strip report-only text; restore Author summary.
2. Export `Fig1–5.tif`; caption-only `plos_my_paper.pdf`.
3. Upload SI PDFs/files per item above.
4. Zenodo tag + DOI in Code availability.
5. Optional word-count expansion toward §8 budget if reviewers need more detail.

---

## 8. Word-count budget (target)

| Section | Words |
|---------|------:|
| Abstract | 250–280 |
| Author summary | 150–200 |
| Introduction | 900–1,100 |
| Results | 1,800–2,500 |
| Discussion | 800–1,000 |
| Materials and methods | 1,200–1,800 |
| **Total main text** | **~5,000–6,500** |

At ~500 words/page plus figures → **~12–15 pages** — aligned with journal median.

---

## 9. Quick reference commands

```bash
# Friday report (figures, no Author summary)
bash docs/paper/build/compile_report.sh

# PLOS submission PDF (caption-only)
cd docs/paper && pdflatex plos_my_paper.tex && bibtex plos_my_paper && pdflatex plos_my_paper.tex && pdflatex plos_my_paper.tex

# Thesis / framing cross-check
PYTHONPATH=. python3 scripts/paper/extract_source_materials.py
```

---

## 10. Decision log

| Old plan | New plan |
|----------|----------|
| “5-page main text” | 5-page **outline**; submit **~12–15 pages** |
| Methods after Discussion | OK for Comp Biol; keep Methods at end unless cover letter requests move |
| All thesis detail in supplement | **Yes** — main = holdout ladder + decoder-only novelty + K-winner biology only |
