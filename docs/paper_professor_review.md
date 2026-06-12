# Senior professor review — `plos_my_paper_report.pdf`

**Added 2026-06-11.** Read as a special-course / pre-submission manuscript, not a finished PLOS paper. The core idea is sound and the evaluation protocol is more serious than most student work. Several places still read like an internal report or LLM-polished draft, and a few scientific choices would not survive a sharp reviewer without tightening.

**Manuscript:** [`docs/paper/plos_my_paper_report.pdf`](paper/plos_my_paper_report.pdf) · source: [`docs/paper/plos_my_paper.tex`](paper/plos_my_paper.tex)

---

## What stands out positively

**The problem is well chosen.** Cross-lab mouse sleep staging under MSSV is a real failure mode for supervised models; framing zero-shot holdout without per-mouse calibration is a legitimate contribution.

**The evaluation design is the strongest part.** Leave-mice-out across three sites, locked preprocessing, decoder-only conditioning at inference, and a matched model ladder (static GMM vs sticky HMM–GMM) are the right ingredients. Table 1’s consistent cHMM–GMVAE gain over cGMVAE on every fold is a clear, defensible headline.

**You are unusually honest in places.** Within-lab beating joint, lab 2 weakness, seed/fold fragility, NMI vs macro-accuracy mismatch, and incomplete HMM(raw) runs are acknowledged rather than hidden.

**Fig 1 and the Methods block are close to publishable structure** — train vs holdout path, prior-based staging, equations where needed.

---

## Where it reads “weird” or like AI / internal tooling

### 1. Meta-document language in a scientific manuscript

These sections sound like project management, not science:

- **“Research arc: from thesis to this report”**
- **“Explored but not in the locked paper protocol”**
- **“Phase A / B / C / D”** with repository pointers
- **S6 Table:** `See docs/cv4fold/holdout results status.md` — a file path in the PDF is a hard no for any journal

A reviewer will ask: *Is this a paper or a changelog?* Move thesis lineage to a one-paragraph Discussion footnote or drop it; never cite repo markdown paths in tables.

### 2. Placeholder / WIP leakage into captions and tables

| Location | Problem |
|----------|---------|
| Fig 3A | **“K=13 placeholder”**, **“thesis-like”** |
| S3 Table | **“Provisional name”**, **“Expert names pending curation”** |
| References | **“authors S”**, **“replace author list from publisher”** |
| Data availability | **“Zenodo archive will be released before submission”** |
| Fig 1 | Typos: **“SIgnals”**, **“embeeder”** |

These are the clearest “not submission-ready” signals. They undermine trust in every number nearby.

### 3. Typical LLM / template prose patterns

- **Abstract:** stacked novelty claims (“novel in combining…”, “threefold contributions”, “offers a path to compare”) without one crisp quantitative anchor early enough.
- **Introduction:** citation-dense chains (*“X showed… Y remains… Z including… Katsageorgiou et al. showed… Stevner et al. used…”*) — correct but reads assembled rather than argued.
- **Repeated boilerplate:** “contrasting with transductive per-animal calibration” / AccuSleep appears many times; say it once sharply in Intro + Methods.
- **Hedged mechanism language:** “in the spirit of”, “aligns with supervised evidence”, “surfaces transition-rich substates” — fine once, but overused.
- **Conclusion:** restates abstract without new synthesis or forward-looking limitation.

### 4. Formatting artifacts

- Line numbers embedded in body text (`Introduction 1`, `Rodent sleep… 2`) — LaTeX lineno leak.
- **“June 11, 2026”** on every page — fine for a report, wrong for submission.
- Missing **Author Summary** (required for PLOS Computational Biology).

---

## Where rigour is missing

### Statistical reporting

| Issue | Why it matters |
|-------|----------------|
| **Best-of-three seeds** as primary metric (Table 1) | Looks like optimistic selection; report mean ± SD across seeds, or paired seed-level comparison |
| **No uncertainty on fold means** | Four folds is small; bootstrap over mice or report per-mouse distribution (you have S2 Fig — discuss in main text) |
| **No significance / paired tests** | Δ = +0.30 NMI is compelling narratively but not inferential |
| **NMI only** | Sleep field expects **κ**, per-class **F1/recall**, especially REM; Rose 2026 is about rater agreement, not NMI |
| **Macro accuracy in appendix** | S3 confusion shows **REM recall ≈ 3%** on fold 4 — that should be in main Results, not buried in SI |

### Experimental design gaps

1. **K selection on fold 4 only** — choosing K=4 from a sweep on one holdout fold risks overfitting that fold. K should be chosen on training folds or averaged across folds.

2. **HMM(raw) incomplete (7/16 cells)** — yet Table 1 includes a mean row. Either finish the grid or remove incomplete baselines from headline tables.

3. **HMMGMVAE on Lab 2 joint: NMI 0.039** — catastrophic; needs explanation (collapse? wrong prior tier? seed failure?). Without it, the “fair ladder” claim is shaky.

4. **Decoder-only conditioning** is the stated novelty, but the ablation is **appendix-only (S3)**. A reviewer will want one main-text sentence with numbers: encoder+decoder vs decoder-only on the same folds.

5. **20 vs 92 MSSV mice** — you screen to 20 HQ mice (good), but the paper never states the full MSSV scale up front then justifies the subset. That invites “small n” criticism.

6. **Substage story at K=4 is internally inconsistent:**
   - Text: “transition-enriched state at Wake–NREM boundaries”
   - S3 Table: S3 and S4 are **0.3% / 0.2% occupancy**, both REM-dominated
   - S3 note admits transition structure is “visible in the taxonomy schematic (Fig 3)” **at K=13**, not K=4

   So the substage claim is really a **K=13 thesis figure dressed as K=4** — a serious narrative problem.

### Theory / methods depth

Missing or thin:

- **Why prior NMI on μₜ (not sampled zₜ)?** — sensible, but needs one sentence on bias/variance tradeoff.
- **Why d=6, T=64, κ=0.92, β schedule?** — locked recipes are fine for reproducibility, but no sensitivity or ablation.
- **Warm-start schedule (epochs 18, 37…)** — complex; no ablation showing it beats cold HMM–GMM start.
- **Identifiability of mixture components** — label switching across seeds/folds not discussed.
- **Comparison to supervised ceilings** — SPINDLE/REST/García Ciudad numbers belong in Intro or one benchmark paragraph (S1 Table is referenced but not used to calibrate expectations in Results).

---

## Specific missing plots, tables, and information

### Must-have for a credible sleep ML paper

| Missing | Purpose |
|---------|---------|
| **Hypnogram exemplar** (1–2 held-out mice) | Expert vs predicted timeline — standard in mcRBM, Somnotate, SPINDLE |
| **Per-class metrics table** (Wake / NREM / REM κ or F1) | Field-readable performance; exposes REM failure |
| **Seed stability table/plot** | All 3 seeds × 4 folds, not best-only |
| **Decoder-only ablation table** | Core claim needs one number in main text |
| **Supervised reference row** | “Unsupervised NMI 0.58 vs supervised κ ~X when retrained multi-lab” |
| **K-sweep across all folds** (or train-only model selection) | Justify K=4 without fold-4 cherry-picking |

### Should-have for the substage / dynamics story

| Missing | Purpose |
|---------|---------|
| **K=4 transition matrix in main text** (with one quoted probability, e.g. Wake→NREM) | Somnotate-style concrete dynamics |
| **Latent PCA / UMAP figure in main** (not just mentioned) | Shows separability of substates |
| **Expert-validated state names** | mcRBM names states from biology; “S1 Wake / S4 REM′ provisional” is not enough |
| **Consistent K in Fig 3** | Either K=4 taxonomy + physiology, or drop K=13 panel from main figure |
| **REM-specific analysis paragraph** | Literature says temporal context helps REM most; you should report REM recall explicitly |

### Submission hygiene

- Complete **references** (no placeholder authors)
- **Author Summary** (PLOS Comp Biol)
- Remove lineno artifacts and draft date
- Fix Fig 1 typos
- Replace “will be released” with actual Zenodo DOI or “available upon acceptance”

---

## Section-by-section professor notes

**Title** — Long but accurate. “Enable zero-shot” is slightly marketing; “without per-animal calibration at inference” is more precise.

**Abstract** — Too many clauses before the punchline (0.58 vs 0.28). Lead with cohort + protocol + main number in sentence 2.

**Introduction** — Good coverage of related work, but reads like a literature matrix. End with one sharp gap sentence: *No prior work combines X under leave-mice-out cross-lab holdout with Y.*

**Data** — Solid screening criteria. Add: “MSSV comprises N mice across 5 labs; we analyse 20 after quality screening because…”

**Methods** — Strong. Add: metric justification (why NMI over κ), and leakage check (“training mice never appear in validation folds”).

**Results** — Table 1 is the paper. Table 2 is interesting but needs interpretation of HMMGMVAE lab-2 collapse and Lyon parity (HMMGMVAE ≈ cHMM within-lab). Substage section oversells K=4 relative to evidence.

**Limitations** — Good start; add REM failure, best-of-3 selection, single-fold K sweep, incomplete baselines.

**Conclusion** — Too short and repetitive; should state what a sleep biologist can *do* with this (compare studies without harmonised labels) and what they cannot (replace expert scoring).

**Appendix** — Too much thesis arc; not enough ablation numbers in main-facing SI.

---

## Bottom line (examiner voice)

> *You have a defensible special-course contribution: decoder-only conditional VAE + sticky HMM–GMM prior improves cross-lab holdout NMI over a static mixture, with a careful (if small) cohort protocol. The manuscript is not yet at PLOS quality — it still exposes internal workflow (placeholders, repo paths, provisional labels, incomplete baselines), overstates the K=4 substage discovery while showing K=13 in the main dynamics figure, and under-reports REM failure and seed variance. Fix the narrative consistency first; then add κ/F1, hypnograms, and decoder-only ablation numbers in the main text. The science is promising; the packaging still reads like a polished draft rather than a reviewed paper.*

---

## Related docs

- [`docs/paper/notes/report_improvement_plan.md`](paper/notes/report_improvement_plan.md) — actionable revision plan
- [`docs/paper/notes/manuscript_scope_plan.md`](paper/notes/manuscript_scope_plan.md) — scope and Friday checklist
- [`docs/paper/notes/FRIDAY_CHECKLIST.md`](paper/notes/FRIDAY_CHECKLIST.md) — hand-in checklist
