# Word budget — special-course report

**Added 2026-06-09.** Hard cap **4,000** words main text. Target **3,000–3,500** — stop when each section passes the adequacy checklist below.

Count (report build, no Author summary in PDF):

```bash
cd docs/paper && texcount -inc -1 plos_my_paper.tex
```

Exclude from budget: Author summary (PLOS only), SI caption block at end, figure/table captions (count separately).

---

## Section targets

| Section | Target | Max | Role |
|---------|-------:|----:|------|
| Abstract | 220–250 | 280 | Problem, ladder, holdout NMI, substates |
| Introduction | 350–450 | 500 | Rose → gap → contributions |
| **Data** | 300–400 | 450 | MSSV, preprocessing, spectral inputs |
| **Methods** (3 subsections) | 900–1,100 | 1,250 | Protocol, ladder/architecture, training; Fig 1 here |
| **Results and discussion** | 900–1,100 | 1,250 | Figs 2–4, Table 1; interpret inline |
| **Conclusion** | 120–180 | 220 | Summary only |
| Acknowledgments | — | 50 | |
| **Total main text** | **~3,000–3,500** | **4,000** | Appendix captions excluded |

Structure: [`report_outline.md`](report_outline.md).

**Current (~Jun 2026):** ~2,500 words in `plos_my_paper.tex` — Methods already adequate; **add ~400–700 words mainly in Results + Discussion**, not Methods.

---

## What to add (priority order)

1. **Results — within-lab** (~120 w): lab 2 vs 3/5 pattern; cite S2 Fig, not full table in main.
2. **Results — fold interpretation** (~100 w): fold 2 below HMM; seed spread.
3. **Results — K-sweep** (~80 w): grid, selection rule; Fig 3–4 from winner when ready.
4. **Introduction** (~150 w): one sentence each on AccuSleep, SegWay, thesis LOSO vs holdout.
5. **Discussion** (~200 w): when HMM prior helps; decoder-only vs thesis; colleague QSLP contrast (2 sentences).

---

## What to cut or defer (stay concise)

| Move to SI / skip in main | Why |
|---------------------------|-----|
| Full within-lab table | S2 Table + heatmap |
| Synthetic HMM validation | S1 Appendix |
| Encoder+decoder ablation | S3 Appendix |
| Thesis K=13 taxonomy | S4 Fig |
| Locked YAML hyperparameter tables | S1 File |
| Literature comparison table | S1 Table |
| Repeated zero-shot definition | State once in Methods + Fig 1B |

---

## Adequacy checklist (stop writing when all true)

- [ ] Reader understands **train vs holdout** without reading code
- [ ] **Ladder** and **locked** hyperparameters named ($d=6$, $T=64$, $\kappa=0.92$, $K$ ladder vs sweep)
- [ ] **Every main figure** has one interpretive paragraph with a number
- [ ] **Limitations** named (label noise, lab locks, seed fragility, pending HMMGMVAE/K-sweep)
- [ ] `texcount` main text ≤ **4,000**; ideally **3,000–3,500**

---

## vs PLOS submission (later)

Same scientific content can grow for Comp Biol (no hard cap). For the **DTU report**, treat **3,500 words + figures** as the deliverable.
