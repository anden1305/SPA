# Report outline (canonical)

**Added 2026-06-09.** Single source of truth for [`plos_my_paper.tex`](plos_my_paper.tex) structure.  
Style reference: Ciudad et al., QSLP-AE ([OpenReview K3Z4jVHUnf](https://openreview.net/pdf?id=K3Z4jVHUnf)) — **DATA** before **METHODS**, few section headings, **bold paragraph starters** inside Methods.

**Build:** `bash docs/paper/build/compile_report.sh` → `plos_my_paper_report.pdf`  
**Word budget:** [`word_budget.md`](word_budget.md) — target 3,000–3,500 main text.

---

## Top-level sections (in order)

| # | Section | LaTeX | Notes |
|---|---------|-------|-------|
| 1 | **Abstract** | `\section*{Abstract}` | ≤250 w; no citations |
| 2 | **Introduction** | `\section*{Introduction}` | Problem → gap → contributions → pointer to Data/Methods |
| 3 | **Data** | `\section*{Data}` | MSSV cohort, preprocessing, spectral inputs — **numbers first** |
| 4 | **Methods** | `\section*{Methods}` | 1–3 `\subsection*{}`; use `\methodpara{Label.}` for bold inline starters |
| 5 | **Results and discussion** | `\section*{Results and discussion}` | Subsections + main Figs 2–4 + Table 1; interpret inline |
| 6 | **Conclusion** | `\section*{Conclusion}` | 3–5 sentences; no new results |
| 7 | **Acknowledgments** | `\section*{Acknowledgments}` | |
| 8 | **References** | `\bibliography{references}` | |
| 9 | **Appendix** | `\section*{Appendix}` | `\subsection*{}` for SI figures, tables, ablations |

**PLOS submission only** (not report PDF): `\section*{Author summary}` after Abstract when `\ifreportmode` is false.

**Do not use** separate `\section*{Discussion}`, `\section*{Materials and methods}`, or `\section*{Supporting information}` — content lives in the sections above.

---

## Section content map

### Data (~300–400 words)

- MSSV (92 mice, labs 2/3/5, 4 s epochs, Artifact removal)
- Per-lab spectral locks (lab 2 vs 3/5)
- Input tensor $\mathbf{x}_t \in \mathbb{R}^{C \times F}$, sequence length $S$
- Labels for evaluation only

### Methods (3 subsections, ~900–1,100 words total)

**Fig 1** cited here (cohort split, zero-shot path, model ladder — architecture figure).

| Subsection | Bold starters (`\methodpara`) | Content |
|------------|------------------------------|---------|
| **Cross-validation and holdout protocol** | Evaluation scopes. Holdout inference. Agreement metrics. | 4-fold leave-mice-out; joint vs within-lab; zero-shot prior-only path; NMI eq. |
| **Model ladder and architecture** | Model ladder. Encoder–decoder. Latent priors. | Four rungs; CNN+MLP; $d=6$, $T=64$, $\kappa=0.92$; warm HMM–GMM |
| **Training, substages, and reproducibility** | Decoder-only conditioning. Training and checkpoints. Substage sweep. Data and code. | Subject emb decoder-only; seeds; $K$ grid; OpenNeuro + repo |

### Results and discussion (4 subsections, ~900–1,100 words)

| Subsection | Figures / tables |
|------------|------------------|
| **Cross-lab holdout performance** | Fig 2, Table 1 |
| **Laboratory heterogeneity** | cite Appendix within-lab heatmap |
| **Substage structure and physiology** | Fig 3, Fig 4 |
| **Methodological implications** | vs AccuSleep/SegWay, vs thesis, vs QSLP-AE; limitations inline |

### Conclusion (~120–180 words)

Summarise holdout gain, decoder-only + temporal prior, substage programme; one sentence on pending HMMGMVAE / K-sweep.

### Appendix (subsections)

| Subsection | Former SI label | Content |
|------------|-----------------|---------|
| **Additional holdout results** | S1 Fig, S2 Fig, S2 Table | Per-fold ladder; within-lab heatmap |
| **Substage resolution and thesis taxonomy** | S3 Fig, S4 Fig, S3 Table | $K$-sweep; thesis $K{=}13$ |
| **Ablations and validation** | S1 Appendix | Synthetic HMM, raw HMM failure, encoder+decoder ablation |
| **Configuration and literature** | S1 File, S1 Table | Locked YAML; extended comparison table |

Cross-ref in main text: `\nameref{app:...}` or legacy `\nameref{S2_Fig}` labels preserved on appendix figures.

---

## LaTeX conventions

```latex
% Bold paragraph starter (QSLP-style) — defined in plos_my_paper.tex
\methodpara{Dataset.} All experiments used MSSV ...
```

- **≤3** `\subsection*` levels under Methods and under Results and discussion.
- Main **interpretive** figures in Results; **protocol/architecture** figure (Fig 1) in Methods.
- Availability (data/code) = final `\methodpara` block in Methods, not a top-level section.

---

## Agent / author workflow

1. Edit structure only via this outline + `plos_my_paper.tex`.
2. Prose style: [`.cursor/rules/report-writing-qslp-style.mdc`](../../.cursor/rules/report-writing-qslp-style.mdc).
3. PLOS template mechanics: [`.cursor/rules/plos-compbiol-tex.mdc`](../../.cursor/rules/plos-compbiol-tex.mdc).
4. After edits: `bash docs/paper/build/compile_report.sh`.
