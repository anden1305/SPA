# Paper manuscript (`docs/paper/`)

| File | Role |
|------|------|
| Manuscript | [`docs/paper/plos_my_paper.tex`](paper/plos_my_paper.tex) |
| **Writing style rule** | [`.cursor/rules/oliver-thesis-plos-writing.mdc`](../.cursor/rules/oliver-thesis-plos-writing.mdc) |
| **[plos_my_paper.pdf](plos_my_paper.pdf)** | PLOS submission build (caption-only figures) |
| **[plos_my_paper_report.pdf](plos_my_paper_report.pdf)** | Local report with embedded figures |

## Subfolders

| Folder | Contents |
|--------|----------|
| [`assets/`](assets/) | `references.bib`, `plos2025.bst`, generated `tables/`, [`appendix/`](assets/appendix/) fragments |
| [`assets/appendix/`](assets/appendix/) | MSc thesis relation, methods supplement, future work, ablation tables |
| [`figures/`](figures/) | `main/`, `supplementary/`, `k_sweep/`, `curated/`, `archive/`, `professor_meeting/` |
| [`notes/`](notes/) | Planning docs, experiment runbooks, outline, word budget |
| [`build/`](build/) | `compile_report.sh`, LaTeX cache (`cache/`) |

## Build

**Report** (figures embedded):

```bash
bash docs/paper/build/compile_report.sh
```

**PLOS submission** (from `docs/paper/`):

```bash
pdflatex plos_my_paper && bibtex plos_my_paper && pdflatex plos_my_paper && pdflatex plos_my_paper
```

Start here for workflows: [`notes/report_outline.md`](notes/report_outline.md), [`notes/holdout_experiments.md`](notes/holdout_experiments.md).

**Quality review:** remediation tracked against [`../paper_professor_review.md`](../paper_professor_review.md).

**Inferential stats** (paired fold / per-mouse tests, best-of-three headline):

```bash
PYTHONPATH=. python3 scripts/paper/compute_holdout_statistics.py
PYTHONPATH=. python3 scripts/paper/plot_fold_delta_figure.py
```

**Friday hand-in:** [`notes/FRIDAY_CHECKLIST.md`](notes/FRIDAY_CHECKLIST.md) — pending HPC jobs + what to regenerate when they finish.

**End matter order (report PDF):** Acknowledgments → References → Appendix (A Data, B Methods, C Ablations, D Figures, E Tables, F Thesis/future).
