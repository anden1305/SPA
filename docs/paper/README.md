# Paper manuscript (`docs/paper/`)

| File | Role |
|------|------|
| **[plos_my_paper.tex](plos_my_paper.tex)** | Manuscript source |
| **[plos_my_paper.pdf](plos_my_paper.pdf)** | PLOS submission build (caption-only figures) |
| **[plos_my_paper_report.pdf](plos_my_paper_report.pdf)** | Local report with embedded figures |

## Subfolders

| Folder | Contents |
|--------|----------|
| [`assets/`](assets/) | `references.bib`, `plos2025.bst`, `plos_template.tex`, generated `tables/` |
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

**Friday hand-in:** [`notes/FRIDAY_CHECKLIST.md`](notes/FRIDAY_CHECKLIST.md) — pending HPC jobs + what to regenerate when they finish.
