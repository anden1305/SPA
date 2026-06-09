# Special-course / PLOS Computational Biology paper

**Added 2026-06-09.** Manuscript and experiment runbooks for *Decoder-only conditional latent models with temporal mixture priors for unsupervised cross-lab mouse sleep staging and substage discovery*.

## Canonical manuscript (Overleaf)

| File | Role |
|------|------|
| **[plos_my_paper.tex](plos_my_paper.tex)** | **PLOS Comp Biol draft** (single-file, caption-only figures) |
| [plos_template.tex](plos_template.tex) | Official template reference (do not edit) |
| [plos2025.bst](plos2025.bst) | Vancouver BibTeX style |
| [references.bib](references.bib) | Bibliography |
| [figures/](figures/) | Exported `Fig1.tif` … `Fig4.*` for upload |

Cursor rule when editing `.tex`: [`.cursor/rules/plos-compbiol-tex.mdc`](../../.cursor/rules/plos-compbiol-tex.mdc).

```bash
cd docs/paper
pdflatex plos_my_paper && bibtex plos_my_paper && pdflatex plos_my_paper && pdflatex plos_my_paper
```

Initial submission: compile PDF **without** embedded figures; upload `figures/Fig*.tif` separately.

## Docs and runbooks

| Doc | Purpose |
|-----|---------|
| [related_work_novelty.md](related_work_novelty.md) | Literature tiers, comparison table, reviewer stress test |
| [deep_research_synthesis.md](deep_research_synthesis.md) | ChatGPT + Gemini merge for supervisors |
| [holdout_experiments.md](holdout_experiments.md) | P0 ladder + zero-shot inference protocol |
| [k_sweep_experiments.md](k_sweep_experiments.md) | P1: K substage sweep on cHMM |
| [birgitte_interview_guide.md](birgitte_interview_guide.md) | Biological interpretation script |

## Figure scripts

```bash
PYTHONPATH=. python3 scripts/paper/plot_fig1_schematic.py
PYTHONPATH=. python3 scripts/paper/plot_ladder_figure.py
PYTHONPATH=. python3 scripts/paper/plot_ladder_figure.py --by-fold
PYTHONPATH=. python3 scripts/paper/plot_within_lab_heatmap.py
PYTHONPATH=. python3 scripts/paper/plot_k_sweep_curve.py
PYTHONPATH=. python3 scripts/paper/plot_figure27_compact.py --npz <path/to/results.npz>
```

Figure staging copies: [`../../paper/overleaf/figures/`](../../paper/overleaf/figures/).
