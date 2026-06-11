# Paper figure assets

**Updated 2026-06-11.**

| Folder | Contents |
|--------|----------|
| [`main/`](main/) | Active manuscript figures (`plos_my_paper.tex` / `compile_report.sh`) |
| [`supplementary/`](supplementary/) | SI figures (appendix) |
| [`k_sweep/`](k_sweep/) | K-sweep plots, metrics CSV/JSON, alternate exports |
| [`archive/`](archive/) | Deprecated or removed main-text figures (Fig 2–4 ladder/heatmap/compact biology) |
| [`professor_meeting/`](professor_meeting/) | Per-K biology packs for supervisor meetings |

Regenerate curated set:

```bash
PYTHONPATH=. python3 scripts/paper/build_curated_figures.py
```

Canonical sources also live in [`curated/`](curated/) and `results/cv4fold/paper_figures/`.
