# Figure and table staging (not canonical manuscript)

**Canonical PLOS manuscript:** [`docs/paper/plos_my_paper.tex`](../../docs/paper/plos_my_paper.tex)

Upload `docs/paper/plos_my_paper.tex` + `references.bib` + `plos2025.bst` to Overleaf. Export figures from `docs/paper/figures/` (or regenerate below).

## Regenerate artifacts

```bash
PYTHONPATH=. python3 scripts/paper/summarize_holdout_ladder.py --out paper/overleaf/tables/holdout_ladder.csv
PYTHONPATH=. python3 scripts/paper/plot_ladder_figure.py
PYTHONPATH=. python3 scripts/paper/plot_fig1_schematic.py
PYTHONPATH=. python3 scripts/paper/plot_figure27_compact.py --npz <path/to/results.npz>
```

| Path | Content |
|------|---------|
| `figures/` | Ladder, overview, substage panels (mirrored from `docs/paper/figures/`) |
| `tables/holdout_ladder.csv` | Auto-generated holdout NMI |

Legacy skeleton: `main.tex` (superseded by `plos_my_paper.tex`).
