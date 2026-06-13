# Population biology — substage K (appendix)

Generated: 2026-06-13T14:51:38.987361+00:00

**Model:** cHMM–GMVAE locked, **all 20 mice** (incohort train+val, 5 seeds)
**Incohort pick:** K=4 (best prior NMI across seeds; appendix only)

## Start here

1. `00_overview/k_sweep_dual_axis.pdf` — NMI + log p(z) vs K (thesis Fig 23)
2. `00_overview/k_sweep_active_substages.csv` — configured K vs **active** substates used
3. Compare candidate folders (see table below)
4. Per-K README in each folder lists what to show

## Candidate comparison

| K | Best NMI | Mean ± SD | Biology folder |
|---|---------|-----------|----------------|
| 4 | 0.5167 | 0.4849 ± 0.0444 | `K04/` — Best holdout NMI |
| 5 | 0.4844 | 0.4435 ± 0.0254 | `K05/` — Slightly finer |
| 7 | 0.4684 | 0.4606 ± 0.0078 | `K07/` — Thesis-like resolution |

## All K

Folders `K03/` … `K15/` each contain the full thesis plot stack.

Interview script: `docs/paper/birgitte_interview_guide.md`
