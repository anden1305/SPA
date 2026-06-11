# Biology meeting — substage K selection

Generated: 2026-06-11T09:32:15.348240+00:00

**Model:** cHMM–GMVAE locked, joint holdout **fold 4**
**Stats pick:** K=4 (best holdout NMI)

## Start here

1. `00_overview/k_sweep_dual_axis.pdf` — NMI + log p(z) vs K (thesis Fig 23)
2. Compare candidate folders: **K03, K04, K05, K07** (see table below)
3. Per-K README in each folder lists what to show

## Candidate comparison

| K | Best NMI | Mean ± SD | Biology folder |
|---|---------|-----------|----------------|
| 3 | 0.6350 | 0.6161 ± 0.0199 | `K03/` — Near-macro |
| 4 | 0.6428 | 0.6174 ± 0.0233 | `K04/` — Best holdout NMI |
| 5 | 0.6285 | 0.5809 ± 0.0404 | `K05/` — Slightly finer |
| 7 | 0.5874 | 0.5331 ± 0.0558 | `K07/` — Thesis-like resolution |

## All K

Folders `K03/` … `K15/` each contain the full thesis plot stack.

Interview script: `docs/paper/birgitte_interview_guide.md`
