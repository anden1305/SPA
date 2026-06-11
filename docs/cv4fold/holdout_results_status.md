# Holdout results status (locked ladder + HMM raw)

**Updated 2026-06-11.** Canonical aggregation for the paper fair-comparison table (Table 2).

## Reporting rules

| Item | Rule |
|------|------|
| **VAE models** (cGMVAE, HMMGMVAE, cHMM–GMVAE) | `runs: 3` → **best-of-3** prior NMI per fold; accuracy from same seed's `results.npz` |
| **HMM (raw)** | `runs: 1` per fold (time-domain HMM, thesis-style) |
| **Point estimate** | Mean over four folds of best-of-three-seed values (only folds with finished `results.npz`) |
| **Uncertainty** | Not shown in Table 2; seed instability motivates best-of-three selection (see table caption). Fold ranges appear in Fig~2 error bars. |
| **Fair joint vs within** | Joint train: metrics on **held-out mice from one lab only**; within train: same mice, single-site training pool |

Regenerate tables:

```bash
PYTHONPATH=. python3 scripts/paper/summarize_holdout_ladder.py
PYTHONPATH=. python3 scripts/cv4fold/collect_hmm_raw_holdout.py
PYTHONPATH=. python3 scripts/paper/scrape_holdout_accuracy.py
PYTHONPATH=. python3 scripts/paper/build_holdout_scope_table.py
```

Outputs: `docs/paper/assets/tables/T2_holdout_fair_comparison.tex`, `paper/overleaf/tables/hmm_raw_holdout_status.json`.

---

## VAE locked ladder (frequency + decoder-only conditioning)

**Status: complete** — 48/48 cells (3 models × 4 folds joint + 3 models × 3 labs × 4 folds within-lab).

| Scope | Path | Cells |
|-------|------|-------|
| Joint | `results/cv4fold/joint_holdout/fold_{k}/{model}_locked/` | 12/12 |
| Within-lab | `results/cv4fold/unified_holdout/{lab}/fold_{k}/{model}_locked/` | 36/36 |

CSV: `paper/overleaf/tables/holdout_ladder.csv`, `holdout_accuracy.csv`.

---

## HMM (raw) baseline — **incomplete**

Time-domain HMM on raw EEG/EMG (`window_size: null`, `runs: 1`). Same cv4fold folds as the VAE ladder; **not comparable** to thesis expert-feature HMM (~0.51 LOSO).

### Completion grid (`results.npz` present)

| | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|--|--------|--------|--------|--------|
| **Joint (all labs)** | — | — | — | ✓ |
| **Within lab 2** | ✓ | ✓ | ✓ | — |
| **Within lab 3** | — | ✓ | — | — |
| **Within lab 5** | ✓ | ✓ | — | — |

**7 / 16** cells with validation `results.npz` (as of 2026-06-11). The collector prefers the newest run **with** `results.npz`; newer failed retries without `results.npz` are ignored.

Submit remaining jobs:

```bash
PYTHONPATH=. python3 scripts/cv4fold/generate_hmm_raw_holdout.py
bash hpc/submit/cv4fold/submit_hmm_raw_holdout.sh
```

Log: `hpc/output/cv4fold/hmm_raw/%J.out`

Status JSON (auto): `paper/overleaf/tables/hmm_raw_holdout_status.json`

**Paper:** HMM (raw) rows appear in Table 2 / S2 where data exist; `---` elsewhere until the grid is complete.

---

## Related docs

- [unified_holdout_paper_line.md](unified_holdout_paper_line.md) — design
- [../paper/holdout_experiments.md](../paper/holdout_experiments.md) — paper Methods / submit
- [postprocess.md](postprocess.md) — per-mouse split after training
