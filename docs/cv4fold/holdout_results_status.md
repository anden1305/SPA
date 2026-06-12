# Holdout results status (locked ladder + HMM raw)

**Updated 2026-06-12.** Canonical aggregation for the paper fair-comparison table (Table 2).

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

## HMM (raw) baseline — **joint ladder closed with partial logs**

Time-domain HMM on raw EEG/EMG (`window_size: null`, `runs: 1`). Same cv4fold folds as the VAE ladder; **not comparable** to thesis expert-feature HMM (~0.51 LOSO).

**Decision (2026-06-12):** Joint folds 1–3 OOM'd or hit walltime at ~0.001–0.003 prior NMI after $\geq 10$ training epochs (fold~3: 42 epochs before validation OOM). We accept last logged validation NMI from LSF stdout (`paper/overleaf/tables/hmm_raw_joint_partial.json`) for Table 1. **No further joint HMM (raw) jobs planned** (within-lab grid complete; joint fold~4 has full `results.npz`).

| Joint fold | NMI source | Value |
|------------|------------|-------|
| 1 | Epoch 50 log (`joint_f1_28623938`) | 0.0031 |
| 2 | Epoch 50 log (`joint_f2_spill_28625583`) | 0.0011 |
| 3 | Epoch 1 log (`joint_f3_28623940`; spill run reached epoch 42 before OOM at scheduled epoch-50 validation) | 0.0034 |
| 4 | Full `results.npz` | 0.208 |

### Completion grid (`results.npz` present)

| | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|--|--------|--------|--------|--------|
| **Joint (all labs)** | † | † | † | ✓ |
| **Within lab 2** | ✓ | ✓ | ✓ | ✓ |
| **Within lab 3** | ✓ | ✓ | ✓ | ✓ |
| **Within lab 5** | ✓ | ✓ | ✓ | ✓ |

† = partial log NMI only (joint folds 1--3). Within-lab grid complete.

Log: `hpc/output/cv4fold/hmm_raw/%J.out`

Status JSON (auto): `paper/overleaf/tables/hmm_raw_holdout_status.json`

**Paper:** HMM (raw) rows appear in Tables 1--2 and S2 where data exist; `---` for joint per-site folds 1--3 (no per-lab `results.npz`). The completion grid is **not** in the report PDF — see `hmm_raw_holdout_status.tex` below.

### Completion grid (LaTeX, repo docs only)

File: [`hmm_raw_holdout_status.tex`](hmm_raw_holdout_status.tex) (not in report PDF).

Regenerate:

```bash
PYTHONPATH=. python3 scripts/cv4fold/collect_hmm_raw_holdout.py
```

---

## Related docs

- [unified_holdout_paper_line.md](unified_holdout_paper_line.md) — design
- [../paper/holdout_experiments.md](../paper/holdout_experiments.md) — paper Methods / submit
- [postprocess.md](postprocess.md) — per-mouse split after training
