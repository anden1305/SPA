# Friday deadline — polished article checklist

**Hard deadline:** Friday hand-in of a **polished** special-course article — complete narrative, integrated figures, no “draft / WIP / placeholder” framing in the PDF.

**Deliverable:** `docs/paper/plos_my_paper_report.pdf` (embedded figures, no Author summary). Same scientific content as the eventual PLOS file; report build differs only in layout (`\ifreportmode`).

Rebuild after any tex edit:

```bash
bash docs/paper/build/compile_report.sh
```

## Must ship Friday

| Item | Standard |
|------|----------|
| **Main PDF** | `docs/paper/plos_my_paper_report.pdf` — typos skimmed, consistent notation |
| **Word budget** | **3,000–3,500** main text (max **4,000**) — [`word_budget.md`](word_budget.md); every section at target |
| **Intro / Results / Discussion** | Expanded per [`manuscript_scope_plan.md`](manuscript_scope_plan.md) §4 — not stub paragraphs |
| **Fig 1–4 + S1–S3** | Final or best available data; no “pending” in captions unless experiment truly blocked (then one honest limitation sentence in Discussion, not banner text) |
| **Holdout ladder** | cHMM vs cGMVAE complete; HMMGMVAE rung finished or explicitly excluded with reason in Methods/Results |
| **K-sweep** | Real S3 curve + Figs 3–4 at chosen **K** (not macro K=3 placeholder) |
| **Methods** | Standalone-readable; thesis delta (decoder-only) clear |
| **Limitations** | Label noise, lab heterogeneity, seed fragility — inline in Results/Discussion, not a separate “gaps” box |

## Pending HPC jobs (refresh on Friday)

Snapshot **2026-06-11** — check live queue before acting:

```bash
bjobs -u $USER
bjobs -u $USER | egrep 'hmm_raw|seed_rerun|k_sweep|kx2'
```

### HMM raw (10 PEND + 3 RUN) — finish incomplete holdout grid

| Scope | Cells still needed |
|-------|-------------------|
| Within-lab | f3 lab_3, lab_5; f4 lab_2, lab_3, lab_5 |
| Spill duplicates | f3 lab_5, f4 lab_2, f4 lab_5 (safe to `bkill` if primary finishes first) |

**RUN (3):** `cv4_hmm_raw_f1_lab_3`, `cv4_hmm_raw_joint_f2_spill`, `cv4_hmm_raw_f4_lab_3_spill`

**When done — Friday updates:**

- [ ] `PYTHONPATH=. python3 scripts/cv4fold/collect_hmm_raw_holdout.py`
- [ ] Regenerate holdout tables: `PYTHONPATH=. python3 scripts/paper/build_holdout_scope_table.py` (if used in tex)
- [ ] Update `docs/cv4fold/holdout_results_status.md` grid (target **16/16** cells)
- [ ] Add HMM (raw) rung to Fig 2 / Table 2 where data exist; `---` only if still missing
- [ ] Kill redundant `_spill` / duplicate jobs once `results.npz` exists for that cell

Logs: `hpc/output/cv4fold/hmm_raw/*.out`

### VAE seed reruns (7 PEND) — fill missing 3rd seeds

| Job family | Targets |
|------------|---------|
| Primary (v100) | `cv4_seed_rerun_f1_lab3_cgmvae`, `f3_lab5_cgmvae`, `f4_lab5_cgmvae`, `f4_lab5_chmmgmvae` |
| Spill mirrors | f4 cgmvae, f3 lab5 cgmvae, f4 lab5 chmmgmvae (a10/a40) |

**When done — Friday updates:**

- [ ] `PYTHONPATH=. python3 scripts/paper/summarize_holdout_ladder.py`
- [ ] `PYTHONPATH=. python3 scripts/paper/scrape_holdout_accuracy.py`
- [ ] Confirm within f4 lab_5 chmmgmvae has **3/3** seeds; refresh `holdout_ladder.csv`
- [ ] Kill spill mirrors if primary seed run finished

### K-sweep fold 4 (26 PEND, all v100)

| Batch | Jobs |
|-------|------|
| `cv4_kx2_f4_K3` … `K15` | 13 jobs (extra seeds / x2 batch) |
| `cv4_k_sweep_pop_K3` … `K15` | 13 jobs (population sweep) |

**When done — Friday updates:**

- [ ] `PYTHONPATH=. python3 scripts/paper/plot_k_sweep_dual_axis.py` → `docs/paper/figures/main/S3_k_sweep.pdf`
- [ ] `PYTHONPATH=. python3 scripts/paper/run_biology_meeting_pack.py` if refreshing K panels
- [ ] Lock chosen **K** in Results + Methods; update Fig 3–4 paths if best K changed
- [ ] Rebuild report: `bash docs/paper/build/compile_report.sh`

Logs: `hpc/output/cv4fold/` (k-sweep submit scripts)

---

## Build & verify (before hand-in)

```bash
cd docs/paper && texcount -inc -1 plos_my_paper.tex
bash docs/paper/build/compile_report.sh
```

- [ ] Every figure cited before it appears; caption panels match text
- [ ] Every numeric claim traceable to table, figure, or cited thesis value
- [ ] Holdout protocol in Methods **and** Fig 1B
- [ ] K selection rule stated even if sweep was tight
- [ ] `texcount` main text ≤ 4,000 (target 3,000–3,500)

## Cover note (optional)

> Special-course article: zero-shot cross-lab holdout with decoder-only cHMM–GMVAE (mean NMI 0.56 vs cGMVAE 0.33, joint 4 folds). Full methods and SI pointers in repo `docs/paper/`.

## After Friday (PLOS packaging only)

Strip report-only macros, add Author summary, caption-only `plos_my_paper.pdf`, export `Fig*.tif`, Zenodo tag — not deferred article writing.
