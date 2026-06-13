# Population K-sweep (all 20 mice, appendix biology)

**Added 2026-06-11.** Incohort substage discovery complement to fold-4 **holdout** K-sweep. Same locked cHMM–GMVAE recipe; train and val both use all 20 HQ mice (42 runs).

## Purpose

- Richer Fig 27 panels (more active substates at high K)
- Expert / appendix substage naming (Birgitte)
- **Not** used to pick K for the main holdout generalization claim

## Generate configs

```bash
PYTHONPATH=. python3 scripts/cv4fold/generate_chmm_k_sweep_population.py
# Holdout-style (no artifact epochs): add --no-artifact
```

Default keeps artifact epochs (`remove_artifact: false`). `--no-artifact` sets `remove_artifact: true` on all mice (same as fold-4 K-sweep).

Configs: `src/config/run/cvaemarhmm/cv4fold/paper_k_sweep/population/chmmgmvae_K{K}.yaml`  
`runs: 5`, `seed: 499`, K=3…15. **Artifact epochs retained** (`remove_artifact: false` on all mice).

## Submit (gpuv100)

```bash
bash hpc/submit/cv4fold/submit_chmm_k_sweep_population.sh
```

**Spill backups** (9 jobs: K7–K15, one per K rotated a100/a10/l40s; seeds 599/699/799):

```bash
bash hpc/submit/cv4fold/submit_chmm_k_sweep_population_spill.sh
```

**OOM retry** (gpua100, -n 4, 8GB/slot, batch 64, top-up seeds for partial K):

Regenerate configs before submit:

```bash
PYTHONPATH=. python3 scripts/cv4fold/generate_chmm_k_sweep_population_retry.py
```

```bash
bash hpc/submit/cv4fold/submit_chmm_k_sweep_population_retry.sh
```

Configs: `paper_k_sweep/population_retry/` (same `results_dir` as `population/`).

`hpc/output/cv4fold/paper_k_sweep/population/k_sweep_K*_*.out`

## Figures (after training)

Full pack (Fig 27, t-SNE, PCA, transitions, …):

```bash
bsub < hpc/submit/paper/run_population_biology_meeting.sh
```

**Incremental** (Fig 27 + t-SNE only, for K that already have `results.npz` — safe while GPU jobs still run):

```bash
bsub < hpc/submit/paper/run_population_biology_meeting_incremental.sh
```

Per-K outputs include `K{k}_tsne_scatter_true.png` and `K{k}_tsne_scatter_predicted.png` (shared embedding).

Output: `results/cv4fold/paper_figures/biology_meeting_population/`  
Staging: `docs/paper/figures/professor_meeting_population/`

Holdout pack (unchanged): `results/cv4fold/paper_figures/biology_meeting/`
