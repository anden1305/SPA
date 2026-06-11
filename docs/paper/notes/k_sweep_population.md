# Population K-sweep (all 20 mice, appendix biology)

**Added 2026-06-11.** Incohort substage discovery complement to fold-4 **holdout** K-sweep. Same locked cHMM–GMVAE recipe; train and val both use all 20 HQ mice (42 runs).

## Purpose

- Richer Fig 27 panels (more active substates at high K)
- Expert / appendix substage naming (Birgitte)
- **Not** used to pick K for the main holdout generalization claim

## Generate configs

```bash
PYTHONPATH=. python3 scripts/cv4fold/generate_chmm_k_sweep_population.py
```

Configs: `src/config/run/cvaemarhmm/cv4fold/paper_k_sweep/population/chmmgmvae_K{K}.yaml`  
`runs: 5`, `seed: 499`, K=3…15.

## Submit (gpuv100)

```bash
bash hpc/submit/cv4fold/submit_chmm_k_sweep_population.sh
```

`hpc/output/cv4fold/paper_k_sweep/population/k_sweep_K*_*.out`

## Figures (after training)

```bash
bsub < hpc/submit/paper/run_population_biology_meeting.sh
```

Output: `results/cv4fold/paper_figures/biology_meeting_population/`  
Staging: `docs/paper/figures/professor_meeting_population/`

Holdout pack (unchanged): `results/cv4fold/paper_figures/biology_meeting/`
