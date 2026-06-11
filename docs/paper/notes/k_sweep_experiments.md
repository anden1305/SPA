# K substage sweep (cHMMGMVAE)

**Added 2026-06-09.** Vary `num_gmm_states` on locked joint holdout cHMM to pick K for main Fig 3. Thesis K=13 analysis lives in supplement.

---

## Generate

```bash
source .venv/bin/activate
PYTHONPATH=. python3 scripts/cv4fold/generate_chmm_k_sweep.py --fold 4
```

Configs: `src/config/run/cvaemarhmm/cv4fold/paper_k_sweep/fold_4/chmmgmvae_K{K}.yaml`  
Default K: 3, 4, 5, …, 15 (every integer).

---

## Submit

```bash
bash hpc/submit/cv4fold/submit_chmm_k_sweep.sh
```

`hpc/output/cv4fold/paper_k_sweep/k_sweep_%J.out`

Base sweep: `runs: 3` (seeds 124–126). Supplemental **2 seeds** only:

```bash
PYTHONPATH=. python3 scripts/cv4fold/generate_chmm_k_sweep_extra_seeds.py --fold 4
bash hpc/submit/cv4fold/submit_chmm_k_sweep_extra2_v100.sh
```

Logs: `hpc/output/cv4fold/paper_k_sweep/k_extra2_K*_*.out`  
Plotting merges metrics across base + `*_extra2` run dirs (5 seeds total).

---

## Pick K for biology

**Review pack** (all K curves + per-K biology panels, best seed each):

```bash
PYTHONPATH=. python3 scripts/paper/run_k_sweep_review_pack.py
```

Output: `results/cv4fold/paper_figures/k_sweep_review/` (+ `docs/paper/figures/professor_meeting/` copies).

1. Plot validation NMI + log-likelihood vs K (reuse thesis Fig 23 logic).
2. Prefer K with stable cross-seed NMI and non-collapsed `entropy_norm`.
3. Generate Fig-27 panel at chosen K:

```bash
PYTHONPATH=. python3 scripts/paper/plot_figure27_compact.py \
  --npz results/cv4fold/paper_k_sweep/fold_4/K11/<run_ts>/plots/<best_run>/results.npz \
  --out paper/overleaf/figures/figure3_substage_panels.pdf
```

4. Schedule [birgitte_interview_guide.md](birgitte_interview_guide.md).

**Population (all 20 mice, appendix biology):** [k_sweep_population.md](k_sweep_population.md)
