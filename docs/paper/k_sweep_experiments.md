# K substage sweep (cHMMGMVAE)

**Added 2026-06-09.** Vary `num_gmm_states` on locked joint holdout cHMM to pick K for main Fig 3. Thesis K=13 analysis lives in supplement.

---

## Generate

```bash
source .venv/bin/activate
PYTHONPATH=. python3 scripts/cv4fold/generate_chmm_k_sweep.py --fold 4
```

Configs: `src/config/run/cvaemarhmm/cv4fold/paper_k_sweep/fold_4/chmmgmvae_K{K}.yaml`  
Default K: 3, 5, 7, 9, 11, 13, 15.

---

## Submit

```bash
bash hpc/submit/cv4fold/submit_chmm_k_sweep.sh
```

`hpc/output/cv4fold/paper_k_sweep/k_sweep_%J.out`

7 jobs × ~2h on `gpuv100` (mixed queue).

---

## Pick K for biology

1. Plot validation NMI + log-likelihood vs K (reuse thesis Fig 23 logic).
2. Prefer K with stable cross-seed NMI and non-collapsed `entropy_norm`.
3. Generate Fig-27 panel at chosen K:

```bash
PYTHONPATH=. python3 scripts/paper/plot_figure27_compact.py \
  --npz results/cv4fold/paper_k_sweep/fold_4/K11/<run_ts>/plots/<best_run>/results.npz \
  --out paper/overleaf/figures/figure3_substage_panels.pdf
```

4. Schedule [birgitte_interview_guide.md](birgitte_interview_guide.md).
