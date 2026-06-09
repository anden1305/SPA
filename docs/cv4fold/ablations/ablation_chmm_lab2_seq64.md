# lab_2 cHMM seq64 stability ablations (pruned)

**Added 2026-06-08.** Side experiment for **T=64** temporal cHMM on lab_2. Baseline `hmm_gmm_locked_seq64`: best **0.546**, seeds **[0, 0, 0.546]** (2/3 collapse).

**Only 3 jobs** — variants without incohort or sweep support were dropped.

---

## Run (3 configs)

| Config | Hypothesis | Source |
|--------|------------|--------|
| **`sweep_chmm_winner_seq64`** | Smaller latent + higher LR may stabilise long-seq HMM | Bayes sweep **5bkceef8** (disk NMI 0.560): `latent_dim=4`, `emb_dim=8`, `lr≈1.3e-3` |
| **`notch50_warm_seq64`** | Best **lab_2 cHMM incohort at seq=1** (0.571) — test if front-end + warm prior transfer to T=64 | `notch50_warm` ablation |
| **`beta_anneal_seq64`** | Gradual β ramp after `no_beta_epochs` may reduce early collapse before HMM learns transitions | cGMVAE sweep **8duimqhc** used anneal β; `beta_warmup_epochs=50`, `no_beta_epochs=10` |

Baseline for comparison (already run): `hmm_gmm_locked_seq64.yaml`.

---

## Dropped (not worth compute)

| Variant | Why dropped |
|---------|-------------|
| `warm_hmm_gmm` alone | lab_2 seq1 warm **0.540** < simple **0.569** — warm helps lab_3, not lab_2 |
| `notch50_simple` | seq1 best **0.497**, unstable seeds |
| `epochs300` | cGMVAE +0.005 at best (0.588 vs 0.593); no cHMM seq64 signal |
| `rem_recall_ckpt` | cGMVAE followup **collapsed** [0, 0.16, 0] |
| `cyclical β` | Not implemented in current codebase |
| `num_batches=256` | Deferred — test sweep winner first; add only if `sweep_chmm_winner` is healthy |

---

## Walltime (from completed LSF runs)

| Reference job | Run time | This submit |
|---------------|----------|-------------|
| lab_2 `hmm_gmm` **seq64** (28609769) | **16 min** | `sweep_*`, `beta_anneal`: **1:30** |
| lab_2 `notch50_warm` **seq1** (28609381) | **3.1 h** | `notch50_warm_seq64`: **2:30** (seq64 ≈4× faster on simple prior) |

Previous default **12:00** was ~8× too long for simple-prior seq64 jobs.

```bash
PYTHONPATH=. python3 scripts/cv4fold/generate_ablation_chmm_lab2_seq64.py
bash hpc/submit/cv4fold/submit_ablation_chmm_lab2_seq64.sh
```

Logs: `hpc/output/cv4fold/ablation_chmm_seq64/lab_2/<variant>_%J.out`  
Results: `results/cv4fold/ablation_chmm_seq64/lab_2/`

**Gate:** ≥2/3 healthy seeds and best-of-3 **> 0.546** before updating holdout lock.

**Winner (2026-06-08):** `sweep_chmm_winner_seq64` — best **0.576**, 3/3 healthy. Incohort **ceiling** reference only; paper main line uses unified recipe in [unified_holdout_paper_line.md](../unified_holdout_paper_line.md).
