# Incohort ablation findings (2026-06-08)

**Added 2026-06-08.** Synthesis of overnight cHMM incohort, seq-length, enc+dec conditioning, and latent-separability analysis. Use this to lock recipes before **per-lab holdout (fold 4)**.

Related: [ablation_chmm_incohort.md](ablation_chmm_incohort.md), [cgmvae_incohort_analysis_20260608.md](cgmvae_incohort_analysis_20260608.md), [latent_separability_guide.md](../latent_separability_guide.md), [incohort_to_cross_lab_strategy.md](../incohort_to_cross_lab_strategy.md), [`locked_recipes.py`](../../../scripts/cv4fold/locked_recipes.py).

---

## Executive summary

| Lab | Ship model | Prior / T | Prepro | Best incohort NMI | vs cGMVAE | Stability |
|-----|------------|-----------|--------|-------------------|-----------|-----------|
| **lab_2** | **cGMVAE** | GMM, T=1 | `rem_emg_wide_eeg4` | **0.593** | — | 3/3 healthy |
| **lab_3** | **cHMM** | **warm**, T=64 | `baseline_long` + `wide_mlp` | **0.734** (warm) | −0.003 vs 0.737 | 3/3 warm healthy |
| **lab_5** | **cHMM** | hmm_gmm, **T=32** | `emg_wide_notch50` (pending lock) | **0.640** (seq32 s2) | +0.10 vs 0.534 | **1/3** lucky seed — re-run reliability |

**Do not run more seq-length ablations.** Move to holdout with locked T per lab. Optional: 10-seed reliability on lab_5 seq32 before thesis lock.

**Conditioning:** keep `decoder_only_conditioning: true` (enc+dec ablation worse on lab_2; lab_3/5 incomplete but trending down).

---

## cHMM Phase 1/2 — prior tier (seq=1)

cGMVAE targets: lab_2 **0.593**, lab_3 **0.737**, lab_5 **0.534**.

| Lab | hmm_gmm best [s1,s2,s3] | warm best [s1,s2,s3] | Lock | Notes |
|-----|-------------------------|----------------------|------|-------|
| lab_2 | **0.569** [0.561, 0.569, 0.541] | 0.540 [0.540, 0.458, 0.509] | **simple** (both below cGMVAE) | notch50_warm 0.571 — still < cGMVAE; **stay cGMVAE-first** |
| lab_3 | 0.694 [0.694, 0.529, 0.564] | **0.734** [0.726, 0.576, 0.734] | **warm** | warm wins +0.04; ≥2/3 healthy |
| lab_5 | 0.508 [0.356, 0.480, 0.508] | 0.532 [0.508, 0.532, 0.414] | **simple** | weak seeds; prepro delta helps |

Code: `LOCKED_CHMM_PRIOR_TIER` updated — lab_3 → `warm`.

---

## cHMM Phase 3 — prepro deltas (lab_5, lab_2)

| Lab | Variant | Best-of-3 | vs baseline hmm_gmm |
|-----|---------|-----------|---------------------|
| lab_5 | `emg_wide_notch50` + simple | **0.539** [0.497, 0.539, 0.536] | +0.03 |
| lab_5 | `emg_wide` + simple | 0.538 | tie |
| lab_2 | `notch50` + warm | 0.571 | still < cGMVAE 0.593 |

**lab_5 prepro lock candidate:** EMG 3–100 Hz + 50 Hz notch on top of `long` + `wide_mlp`. Confirm with followup job `emg_wide_notch50` if still pending.

---

## Phase 4 — sequence length (locked prepro, hmm_gmm; lab_3 warm not yet re-run at T>1)

Compare **best-of-3** prior NMI; flag collapsed seeds (NMI ≈ 0, 1 predicted state).

| Lab | T=1 ref | T=32 | T=64 | T=128 | Lock T |
|-----|---------|------|------|-------|--------|
| lab_2 | 0.569 (cHMM) / **0.593 (cGMVAE)** | 0.539 (2/3 collapse) | 0.546 (2/3 collapse) | 0.548 (2/3 collapse) | **1** (cGMVAE) |
| lab_3 | 0.694 (hmm) / **0.734 (warm)** | 0.598 | **0.696** | TBD | **64** (with warm prior) |
| lab_5 | 0.539 (notch50) | **0.640** [0.488, **0.640**, 0.507] | 0.571 | 0.536 | **32** (reliability run first) |

**lab_2:** long sequences → HMM collapse on 2/3 seeds; do not use cHMM at T>1 for this lab.

**lab_5:** T=32 seed 2 drives headline 0.640; treat as hypothesis until ≥2/3 seeds healthy.

Results: `results/cv4fold/ablation_chmm_seq{T}/{lab}/`

---

## Enc+dec conditioning (cGMVAE)

Scratch, `decoder_only_conditioning: false`, locked prepro/arch, seq=1.

| Lab | Dec-only ref | enc+dec best [seeds done] | Verdict |
|-----|--------------|---------------------------|---------|
| lab_2 | **0.593** | 0.578 [0.578, 0.532, 0.513] | **Worse** — keep dec-only |
| lab_3 | **0.737** | 0.621 [1/3 seeds] | Incomplete; trending down |
| lab_5 | **0.534** | 0.430 [2/3 seeds] | Incomplete; trending down |

Results: `results/cv4fold/ablation_conditioning/{lab}/`

---

## Latent separability — what the plots mean

**“Feature N” in amplitude / separability plots = latent dimension N** (encoder or HMM posterior μ), **not** an input channel. See [latent_separability_guide.md](../latent_separability_guide.md).

### lab_5 seq32 seed 2 (NMI 0.640) — Feature 7 / REM

Example: `results/cv4fold/ablation_chmm_seq32/lab_5/abl_chmm_lab_5_hmm_gmm_locked_seq32_20260608-074558/plots/2/feature_amplitude_per_state.png`

| Latent dim | Role (seed 2) | Mechanism |
|------------|---------------|-----------|
| **Feature 7** | REM axis | REM μ ≈ +2.6; Awake/NREM ≈ −0.5 to −0.7; tight variance → clean REM gap |
| **Feature 4** | NREM axis | NREM ≈ −1.8; Awake ≈ +0.7; secondary separator |
| Features 1–3, 5–6 | Near-degenerate | Normalization anchors; little class info |

**Seed-specific:** seed 1 (NMI 0.488) has flat Feature 7; seed 3 (0.507) similar. The 0.640 headline is **one initialization**, not a stable recipe property.

**Replication levers (not “pick input feature 7”):**

1. Lock T=32 + EMG-wide+notch prepro for lab_5 cHMM.
2. Run ≥10 seeds; require ≥2/3 healthy before trusting best-of-3.
3. After training, inspect `plots/<best>/feature_amplitude_per_state.png` on **prior μ** for a REM-isolated dim.
4. Optional: REM-recall checkpoint selection (`ablation_incohort_followup` / `rem_recall_ckpt`).

### Other separability wins

| Run | Pattern | Takeaway |
|-----|---------|----------|
| **lab_3 warm** | Feature **4** dominates (Fisher ~3.8 on prior μ) | Warm prior stabilizes GMM→HMM handoff |
| **lab_5 seq32 s2** | Multi-dim REM (F7) + NREM (F4) | Prefer multi-axis structure over single high-variance dim |
| **lab_5 seq1** | Single dim REM spike, high variance | Less stable than seq32 multi-dim layout |
| **lab_2** | Input EMG/atonia gap exists; cHMM does not beat cGMVAE | Fix input/prepro before prior/arch |

**Pre vs post training:** root `data_validations.json` latent Fisher rankings reflect the **last seed’s** post-train encoder; they can disagree with the best seed’s prior-μ amplitude plot. Use **`plots/<best>/`** after the plot pipeline fix (below).

---

## Plot pipeline fix (2026-06-08)

**Problems observed:** latent separability only at `plots/` root for seed 1; `losses.png` missing (CVAE path skips epoch predictions); root diagnostics overwritten by last seed, not best NMI.

**Fixes** ([`orchestrator.py`](../../../src/orchestrator/orchestrator.py), [`visualizer.py`](../../../src/visuals/visualizer.py), [`locked_recipes.py`](../../../scripts/cv4fold/locked_recipes.py)):

- `ensure_cv4fold_diagnostics()` — `validate_data` + separability flags on all generated configs.
- Every seed → `plots/<n>/` gets encoder + **prior-μ** amplitude, variance, latent separability.
- `plots/<n>/losses.png` from trainer even when epoch predictions missing.
- `plots/` root copies diagnostics from **best prior-NMI** seed.

Details: [latent_separability_guide.md](../latent_separability_guide.md#two-spaces--do-not-mix-them).

---

## Recommended locks (for holdout generator)

| Lab | Model | `LOCKED_SOURCES` prepro | Arch | Prior | `sequence_length` | Other |
|-----|-------|-------------------------|------|-------|-------------------|-------|
| lab_2 | cgmvae | `rem_emg_wide_eeg4` | default | gmm | 1 | — |
| lab_3 | chmmgmvae | `baseline_long` | `wide_mlp` | **warm** | **64** | — |
| lab_5 | chmmgmvae | **`emg_wide_notch50`** (confirm) | `wide_mlp` | hmm_gmm | **32** | 300 ep; 10-seed reliability optional |

**Still TODO in code:** point `LOCKED_SOURCES[lab_5].prepro` at EMG-notch YAML when followup confirms; add `sequence_length` to holdout generator per lab.

Regenerate holdout:

```bash
PYTHONPATH=. python3 scripts/cv4fold/generate_configs.py --phase per_lab_holdout --fold 4
bash hpc/submit/cv4fold/submit_per_lab_holdout.sh
```

Use `save_results_npz: true` + postprocess `--per-mouse-metrics` on holdout ([incohort_to_cross_lab_strategy.md](../incohort_to_cross_lab_strategy.md)).

---

## Next steps (priority order)

1. **Wait** for pending jobs (enc+dec lab_3/5, seq128 lab_3, followup EMG/notch, wide_mlp, rem_recall_ckpt).
2. **Optional:** 10-seed reliability — lab_5 cHMM seq32 + emg_wide_notch50.
3. **Holdout fold 4** — locked recipes above; per-mouse NMI gate.
4. **Do not** re-sweep seq length or full joint 4-fold Bayes on current branch.
5. **After holdout gate** — small ablations from old `chmmgmm_fold` sweeps (see below).

---

## Old joint-fold Bayes sweeps → targeted ablations (after holdout)

Previous sweeps on `chmmgmm_fold` used **all folds + `subject_lab`** — not directly comparable to locked per-lab scratch line.

| Sweep winner | Notable params | Current locked line | Ablate after holdout? |
|--------------|----------------|---------------------|------------------------|
| chmm **5bkceef8** (disk 0.560) | latent_dim=4, emb_dim=8, lr≈1.3e-3, cyclical β | dim=8, emb=4, lr=3e-4, fixed β | **latent_dim** {4,8,16}, **lr** log grid, cyclical β (low priority) |
| cgmvae **8duimqhc** (disk 0.513) | latent_dim=16, lr≈1.2e-5, anneal β, num_batches=256 | dim=8, lr=3e-4 | **latent_dim**, **lr**, batch for **lab_2 cGMVAE only** |
| Both | Many trials NMI≈0 | — | Always require ≥2/3 healthy seeds |

**Skip until holdout:** `subject_lab` conditioning, full joint re-sweep, architecture beyond locked `wide_mlp`.

---

## Scrape commands

```bash
PYTHONPATH=. python3 scripts/cv4fold/scrape_experiment_results.py --search-root results/cv4fold/ablation_chmm
PYTHONPATH=. python3 scripts/cv4fold/scrape_experiment_results.py --search-root results/cv4fold/ablation_chmm_seq32
PYTHONPATH=. python3 scripts/cv4fold/scrape_experiment_results.py --search-root results/cv4fold/ablation_conditioning
```
