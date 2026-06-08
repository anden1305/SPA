# cHMM-GMVAE incohort ablations

**Added 2026-06-07.** Scratch-only incohort sweep for cHMM-GMVAE on **locked cGMVAE prepro/arch** per lab. **No checkpoint hotstart** (`model_checkpoint_path: null` always).

**Goal:** Lock per-lab cHMM recipe — prefer **`prior: hmm_gmm`** unless **`warm_hmm_gmm`** clearly wins (+0.01 best-of-3 NMI, stable seeds).

Related: [lab_preprocessing_review_20260607.md](../lab_preprocessing_review_20260607.md), [ablation_lab2_findings_20260607.md](ablation_lab2_findings_20260607.md), Phase 1 decoder-only [decoder_only_lab3_chmm_experiments.md](../../decoder_only/decoder_only_lab3_chmm_experiments.md) (informational; hotstart jobs excluded here).

---

## cGMVAE targets (best-of-3 incohort)

| Lab | Locked recipe | cGMVAE NMI |
|-----|---------------|------------|
| lab_2 | `rem_emg_wide_eeg4` | **0.593** |
| lab_3 | `baseline_long` + `wide_mlp` | **0.737** |
| lab_5 | `long` + `wide_mlp` | **0.534** |

---

## Prior tiers

| Tier | `prior` | When to lock |
|------|---------|--------------|
| **simple** (default) | `hmm_gmm` | Best-of-3 ≥ warm or within **+0.01** tie band |
| **warm** | `warm_hmm_gmm` | Best-of-3 **> simple + 0.01**, ≥2/3 healthy seeds |

Warm schedule: GMM warmup 18 ep → HMM warmup 37 ep → transition ramp 18 ep, sticky κ=0.86 ([`locked_recipes.py`](../../scripts/cv4fold/locked_recipes.py)).

**Not in scope:** loading baseline `.pth` checkpoints (decoder-only hotstart).

---

## Phase 1 & 2 — baseline (6 jobs)

Same locked prepro/arch as cGMVAE; only prior differs.

| Variant | Prior | Config |
|---------|-------|--------|
| `hmm_gmm_locked` | `hmm_gmm` | `ablation_chmm/{lab}/hmm_gmm_locked.yaml` |
| `warm_hmm_gmm_locked` | `warm_hmm_gmm` | `ablation_chmm/{lab}/warm_hmm_gmm_locked.yaml` |

**Generate (full sweep — baselines + prepro deltas × both priors):**

```bash
PYTHONPATH=. python3 scripts/cv4fold/generate_ablation_chmm_incohort.py --phase all
bash hpc/submit/cv4fold/submit_ablation_chmm_sweep.sh
```

**12 jobs:** 6 baselines (`hmm_gmm` + `warm_hmm_gmm` × 3 labs) + 6 prepro deltas (lab_2 notch50, lab_5 EMG wide/notch50 × both priors). **lab_5 uses 300 epochs, 6:00 walltime**; lab_2/lab_3 use 200 ep, 6:00.

**Generate (baselines only):**

```bash
cd /work3/s204070/SPA && source .venv/bin/activate
PYTHONPATH=. python3 scripts/cv4fold/generate_ablation_chmm_incohort.py --phase all_baselines
```

**Submit** (run with `bash`, not `bsub <`):

```bash
bash hpc/submit/cv4fold/submit_ablation_chmm_baseline.sh all
# or: simple | warm
```

Logs: `hpc/output/cv4fold/ablation_chmm/{lab}_hmm_gmm_%J.out` / `{lab}_warm_%J.out`  
Results: `results/cv4fold/ablation_chmm/{lab}/abl_chmm_{lab}_{variant}_*`

**Metric:** best-of-3 **HMM-GMM prior NMI** from `plots/{1,2,3}/metrics.txt`. Keep `no_beta_epochs: 10`.

---

## Phase 3 — prepro deltas (conditional)

Run only when locked prior from Phase 1/2 **still trails cGMVAE NMI** for that lab.

| Lab | Delta | cGMVAE source |
|-----|-------|---------------|
| lab_5 | EMG 3–100 Hz | `wide_mlp_emg_wide` |
| lab_5 | EMG wide + 50 Hz notch | `wide_mlp_emg_wide_notch50` |
| lab_2 | 50 Hz notch | `rem_emg_wide_eeg4_notch50` |
| lab_3 | — | skip (prepro already optimal) |

**Generate** (default prior: `hmm_gmm`; add `--prior-tier both` if warm won):

```bash
PYTHONPATH=. python3 scripts/cv4fold/generate_ablation_chmm_incohort.py --phase prepro_delta --labs lab_5 --prior-tier simple
PYTHONPATH=. python3 scripts/cv4fold/generate_ablation_chmm_incohort.py --phase prepro_delta --labs lab_2 --prior-tier simple
```

**Submit:**

```bash
bash hpc/submit/cv4fold/submit_ablation_chmm_prepro_delta.sh
```

---

## Lock & holdout

After Phase 1/2 (and optional Phase 3), set per-lab tier in [`LOCKED_CHMM_PRIOR_TIER`](../../scripts/cv4fold/locked_recipes.py) (`simple` or `warm`), then regenerate holdout:

```bash
PYTHONPATH=. python3 scripts/cv4fold/generate_configs.py --phase per_lab_holdout --fold 4 --models chmmgmvae
bash hpc/submit/cv4fold/submit_per_lab_holdout.sh --models chmmgmvae
```

Holdout uses the same prior tier as incohort lock — scratch only.

---

## Phase 4 — sequence length (after prior/prepro lock)

At **seq=1**, HMM transitions are inactive. After Phase 1–3 pick the **locked prior + prepro**, compare **T ∈ {32, 64, 128}** with that recipe only — **9 jobs** (3 labs × 3 lengths), not a full prepro matrix per T.

```bash
# Default: hmm_gmm + locked cGMVAE prepro (change --prior-tier warm if warm won)
PYTHONPATH=. python3 scripts/cv4fold/generate_ablation_chmm_incohort.py \
  --phase seq_length_compare --sequence-lengths 32 64 128 --prior-tier simple
bash hpc/submit/cv4fold/submit_ablation_chmm_seq_length_compare.sh
```

Walltime: seq32 8h/12h/12h · seq64 12h/24h/24h · seq128 24h all labs.

Configs: `ablation_chmm_seq{T}/{lab}/hmm_gmm_locked_seq{T}.yaml`  
Results: `results/cv4fold/ablation_chmm_seq{T}/{lab}/`

Pick best **T** per lab (best-of-3 NMI), then lock prior + prepro + T for holdout.

**Cross-lab cv4fold:** after locks are set, read [incohort_to_cross_lab_strategy.md](../incohort_to_cross_lab_strategy.md) for how incohort results feed per-lab holdout and joint training.

---

## Compare table (updated 2026-06-08)

| Lab | cGMVAE | hmm_gmm | warm_hmm_gmm | Prior lock | Seq T | Next |
|-----|--------|---------|--------------|------------|-------|------|
| lab_2 | **0.593** | 0.569 | 0.540 | **cGMVAE** (cHMM < cGMVAE) | 1 | holdout cGMVAE |
| lab_3 | 0.737 | 0.694 | **0.734** | **warm** | **64** | holdout cHMM warm T=64 |
| lab_5 | 0.534 | 0.508 | 0.532 | **simple** | **32** | emg_wide_notch50 + reliability |

Full analysis (latent dims, enc+dec, plot pipeline): [ablation_findings_20260608.md](ablation_findings_20260608.md).

Scrape: `PYTHONPATH=. python3 scripts/cv4fold/scrape_experiment_results.py --search-root results/cv4fold/ablation_chmm`
