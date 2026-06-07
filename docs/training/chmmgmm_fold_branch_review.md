# `chmmgmm_fold` branch review — migration guide for `vae_decoder_chmm_cv4`

**Added:** 2026-06-07  
**Scope:** Read-only comparison. No code was merged; this doc records what is worth cherry-picking later.

## Context

| | |
|---|---|
| **Current branch** | `vae_decoder_chmm_cv4` (6 commits ahead of merge base) |
| **Review branch** | `chmmgmm_fold` (9 commits ahead of merge base) |
| **Common ancestor** | `d926c56` — *Add MAR data synthetic data configurations* |

Both branches diverged from the same ~2‑month-old point. You rebuilt `vae_decoder_chmm_cv4` iteratively; `chmmgmm_fold` accumulated parallel QoL, conditioning, and infrastructure work. A full merge would touch **~479 files** (+50k lines) and would conflict with ablation configs, validation viz, and submit layout that already work on the current branch.

**Bottom line:** Treat `chmmgmm_fold` as an **idea backlog**, not a merge target. Copy small, isolated pieces; avoid wholesale config/submit tree imports.

---

## What `chmmgmm_fold` changed (summary)

### Main themes

1. **Lab / subject conditioning**
   - `model.conditioning_source`: `subject` \| `lab` \| `subject_lab`
   - `DataLoaderCollection`: lab maps, paired subject+lab IDs, MSSV expansion by lab + quality filter
   - `ConditionalVAE`: `lab_emb`, combined conditioning dim, decoder-only conditioning paths
   - Large YAML + submit trees under `cvaeprior/lab_conditioning/`, `decoder_only_lab_2/`, `decoder_only_lab_5/`

2. **CNN raw-window front-end (Story A)**
   - `feature_pipeline: raw_cnn` vs `fft`
   - New modules: `temporal_front.py`, `raw_window_preprocessing.py`
   - VAE changes: learnable CNN front, optional time-domain recon, FFT anchor loss

3. **Validation & batch tooling**
   - `main.py` methods: `validate_cvae_hmm`, `validate_cvae_all_runs`
   - `Orchestrator.validate_all_cvae_runs()`, reliability summary writers
   - `scripts/hmmgmm_validate_checkpoint.py`
   - cv4fold postprocess: `postprocess_fold.py`, `split_per_mouse.py`, `aggregate_val_nmi_by_lab.py`, `select_best_vae_run.py`, summarize scripts

4. **Training / checkpoint behaviour**
   - Thesis composite `checkpoint_score` + W&B logging (shared with current — see below)
   - **During training:** saves `cvae_best_kmeans_nmi.pth` (pre-collapse), not `prior_pred_nmi`
   - `pretrained_checkpoint_path` separate from `model_checkpoint_path` (read-only hotstart vs write target)
   - Non-finite loss / grad-norm guards in `Trainer`

5. **W&B sweep infrastructure**
   - Reworked `wandb_sweep_runner.py`: grid vs Bayes mode, broader sweep key map, `max_beta` / `num_batches`
   - Bayes50 sweep YAMLs for cv4fold joint tuning
   - `save_results_npz` toggle to skip large `results.npz` on sweeps

6. **Preprocessing audit pipeline**
   - Five-phase audit under `scripts/preprocessing_audit/` + `hpc/submit/preprocessing_audit/`
   - Docs: `preprocessing_audit_*.md`

7. **Visualization**
   - `pca_tripanel.py`, `visualize_hmmgmm_trajectory`, structured `results.npz` helpers
   - **Removed** input/latent separability plots that current branch added later

8. **Config / repo hygiene**
   - Typo fix: `traning_pipeline` → `training_pipeline` everywhere
   - Config tree moved to `src/config/run/cvaeprior/` (from `cvaemarhmm/`)
   - `validate_train`, `max_batches_per_epoch` config knobs
   - Expanded pytest suite (`test_cnn_front_vae`, `test_vae_hmmgmm_integration`, etc.)

### Minor / QoL on `chmmgmm_fold`

| Area | Change |
|------|--------|
| Config typo | `training_pipeline` spelling fixed in Python + all YAML |
| Trainer | Skip non-finite batches; empty-epoch loss fallback |
| Validator | `validate_train` gate on train-set NMI; `latent_autocorr` in epoch metrics |
| Data | Optional `lab`, `quality_filter`, `signals` on dataset entries; `max_batches_per_epoch` cap |
| Orchestrator | `_remap_subject_embedding_state`, `_expand_mssv_lab_configs`, cleaner pretrained load |
| Helpers | `model_display_name.py` |
| Noise | `scripts/noise/main.py` coordinator (labs 2/3/5 scripts exist on both branches) |
| HPC | `run_data_processing.sh`, cv4fold bsub helpers, tune-winner submit scripts |
| Docs | `HPC_SUBMIT_GUIDE.md`, `cvae_wandb_checkpoint_metrics.md`, conditioning / CNN architecture docs |
| Tests | `conftest.py`, dev pytest group in `pyproject.toml` |

---

## What `vae_decoder_chmm_cv4` already has (keep — do not regress)

These are **newer or better** on the current branch; `chmmgmm_fold` would overwrite or delete them.

| Feature | Notes |
|---------|--------|
| **Ablation program** | Prepro, REM, signals, arch configs + submit scripts + findings docs (`ablation_*`, `overnight_experiments_20260606.md`) |
| **Input / latent diagnostics** | `input_channel_statistics.py`, compact distinctness, separation-gap plots, input vs latent comparison in `Visualizer` / `Validator` |
| **`prior_pred_nmi` training checkpoint** | Saves `cvae_best_prior_pred_nmi.pth` — aligned with thesis CV metric; `chmmgmm_fold` tip saves KMeans instead |
| **`validation_tag`** | Multi-seed validation loads correct checkpoint + subfolder under `plots/` |
| **`validate_decoder_only_all.py`** | Batch validation under `results/decoder_only/` (overlaps partially with `validate_cvae_all_runs`) |
| **Preprocessing ablation knobs** | `robust_normalize`, `append_channel_rms` in config — **removed on `chmmgmm_fold`** |
| **Decoder-only submit layout** | `hpc/submit/decoder_only/{baseline,reliability}/` — cleaner than flat + lab_2/lab_5 trees |
| **cv4fold docs** | `latent_separability_guide.md`, interim findings, cross-lab story |
| **`887b6ed` fix** | `start_t` when sequence shorter than requested windows |
| **Checkpoint docs** | `docs/training/cvae_checkpointing.md` (SEM / scratch vs hotstart guidance) |
| **Agent / HPC rules** | `.cursor/rules/*` (discipline, bsub communication, document-changes) |

**Already shared (both branches, possibly with small diffs):** `hmm_gmm_prior.py`, core HMM-GMM prior training, `checkpoint_score` composite, noise scripts per lab, `generate_configs.py` / manifest utils, quality cohort manifest.

---

## Recommendations

### ✅ Copy when ready (low risk, clear value)

Port as **small cherry-picks**, not a merge.

| Item | Why | Risk |
|------|-----|------|
| **`training_pipeline` typo fix** | `traning_pipeline` still in current YAML + `config.py`; fix is mechanical | Low — run grep after rename |
| **Trainer non-finite guards** | Prevents bad batches from poisoning weights; no behaviour change on clean runs | Low |
| **`validate_train: bool`** | Skip expensive train-set NMI during sweeps | Low — default can stay `True` |
| **`max_batches_per_epoch`** | Smoke / debug without full epoch | Low |
| **`save_results_npz: bool`** | Saves disk on W&B sweeps | Low — **ported**; default `false` on cv4fold via `locked_recipes.py` |
| **cv4fold postprocess scripts** | `postprocess_fold.py`, … — **ported lean** | See [cv4fold/postprocess.md](../cv4fold/postprocess.md) |
| **`docs/training/cvae_wandb_checkpoint_metrics.md`** | Complements `cvae_checkpointing.md`; documents W&B keys + \(S(\varepsilon)\) | None |
| **`scripts/noise/main.py`** | One entry point to run all lab noise analyses | Low |
| **`latent_autocorr` in `validate_cvae_epoch`** | Cheap trajectory metric when `sequence_length > 1` | Low — **ported** |
| **Pytest dev group + selected tests** | `test_hmmgmm_checkpoint_score`, `test_warm_schedule`, `test_vae_prior_compat` — skip duplicating what you already have | Low — run on compute node |

### ❌ Do not copy (would bloat or break current work)

| Item | Why not |
|------|---------|
| **Full config tree move to `cvaeprior/`** | Breaks every current YAML path and submit script |
| **Removing `robust_normalize` / `append_channel_rms`** | Active ablation axes on current branch |
| **Replacing `prior_pred_nmi` checkpoint with KMeans-only** | Current choice matches thesis CV metric; KMeans peaks pre-collapse but is not the paper objective |
| **Replacing / stripping `Visualizer` separability plots** | Core to current ablation interpretation |
| **Lab conditioning YAML + submit forests** | Hundreds of configs for a **different experiment track** (Phase 2 conditioning); import only when you start that track intentionally |
| **`decoder_only_lab_2/` / `decoder_only_lab_5/` submit duplicates** | Superseded by cv4fold per-lab + ablation submit scripts |
| **Subject-lab tune-winner batch configs** | Tied to old sweep layout; current cv4fold ablation program replaced this workflow |
| **`.cursor/plans/*.plan.md`** | Planning artifacts, not runtime code |
| **Wholesale `Orchestrator` / `main.py` swap** | Loses `validation_tag`, `_validate_trained_cvae_prior`, current validation flow |
| **Full merge of `wandb_sweep_runner.py`** | Large rewrite; only needed if you restart Bayes50 tune sweeps |

### 🤔 Maybe (decide per goal)

| Item | Copy if… | Skip if… |
|------|----------|----------|
| **`conditioning_source` + lab embeddings** | You resume **decoder conditioning / LOLO lab holdout** (roadmap Phase 2) | Staying on per-lab incohort + ablations only |
| **`pretrained_checkpoint_path` split** | You want read-only hotstart path separate from `model_checkpoint_path` writes | Current single-knob checkpointing is enough (`docs/training/cvae_checkpointing.md`) |
| **CNN `raw_cnn` front-end** | You want Story A (learned spectral front) as a **dedicated experiment** | FFT + ablation prepro is the current winning path |
| **`validate_cvae_all_runs` in `main.py`** | You prefer orchestrator-native batch validation | `validate_decoder_only_all.py` already covers decoder-only |
| **W&B sweep runner Bayes mode** | You rerun cv4fold joint Bayes50 tuning | Manual ablation grid is sufficient |
| **`pca_tripanel` + trajectory viz** | You want HMM tripanel plots in cv4fold results layout | Current separability plots answer different questions |
| **Preprocessing audit pipeline** | You need cohort-level audit for thesis methods section | Ablation docs already capture lab-specific findings |
| **`HPC_SUBMIT_GUIDE.md`** | Useful reference for new submit scripts | Duplicates `.cursor/rules/hpc-*.mdc` partially |
| **`beta_schedule` / cyclical beta** | Exploring β scheduling beyond current anneal | Not needed until latent collapse becomes an issue again |
| **Broader pytest (`test_cnn_front_vae`, integration)** | After porting CNN front or large refactors | Before any CNN port — tests target code you do not have |
| **Tune sweep submit scripts** | Resuming `submit_tune_sweep.sh` / subject_lab winners | Current overnight ablation scripts are the active queue |

---

## Overlap and conflicts to watch

When cherry-picking, these files were edited on **both** sides — expect manual merge:

| File | Current branch emphasis | `chmmgmm_fold` emphasis |
|------|-------------------------|-------------------------|
| `src/models/vae.py` | HMM-GMM + existing FFT path | + lab conditioning, raw CNN, β cyclical |
| `src/validation/validator.py` | Input/latent distinctness, `prior_pred_nmi` | `validate_train`, `latent_autocorr`, slimmer data_validations |
| `src/training/trainer.py` | `prior_pred` checkpoint | KMeans checkpoint + non-finite guards |
| `src/orchestrator/orchestrator.py` | `validation_tag`, ablation-friendly validation | `validate_all_cvae_runs`, MSSV lab expand, pretrained split |
| `src/visuals/visualizer.py` | Separability / gap plots | Tripanel, npz helpers, less diagnostic plotting |
| `src/data/data_loader_collection.py` | Legacy 93-row subject emb, channel names | Lab/subject conditioning maps |
| `src/config/config.py` | Ablation prepro fields | `conditioning_source`, typo fix, CNN fields |
| `main.py` | `--validation_tag` | Extra validate methods without tag |

**Checkpoint policy difference (important):**

| Checkpoint file | `vae_decoder_chmm_cv4` | `chmmgmm_fold` |
|-----------------|------------------------|----------------|
| Thesis composite | `cvae_best_checkpoint_score.pth` | same |
| Per-epoch “best CV” | `cvae_best_prior_pred_nmi.pth` | `cvae_best_kmeans_nmi.pth` |

Keep the current **`prior_pred_nmi`** behaviour unless you explicitly want pre-collapse KMeans selection again.

---

## Suggested cherry-pick order (when you choose to port)

1. Typo fix (`training_pipeline`) + grep CI
2. Trainer non-finite guards
3. Config toggles: `validate_train`, `max_batches_per_epoch`, `save_results_npz`
4. cv4fold postprocess scripts (no training code touched)
5. `latent_autocorr` logging + `docs/training/cvae_wandb_checkpoint_metrics.md`
6. Pytest additions for anything you port
7. **Only then** consider larger tracks: conditioning, CNN front, preprocessing audit

---

## Commit map (`chmmgmm_fold`-only)

| Commit | One-line summary |
|--------|------------------|
| `3bd47c7` | Remove stale decoder-only subjectwise YAML |
| `8336df6` | Noise analysis scripts labs 2/3/5 |
| `9529526` | Lab conditioning configs + dataset/orchestrator conditioning |
| `13717de` | `run_data_processing.sh`, lab conditioning queue tweaks, data_processing robustness |
| `3ad7f59` | **`training_pipeline` typo fix** |
| `3755b87` | Lab conditioning walltime tweak |
| `d4c3d60` | `validate_cvae_hmm`, pytest dev deps, lab conditioning cleanup |
| `5b705eb` | `validate_cvae_all_runs`, gitignore, planning docs |
| `882e2b4` | CVAE validation metrics, checkpoint score, trainer W&B summary *(partially superseded on current)* |

## Commit map (`vae_decoder_chmm_cv4`-only)

| Commit | One-line summary |
|--------|------------------|
| `d3b37d8` | `validate_decoder_only_all.py`, decoder-only validation orchestration |
| `887b6ed` | **`start_t` fix** for short sequences |
| `0e9c4c4` | Submit script cleanup |
| `d3a9f28` | Agent / HPC / docs rules |
| `cd74a1f` | Checkpoint docs, paper cohort docs, ablation REM/signals |
| `451bb59` | Input/latent separability metrics + plots |

---

## Quick decision checklist

Before copying anything from `chmmgmm_fold`, ask:

1. **Does current branch already solve this?** (e.g. noise scripts, checkpoint score, validate-all)
2. **Does it touch ablation configs or prepro knobs?** → extra caution
3. **Is it a whole experiment tree (lab conditioning, CNN)?** → separate branch / PR, not drip merge
4. **Will it change which checkpoint file postprocess expects?** → update docs + scripts together

When in doubt, prefer **scripts + docs + trainer guards** over **model / orchestrator / config tree** changes.
