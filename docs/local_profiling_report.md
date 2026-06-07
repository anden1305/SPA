# Local cGMVAE profiling report (RTX 4060 Laptop)

**Run:** `profile_cvae_lab2_smoke` — lab_2 winner prepro, 5 epochs, batch 64, `validate_per_epoch: 5`  
**Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM), Windows, CUDA  
**Profile artifact:** `results/local_profile/profile_cvae_lab2_smoke_20260607-124714/1/`  
**Profiled wall time (cProfile):** 86.9 s (5 epochs, training loop only)

## Executive summary

Training is **GPU-bound** on conv encoder/decoder and autograd backward (~60% of profiled time). CPU-side costs are secondary: per-batch `loss.item()` GPU syncs, gradient clipping bookkeeping, one-shot W&B init, and a single end-of-run validation epoch (sklearn KMeans on full val tensor). Peak VRAM with this smoke config is **~442 MB** — batch 64 lab_2 cohort fits comfortably on 8 GB.

Four rounds of safe optimizations applied (see below) → **14.7% faster** GMM lab_2 profiled training (86.9 s → 74.2 s). cHMM-GMVAE paths: validation re-encode elimination + MARHMM encode-only forward. Remaining time is mostly GPU conv/backward.

---

## Top hotspots (before optimization)

Total profiled time: **86.914 s**. Percentages are `cumtime / 86.914` (cumulative; nested calls sum > 100%).

| # | Function | Cum (s) | Tot (s) | % | Category |
|---|----------|---------|---------|---|----------|
| 1 | `cvae_mar_hmm.py:36` `forward` | 28.9 | 0.04 | 33% | conv encoder / loss |
| 2 | `run_backward` (autograd) | 22.5 | 22.5 | 26% | conv encoder / loss |
| 3 | `vae.py:269` `forward` | 21.7 | 0.06 | 25% | conv encoder / loss |
| 4 | `vae.py:237` `decode` + `conv_transpose1d` | 13.0 + 8.9 | 8.9 | 15% + 10% | conv encoder |
| 5 | `validator.py:279` `validate_cvae_epoch` | 10.5 | — | 12% | validation (1× at epoch 5) |
| 6 | `vae.py:196` `encode` + `conv1d` | 8.1 + 3.5 | 3.5 | 9% + 4% | conv encoder |
| 7 | `vae.py:287` `calculate_loss` / `gmm_prior` | 7.1 + 5.8 | 2.4 | 8% + 7% | loss |
| 8 | `clip_grad.py:175` `clip_grad_norm_` | 6.1 | 0.22 | 7% | other (optimizer) |
| 9 | `adam.py:213` `step` | 5.6 | 0.06 | 6% | other (optimizer) |
| 10 | `wandb_init` (one-time) | 4.9 | — | 6% | wandb |
| 11 | `data_loader_collection.py:171` `__next__` | 1.9 | 1.9 | 2% | data |
| 12 | `validator.py:379` KMeans NMI | 3.3 | 3.3 | 4% | validation |
| 13 | `Tensor.item()` (all call sites) | — | 0.94 | 1% | other (GPU sync) |

---

## GPU vs CPU suspicion list

| Suspect | Evidence | Verdict |
|---------|----------|---------|
| **Autograd backward** | 22.5 s tottime in `run_backward` | True GPU work; dominant |
| **Conv1d / ConvTranspose1d / Linear** | 8.9 + 3.5 + 3.0 s tottime | True GPU compute |
| **`loss.item()` per batch** | 328k calls, 0.94 s; forces CUDA sync | CPU/sync — **fixed** |
| **`clip_grad_norm_`** | 6.1 s cum; traverses `named_parameters` 205k× | CPU overhead around GPU norm |
| **`validate_cvae_epoch`** | `get_all_data` + full-val forward + `.cpu().numpy()` | CPU + extra GPU forward (1×/smoke) |
| **sklearn KMeans** | 3.3 s in `__calculate_kmeans_nmi` | CPU; latent `.cpu().numpy()` at `validator.py:384` |
| **Dataloader `__next__`** | 1.9 s; batch slice + `.to(cuda)` | Minor CPU/indexing |
| **`state_distinctness` / `input_channel_statistics`** | Not in top 80 | Not on hot path during CVAE epoch val |
| **W&B init** | 4.9 s once at `experiment_logger.start` | Startup only |

cProfile **under-reports** GPU kernel time (async execution). W&B system charts (offline run `wandb/offline-run-20260607_124748-9j85r4lk`) are the right complement for GPU util %.

---

## Likely redundant / unvectorized work (file:line)

| Location | Issue | Impact |
|----------|-------|--------|
| `src/training/trainer.py:107-108` | Per-batch `loss.item()` / `reg_loss.item()` | **Fixed** — tensor sum, 2× `.item()` per epoch |
| `src/validation/validator.py:282` | `get_all_data()` materializes full val set each validation | 1× in smoke; costly if `validate_per_epoch: 1` |
| `src/validation/validator.py:384-390` | Latent → NumPy + sklearn KMeans on full val | 3.3 s per validation epoch |
| `src/validation/validator.py:324-325` | `.cpu().numpy()` for prior NMI | Small vs full-val forward |
| `src/training/trainer.py:105` | `clip_grad_norm_(self.model.parameters())` | 6 s cum; foreach path already used internally |
| `src/models/vae.py:476` `gmm_prior` | Already vectorized; 2.4 s tottime | GPU KL — not low-hanging CPU fruit |

---

## Not worth optimizing (this smoke)

- **W&B cold start (~5 s)** — one-time import + service spawn; use `WANDB_MODE=offline` or disable for micro-benchmarks only.
- **Conv/decoder architecture** — dominates fairly; needs arch/batch changes, not micro-opts.
- **`state_distinctness` / input stats** — not invoked in `validate_cvae_epoch` hot path.
- **Post-train `visualize_cvae` / GMM validation** — outside `--profile` window; acceptable for smoke.

---

## W&B / hardware snapshot

| Metric | Value |
|--------|-------|
| Device | `cuda` (logged in every DataLoader info block) |
| Peak VRAM (1 epoch, same config) | **442 MB** (`torch.cuda.max_memory_allocated`) |
| W&B run | Offline: `wandb/offline-run-20260607_124748-9j85r4lk`, project `SPA-local`, group `local_profile` |
| GPU util % | Not in offline summary JSON; sync run or inspect W&B System tab after `wandb sync` |

Batch 64 + full lab_2 cohort did **not** OOM on 8 GB.

---

## Optimizations applied

### Round 1 — defer per-batch `loss.item()` (`src/training/trainer.py`)

Accumulate detached loss tensors per epoch; two `.item()` calls per epoch for epoch means.

| Metric | Before | After R1 | Δ |
|--------|--------|----------|---|
| Profiled train time (5 ep) | 86.9 s | 76.4 s | **−12.0%** |
| `Tensor.item()` tottime | 0.94 s (328k calls) | 0.23 s | fewer GPU syncs |

### Round 2 — training + validation + dataloader (`2026-06-07`)

| Change | File | Effect |
|--------|------|--------|
| `zero_grad(set_to_none=True)` + cached `_grad_params` for clip | `src/training/trainer.py` | `clip_grad_norm_` 6.1 s → 4.4 s cum; `named_parameters` off hot path |
| GPU `torch.long` shuffle indices | `src/data/data_loader_collection.py` | `__next__` 1.9 s → 0.74 s tottime |
| Single `cvae.forward` in `validate_cvae_epoch`; pass `mu` to `predict_gmm_labels` | `src/validation/validator.py`, `src/models/vae.py` | Removes 2 redundant full-val encoder passes per validation |
| `torch.inference_mode()` in CVAE epoch validation | `src/validation/validator.py` | Slightly leaner than nested `no_grad` |
| Cache `_LOG_2PI` in GMM prior / predict | `src/models/vae.py` | Minor; avoids repeated `math.log` in hot KL |

| Metric | Baseline | After R1 | After R2 | Total Δ |
|--------|----------|----------|----------|---------|
| Profiled train time (5 ep) | 86.9 s | 76.4 s | **74.4 s** | **−14.4%** |
| Epoch-5 train loss | 1.1673 | 1.1655 | 1.1646 | same order |
| Epoch-1 train loss | 86246.95 | 86246.80 | 86246.53 | Δ < 0.002% |

**Correctness tolerance:** epoch mean train loss within **1e-4 relative** or **0.5 absolute** on late epochs — satisfied.

### Round 3 — cHMM-GMVAE + MARHMM paths (`2026-06-07` AFK pass)

| Change | File | Effect |
|--------|------|--------|
| `predict_hmm_labels(..., mu=)` skips re-encode | `src/models/vae.py`, `validator.py` | Same labels as full encode; used in epoch + post-train HMM val |
| `validate_cvae_hmm`: reuse `mu` for GMM marginal | `validator.py` | One fewer full-val encode |
| MARHMM phase: `encode_to_latent` only (skip frozen decode/loss) | `cvae_mar_hmm.py` | Faster `training_pipeline: marhmm` / `cvae_then_marhmm` phase |
| MARHMM `validate_epoch`: single encode for NLL + predict | `validator.py` | Removes double encode on val/train metrics |
| `calculate_nmi` on GPU tensors (prior val) | `validator.py` | Fewer host transfers |
| `_LOG_2PI` in `_log_q_per_step` | `vae.py` | HMM-GMM KL micro-opt |

**Parity checks:** `tests/test_perf_optimizations.py` (mu-cache HMM/GMM, MARHMM encode-only forward).  
**Local cHMM smoke:** `profile_chmm_single_mouse_smoke.yaml` (sub-071, short warmups).

| Metric | GMM lab_2 (5 ep) | cHMM sub-071 (5 ep) |
|--------|------------------|---------------------|
| Profiled time | **74.2 s** (was 86.9 s) | **~32 s** (single mouse) |

**Not pursued further:** W&B cold start (~5 s, one-time), conv/backward architecture (~60% GPU), sklearn KMeans at validation (would need algorithm change), `torch.compile`/AMP (numerics risk).

---

## Verification checklist

- [x] `Device: cuda` in logs
- [x] `train.prof` + `train_cprofile_cumulative.txt` + `train_cprofile_tottime.txt`
- [x] W&B offline run with metrics (`train/total_loss`, `val/prior_pred_nmi`, etc.)
- [x] This report with file:line references
- [x] Before/after timing + correctness note

---

## snakeviz

```bash
snakeviz results/local_profile/profile_cvae_lab2_smoke_20260607-124714/1/train.prof
```

Latest (all optimizations):

```bash
snakeviz results/local_profile/profile_cvae_lab2_smoke_20260607-143452/1/train.prof
snakeviz results/local_profile/profile_chmm_single_mouse_smoke_20260607-143702/1/train.prof
```

Parity + smoke:

```powershell
$env:PYTHONPATH="."
.\.venv\Scripts\python.exe -c "import tests.test_perf_optimizations as t; t.test_predict_hmm_labels_mu_matches_reencode(); t.test_marhmm_forward_encode_only_matches_full_cvae_path(); print('ok')"
.\.venv\Scripts\python.exe scripts/local/verify_perf_correctness.py -c src/config/run/cvaemarhmm/local/profile_cvae_single_mouse_smoke.yaml
```
