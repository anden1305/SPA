# Local cGMVAE profiling (laptop GPU)

Profile CVAE training on a local NVIDIA GPU without HPC/bsub. Added 2026-06-07.

## Quick start

```bash
# Bash / Git Bash (repo root)
export WANDB_PROJECT=SPA-local   # keep HPC project clean
# optional: export WANDB_MODE=offline
bash scripts/local/run_profile.sh
```

**PowerShell (Windows):**

```powershell
$env:PYTHONPATH="."
$env:WANDB_PROJECT="SPA-local"
$env:PYTHONIOENCODING="utf-8"
.\.venv\Scripts\python.exe main.py --method train_vae --profile -c src/config/run/cvaemarhmm/local/profile_cvae_lab2_smoke.yaml
```

Orchestrator appends a timestamp to `run_name` (e.g. `profile_cvae_lab2_smoke_20260607-124714`). Artifacts are under `results/local_profile/<run_name>_YYYYMMDD-HHMMSS/1/`.

| File | Purpose |
|------|---------|
| `train.prof` | Binary cProfile dump — open with [snakeviz](https://jiffyclub.github.io/snakeviz/) |
| `train_cprofile_cumulative.txt` | Sorted by **cumulative** time (includes callees) |
| `train_cprofile_tottime.txt` | Sorted by **tottime** (time in function body only) |

Interactive view (after `pip install snakeviz`):

```bash
bash scripts/local/open_snakeviz.sh
# or: snakeviz results/local_profile/profile_cvae_lab2_smoke/1/train.prof
```

## What gets profiled

`--profile` wraps **`trainer.train()` only** for `--method train_vae`. Post-train `validate_cvae()`, checkpoint save, and `visualize_cvae()` run **outside** the profiler so the report reflects the training hot path (forward/backward/optimizer + per-epoch validation hooks inside `train()`).

Per-epoch validation runs when `(epoch + 1) % validate_per_epoch == 0`. Smoke configs use `validate_per_epoch: 5` with `epochs: 5` → one validation at the final epoch.

## Smoke configs

| Config | Cohort | Notes |
|--------|--------|-------|
| `src/config/run/cvaemarhmm/local/profile_cvae_lab2_smoke.yaml` | lab_2 winner (6 mice × 2 runs) | Primary profile target |
| `src/config/run/cvaemarhmm/local/profile_cvae_single_mouse_smoke.yaml` | sub-071 only | Fallback if 8 GB OOM |
| `src/config/run/cvaemarhmm/local/profile_chmm_single_mouse_smoke.yaml` | sub-071, `warm_hmm_gmm` | cHMM-GMVAE local profile |

Overrides vs HPC lab_2 winner: `runs: 1`, `epochs: 5`, `batch_size: 64`, `results_dir: results/local_profile`, W&B `group: local_profile`.

## W&B system metrics

Smoke configs keep `wandb.enabled: true`. Set **`WANDB_PROJECT=SPA-local`** so runs do not mix with HPC experiments. W&B **System** charts (GPU util %, GPU memory, CPU, RAM) complement cProfile: cProfile is CPU-side Python; GPU kernels appear indirectly via `cuda` sync points and `.item()` calls.

Use `WANDB_MODE=offline` if you have no network; metrics are stored locally under `wandb/`.

## cProfile: cumulative vs tottime

- **cumulative** — time in a function **plus** everything it calls. Use to find which high-level call chains dominate (e.g. `train` → `validate_cvae_epoch` → `get_all_data`).
- **tottime** — time spent **inside** the function body only. Use to find leaf hotspots (e.g. `torch.pdist`, `numpy` FFT, sklearn KMeans).

PyTorch CUDA ops often show low tottime in cProfile because work runs on GPU; look for `.cpu()`, `.numpy()`, `.item()`, and Python loops as sync / CPU-bound suspects.

## 8 GB VRAM mitigations

1. Lower `dataloader.batch_size` to **32** in the smoke YAML.
2. Switch to `profile_cvae_single_mouse_smoke.yaml`.
3. Reduce `dataloader.num_batches` (fewer windows per epoch) for a quicker smoke only — not for HPC parity.

## Code pointers

- CLI: `main.py --method train_vae --profile`
- Orchestrator: `src/orchestrator/orchestrator.py` → `train_cvae()` uses `trainer.train_profiled()`
- Profiler: `src/training/trainer.py` → `train_profiled()`; `src/helpers/profiling.py` → `write_cprofile_outputs()`

## Correctness checks (before trusting speedups)

```powershell
$env:PYTHONPATH="."
.\.venv\Scripts\python.exe -c "import tests.test_perf_optimizations as t; t.test_predict_hmm_labels_mu_matches_reencode(); t.test_marhmm_forward_encode_only_matches_full_cvae_path(); print('parity ok')"
.\.venv\Scripts\python.exe scripts/local/verify_perf_correctness.py -c src/config/run/cvaemarhmm/local/profile_cvae_single_mouse_smoke.yaml
```

Unit tests assert **bit-identical** HMM/GMM labels and MARHMM losses for optimized vs reference paths. The smoke script prints 1-epoch metrics (GPU may add tiny float noise across repeated inits).

## Hotspot report

See [local_profiling_report.md](local_profiling_report.md) for interpreted results from the RTX 4060 smoke run.
