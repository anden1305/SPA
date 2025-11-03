# Copilot instructions for SPA

Use these notes to be productive immediately in this repo. Keep changes config‑driven, preserve data shapes/contracts, and follow existing extension points.

## Big picture
- Entry point: `main.py --method {train|generate|explore_synthetic|sweep} --config_path <yaml>`.
- Config-first: YAML → Pydantic models in `src/config/config.py` (strict validation; comments document hardcoded numerical safeties).
- Train flow: `Orchestrator` builds `DataLoaderCollection` (from `train_datasets`/`val_datasets`) → model (`HMM`/`MARHMM`) → `Trainer` → `Validator` → `Visualizer`. Repeats `runs` times and aggregates.
- Results: `results/.../<run_name [YYYYmmdd-HHMMSS]>/<runIndex>/` holds `config.json`, `plots/`, and `model.pth`.

## Workflows (Windows + uv)
- Train (example): `uv run main.py --method train --config_path src/config/run/hmm/config_random_uniform.yaml`
- Sweep: `uv run main.py --method sweep --config_path src/config/sweep/hmm_baseline.yaml` (one trial = hyperparam setting; inside each trial, Orchestrator runs multiple `runs` and the sweep reports best-of per trial).
- Quick test: `uv run test.py` (runs `tests/test_pipeline.py`).

## Config conventions (YAML → Pydantic)
- Top-level: `trainer`, `validator`, `visualizer`, `dataloader`, `train_datasets`, `val_datasets`, `model`, plus `verbose|seed|runs|validate_data|wandb`.
- Sweeps: dotted keys (e.g., `trainer.learning_rate`) deep-merge into base config in `wandb_sweep_runner._apply_wandb_config`.

## Data, transforms, and shapes (strict)
- Datasets must yield `data: (C,T)`, `labels: (T,)`, with `channels` values from {`EEG`,`EMG`}.
	- `src/data/synthetic_dataset.py`: expects `data/synthetic_data/<id>/{eeg.npy,labels.npy,config_copy.yml}`.
	- `src/data/mssv_dataset.py`: from `data/ds006366_processed/<sub>/<run>/*.npy` + `metadata.csv`.
- Transforms (`src/preprocessing/*`) run per channel group and return `(N,T,C′)` while keeping `y` consistent across groups. Register new types in `DataLoader.__init_transforms`.
- DataLoader batches contiguous segments and yields tensors shaped `(B,T,D)` (currently B=1). `DataLoaderCollection` enforces same `n_stages` and same post‑transform feature dim across datasets.

## Models and training
- Base model contract: `forward(x: (B,T,D))`, `predict`, `prepare_for_training/inference`, `regularization_loss`, `reset` (see `src/models/base_model.py`).
- `HMM` (`covariance_type: meanonly|diag|full`) with init strategies: `random_uniform|random_dirichlet|random_separated|kmeans|kmeans_pca`; optional `init_noisy`.
- `MARHMM`: autoregressive variant (see `ModelConfig.params` notes in `config.py`).
- Optimizers: `adam|sgd|rmsprop|adamw`; schedulers: `step|exponential` (when enabled). Early stopping behavior documented in `EarlyStoppingConfig` comments.

## Validation, visuals, and logging
- `validator.*` toggles (NMI, accuracy via Hungarian alignment, learning rate, state distinctness, summary stats, cross‑NMI across runs).
- Visuals: losses, PCA tripanel, static + interactive confusion matrices, learning rate, historic parameter tracks, and data diagnostics into `plots/`.
- W&B: `src/training/experiment_logger.py` logs per‑epoch metrics when `wandb.enabled=true`; sweeps reuse one run per trial and aggregate best run into `wandb.summary[metric]`.

## Extending
- New dataset: subclass `BaseDataset`, keep `(C,T)/(T,)` and required config fields; wire in `Orchestrator.get_*_datasets()` by `DatasetConfig.type`.
- New transform: subclass `BaseTransform` and return `(N,T,C′)`; add type dispatch in `DataLoader.__init_transforms`.
- New model: subclass `BaseModel`; extend `Orchestrator.__get_model` with a new `ModelConfig.type` value.

## Gotchas / invariants
- Shape invariants are enforced; mismatches raise early. All datasets in a run must end with identical feature dim and `n_stages`.
- `runs` > 1 means multiple internal runs per trial; seed auto‑increments across runs; sweep scoring uses the best run’s metric.
- Some numerical stabilizers (jitters, floors) are intentionally hardcoded—review comments in `src/config/config.py` and model files before changing.

