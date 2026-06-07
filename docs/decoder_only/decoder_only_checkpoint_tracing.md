# Decoder-only checkpoint tracing

How to confirm which pretrained weights a reliability run used.

For load/save behaviour see [cvae_checkpointing.md](../training/cvae_checkpointing.md).

## 1. WandB config

Run → **Overview** → **Config** → `cvae.model_checkpoint_path`.

April ~0.72 cGMVAE runs ([yklyqlsh](https://wandb.ai/dtu_projects/SPA/runs/yklyqlsh), [cme0uc0e](https://wandb.ai/dtu_projects/SPA/runs/cme0uc0e)) used:

`results/decoder_only/cvae_final_decoder_only/cvae_decoder_only_model.pth`

## 2. HPC stdout

```bash
grep -i "Loading CVAE model from checkpoint" hpc/output/decoder_only/reliability/cgmvae_<JOBID>.out
```

No line → trained from scratch (`model_checkpoint_path: null` or missing file).

## 3. Saved `config.json`

Under the timestamped results dir, e.g.:

`results/decoder_only/reliability/cgmvae_decoder_only/reliability_cgmvae_decoder_only_mssv_frequency_<timestamp>/config.json`

## 4. `validation_info.json`

Written by `predict_cvae` / post-train validation under `plots/` with `checkpoint_path`, `checkpoint_size`, `checkpoint_mtime`.

## 5. Inspect `.pth` shapes (vae_decoder)

```python
import torch
st = torch.load("results/decoder_only/cvae_final_decoder_only/cvae_decoder_only_model.pth", map_location="cpu")
cvae = {k: v for k, v in st.items() if k.startswith("cvae.")}
print(cvae["cvae.encoder_mlp.0.weight"].shape)  # decoder-only: [256, 2176]
print(cvae["cvae.subject_emb.weight"].shape)    # vae_decoder: [92, 4]
```

**Important:** Checkpoints must be trained on the same branch indexing (`vae_decoder` uses 92-row `subject_emb` with legacy sub-NNN indices). Compact 0..N-1 embeddings from other branches will not load.

## Baseline training (before reliability)

| Conditioning | Config | Output checkpoint |
|--------------|--------|-------------------|
| Decoder-only | `src/config/run/cvaemarhmm/final/cvae_final_decoder_only.yaml` | `results/decoder_only/cvae_final_decoder_only/cvae_decoder_only_model.pth` |
| Encoder+decoder | `src/config/run/cvaemarhmm/final/cvae_final.yaml` | `results/decoder_only/cvae_baseline/cvae_baseline_model.pth` |

```bash
cd /work3/s204070/SPA
bsub < hpc/submit/decoder_only/baseline/run_decoder_only.sh
bsub < hpc/submit/decoder_only/baseline/run_encoder_decoder.sh
```

LSF logs: `hpc/output/decoder_only/baseline/` (`decoder_only_<JOBID>.out`, `encoder_decoder_<JOBID>.out`).

Then set `model_checkpoint_path` in the reliability YAML and submit reliability jobs.

## cHMMGMVAE reliability (after minimal port)

| Variant | Config | Baseline checkpoint |
|---------|--------|---------------------|
| Decoder-only | `reliability_chmmgmvae_decoder_only_mssv_frequency.yaml` | `cvae_final_decoder_only/cvae_decoder_only_model.pth` |
| Encoder+decoder | `reliability_chmmgmvae_encoder_decoder_mssv_frequency.yaml` | `cvae_baseline/cvae_baseline_model.pth` |

```bash
# cGMVAE regression (~0.70–0.73 KMeans with HPC-trained baseline)
bsub < hpc/submit/decoder_only/reliability/run_cgmvae.sh

# cHMMGMVAE decoder-only
bsub < hpc/submit/decoder_only/reliability/run_chmm_decoder_only.sh

# cHMMGMVAE encoder+decoder (after cvae_final baseline)
bsub < hpc/submit/decoder_only/reliability/run_chmm_encoder_decoder.sh

# unit tests on compute
bsub < hpc/submit/tests/run_pytest.sh
```

## HPC log layout (`hpc/output/decoder_only/`)

| Job | Script | LSF stdout |
|-----|--------|------------|
| Baseline decoder-only | `decoder_only/baseline/run_decoder_only.sh` | `baseline/decoder_only_<JOBID>.out` |
| Baseline encoder+decoder | `decoder_only/baseline/run_encoder_decoder.sh` | `baseline/encoder_decoder_<JOBID>.out` |
| cGMVAE reliability | `decoder_only/reliability/run_cgmvae.sh` | `reliability/cgmvae_<JOBID>.out` |
| cHMM decoder-only | `decoder_only/reliability/run_chmm_decoder_only.sh` | `reliability/chmmgmvae_decoder_only_<JOBID>.out` |
| cHMM encoder+decoder | `decoder_only/reliability/run_chmm_encoder_decoder.sh` | `reliability/chmmgmvae_encoder_decoder_<JOBID>.out` |

Monitor: `bjobs -J do_rel_cgmvae` or `ls -lt hpc/output/decoder_only/reliability/`

**Walltime (scripts):** baseline decoder `1:30`, baseline encoder `3:00` (3 runs), cGMVAE reliability `6:00` (10×~25 min + buffer), chmm `2:00` (1 run).

Science results (checkpoints, plots, W&B) stay under `results/decoder_only/...` per YAML `results_dir` / `run_name`.

Post-train prior metrics: `results/decoder_only/reliability/<model>/<run_name>_<timestamp>/<seed>/plots/metrics.txt`.
