# HPC Submit Scripts for Experiments 5-7

This directory contains LSF/bsub submit scripts for running experiments on the HPC cluster.

## Directory Structure

```
hpc/submit/
├── reliability/          # Experiment 5: Reliability analysis
├── substages/           # Experiment 6: State count sweep
├── generalization/      # Experiment 7: Between-mouse generalization
└── submit_all_experiments.sh  # Master script to run everything
```

## Quick Start

### Submit All Experiments at Once
```bash
bash hpc/submit/submit_all_experiments.sh
```
This will submit ~30+ jobs for all experiments. **Use with caution!**

### Submit by Experiment Type

```bash
# Reliability only (4 jobs)
bash hpc/submit/reliability/submit_all_reliability.sh

# Substages only (4 sweeps = ~36 jobs)
bash hpc/submit/substages/submit_all_substages.sh

# Generalization only (6 jobs)
bash hpc/submit/generalization/submit_all_generalization.sh
```

### Submit Individual Jobs

```bash
# Single reliability experiment
bsub < hpc/submit/reliability/run_reliability_hmm_mssv_features.sh

# Single substages sweep
bash hpc/submit/substages/launch_substages_hmm_mssv_features.sh

# Single generalization test
bsub < hpc/submit/generalization/run_gen_hmm_test_sub039.sh
```

## Experiment 5: Reliability Analysis

**Purpose**: Test model consistency across multiple runs with different seeds.

**Files**:
- `run_reliability_hmm_mssv_features.sh` - HMM on MSSV (10 runs, 24h)
- `run_reliability_hmm_synth_features.sh` - HMM on Synthetic (10 runs, 12h)
- `run_reliability_marhmm_mssv_features.sh` - MAR-HMM on MSSV (10 runs, 24h)
- `run_reliability_marhmm_synth_features.sh` - MAR-HMM on Synthetic (10 runs, 12h)

**Resource Allocation**:
- Queue: gpuv100
- Walltime: 12-24 hours
- Memory: 8GB
- GPUs: 1 per job

**Metrics**:
- Cross-NMI (consistency across runs)
- NMI vs true states distribution
- Loss distribution

**Submit All**:
```bash
bash hpc/submit/reliability/submit_all_reliability.sh
```

## Experiment 6: Substages (State Count Sweep)

**Purpose**: Find optimal number of states by sweeping from 2 to 10.

**Files**:
- `launch_substages_hmm_mssv_features.sh` - HMM on MSSV (9 agents)
- `launch_substages_marhmm_mssv_features.sh` - MAR-HMM on MSSV (9 agents)
- `launch_substages_hmm_synth_features.sh` - HMM on Synthetic (9 agents)
- `launch_substages_marhmm_synth_features.sh` - MAR-HMM on Synthetic (9 agents)

**Resource Allocation**:
- Queue: gpuv100
- Walltime: 8-12 hours per agent
- Memory: 5GB per agent
- GPUs: 1 per agent
- Total jobs: 9 agents × 4 sweeps = 36 jobs

**Metrics**:
- NMI vs n_states
- Validation loss vs n_states
- Explained NMI (incremental improvement)

**Submit All**:
```bash
bash hpc/submit/substages/submit_all_substages.sh
```

## Experiment 7: Generalization (Between-Mouse)

**Purpose**: Test model generalization to unseen mice (leave-one-mouse-out).

**HMM Files**:
- `run_gen_hmm_test_sub039.sh` - Train on all except sub-039
- `run_gen_hmm_test_sub041.sh` - Train on all except sub-041
- `run_gen_hmm_test_sub048.sh` - Train on all except sub-048

**MAR-HMM Files**:
- `run_gen_marhmm_test_sub039.sh` - Train on all except sub-039
- `run_gen_marhmm_test_sub041.sh` - Train on all except sub-041
- `run_gen_marhmm_test_sub048.sh` - Train on all except sub-048

**Resource Allocation**:
- Queue: gpuv100
- Walltime: 12 hours
- Memory: 8GB
- GPUs: 1 per job
- Total jobs: 3 mice × 2 models = 6 jobs

**Metrics**:
- Validation NMI (on held-out mouse)
- Validation predictive likelihood
- Train-Val gap (generalization quality)

**Submit All**:
```bash
bash hpc/submit/generalization/submit_all_generalization.sh
```

## Monitoring Jobs

```bash
# View all your jobs
bjobs

# View specific job details
bjobs -l <job_id>

# View running jobs
bjobs -r

# View pending jobs
bjobs -p

# Kill a job
bkill <job_id>

# Kill all your jobs (use with caution!)
bkill 0
```

## Output Files

All job outputs are saved to `hpc/output/`:

```
hpc/output/
├── reliability_hmm_mssv_features_<JOB_ID>.out
├── reliability_hmm_mssv_features_<JOB_ID>.err
├── generalization_hmm_test_sub039_<JOB_ID>.out
├── sweep_<SWEEP_ID>/
│   ├── sweep_<SWEEP_ID>_job_<JOB_ID>_agent_1.out
│   └── sweep_<SWEEP_ID>_job_<JOB_ID>_agent_1.err
└── ...
```

## Results Directories

Experiment results are saved to:

```
results/
├── reliability/
│   ├── hmm/
│   │   ├── reliability_hmm_mssv_features_<timestamp>/
│   │   │   ├── 0/  # Run 0
│   │   │   ├── 1/  # Run 1
│   │   │   └── ...
│   │   └── ...
│   └── marhmm/
│       └── ...
├── substages/
│   ├── hmm/
│   │   └── substages_hmm_mssv_features_<timestamp>/
│   └── marhmm/
│       └── ...
└── generalization/
    ├── hmm/
    │   ├── generalization_hmm_mssv_features_test_sub039_<timestamp>/
    │   └── ...
    └── marhmm/
        └── ...
```

## Troubleshooting

### Job Fails Immediately
- Check `.err` file for error messages
- Verify config paths are correct
- Ensure `.venv` exists and is activated
- Check CUDA module is loaded

### Job Runs Out of Memory
- Increase `-R "rusage[mem=XGB]"` in submit script
- Reduce batch size in config
- Reduce number of datasets

### Job Times Out
- Increase `-W HH:MM` walltime
- Enable early stopping in config
- Reduce number of epochs

### W&B Authentication Issues
- Ensure `.env` file exists with `WANDB_API_KEY`
- Or export: `export WANDB_API_KEY=<your_key>`
- Check network connectivity from compute node

## Best Practices

1. **Test First**: Run one job before submitting batches
   ```bash
   bsub < hpc/submit/reliability/run_reliability_hmm_mssv_features.sh
   bjobs  # Monitor until it starts running
   tail -f hpc/output/reliability_hmm_mssv_features_<JOB_ID>.out
   ```

2. **Stagger Submissions**: Don't submit all jobs at once if cluster is busy
   ```bash
   # Submit in batches with delays
   bash hpc/submit/reliability/submit_all_reliability.sh
   sleep 3600  # Wait 1 hour
   bash hpc/submit/substages/submit_all_substages.sh
   ```

3. **Monitor Progress**: Check W&B dashboard for real-time metrics
   - https://wandb.ai/<your-entity>/SPA

4. **Resource Efficiency**: 
   - Use `gpuv100` for most jobs (cheaper, sufficient)
   - Reserve `gpua100` for large/complex jobs only
   - Match walltime to expected duration (don't over-request)

## Estimated Resource Usage

| Experiment | Jobs | GPU Hours | Walltime | Total Cost |
|-----------|------|-----------|----------|------------|
| Reliability | 4 | ~84h | 12-24h | Medium |
| Substages | 36 | ~360h | 8-12h | High |
| Generalization | 6 | ~72h | 12h | Medium |
| **Total** | **46** | **~516h** | **varies** | **High** |

**Note**: Actual usage depends on convergence speed and early stopping.

## Quick Reference Commands

```bash
# Submit all experiments
bash hpc/submit/submit_all_experiments.sh

# Submit by type
bash hpc/submit/reliability/submit_all_reliability.sh
bash hpc/submit/substages/submit_all_substages.sh
bash hpc/submit/generalization/submit_all_generalization.sh

# Monitor
bjobs
tail -f hpc/output/<latest_output>.out

# Cancel all jobs
bkill 0
```

## Related Documentation

- Main experiment overview: [EXPERIMENTS.md](../../EXPERIMENTS.md)
- Config documentation: [src/config/](../../src/config/)
- Reliability README: [src/config/run/hmm/reliability/README.md](../../src/config/run/hmm/reliability/README.md)
- Substages README: [src/config/sweep/substages/README.md](../../src/config/sweep/substages/README.md)
- Generalization README: [src/config/run/hmm/generalization/README.md](../../src/config/run/hmm/generalization/README.md)
