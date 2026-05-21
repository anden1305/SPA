#!/bin/bash
# Lab-2 reliability sweep v2: 6 trials (3 lr x 2 num_batches), 5 seeds each, peak-NMI metric.
# Run from repo root: bash hpc/submit/decoder_only_lab_2/run_reliability_sweep_v2.sh

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "$SPA_ROOT"

bash hpc/submit/launch_wandb_sweep.sh \
  src/config/sweep/decoder_only/lab2_reliability_grid_v2.yaml \
  6 1 10:00 gpuv100
