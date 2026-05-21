#!/bin/bash
# Launch W&B grid sweep for lab-2 population reliability (6 trials: 3 lr × 2 max_beta).
# Run from repo root on login node: bash hpc/submit/decoder_only_lab_2/run_reliability_sweep.sh

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "$SPA_ROOT"

bash hpc/submit/launch_wandb_sweep.sh \
  src/config/sweep/decoder_only/lab2_reliability_grid.yaml \
  6 1 08:00 gpuv100
