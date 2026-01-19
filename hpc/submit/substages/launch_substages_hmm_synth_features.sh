#!/bin/bash
### Substages Sweep: HMM on Synthetic Features
### Sweeps over n_states from 2 to 10 to find optimal model complexity

set -euo pipefail

SWEEP_YAML="src/config/sweep/substages/substages_hmm_synth_features.yaml"
NUM_AGENTS=9  # One agent per state count (2-10)
TRIALS_PER_AGENT=1
WALLTIME="12:00" # TODO 8:00
QUEUE="gpuv100" # TODO gpuv100

bash hpc/submit/launch_wandb_sweep.sh "$SWEEP_YAML" "$NUM_AGENTS" "$TRIALS_PER_AGENT" "$WALLTIME" "$QUEUE"

# Usage:
# bash hpc/submit/substages/launch_substages_hmm_synth_features.sh
#
# Monitor with:
# bjobs
# Check output in: hpc/output/sweep_<SWEEP_ID>/
# After completion: python scripts/analyze_substages_sweep.py <SWEEP_ID>
