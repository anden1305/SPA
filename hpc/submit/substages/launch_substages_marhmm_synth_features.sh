#!/bin/bash
### Substages Sweep: MAR-HMM on Synthetic Features
### Sweeps over n_states from 2 to 10 to find optimal model complexity

set -euo pipefail

SWEEP_YAML="src/config/sweep/substages/substages_marhmm_synth_features.yaml"
NUM_AGENTS=9  # One agent per state count (2-10)
TRIALS_PER_AGENT=1
WALLTIME="08:00"
QUEUE="gpul40s"

bash hpc/submit/launch_wandb_sweep.sh "$SWEEP_YAML" "$NUM_AGENTS" "$TRIALS_PER_AGENT" "$WALLTIME" "$QUEUE"

# Usage:
# bash hpc/submit/substages/launch_substages_marhmm_synth_features.sh
#
# Monitor with:
# bjobs
# Check output in: hpc/output/sweep_<SWEEP_ID>/
# After completion: python scripts/analyze_substages_sweep.py <SWEEP_ID>
