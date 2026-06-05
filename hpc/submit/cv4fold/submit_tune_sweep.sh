#!/bin/bash
# Phase 1: W&B Bayesian tune on full cohort (all mice, in-sample train+val).
#
# Prereq:
#   PYTHONPATH=. python3 scripts/cv4fold/generate_configs.py --phase tune_sweep_base --all-mice --models cgmvae chmmgmvae
#
# Usage:
#   bash hpc/submit/cv4fold/submit_tune_sweep.sh              # both cgmvae + chmmgmvae
#   MODEL=cgmvae bash hpc/submit/cv4fold/submit_tune_sweep.sh # one model
#
# Mem/queue (fixed in script): cgmvae gpuv100 8GB/slot 8h; chmmgmvae gpua100 12GB/slot 12h.

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "$SPA_ROOT"

SWEEP_TAG="${SWEEP_TAG:-bayes50}"
NUM_AGENTS="${NUM_AGENTS:-50}"
TRIALS_PER_AGENT=1
SWEEP_SLOTS=4

submit_one_model() {
  local model="$1"
  local sweep_yaml="src/config/sweep/cv4fold/${model}_joint_all_mice_${SWEEP_TAG}.yaml"
  local sweep_base="src/config/run/cvaeprior/cv4fold/tune/${model}_joint_all_mice_sweep_base.yaml"
  local queue walltime mem

  if [ ! -f "$sweep_yaml" ] || [ ! -f "$sweep_base" ]; then
    echo "Missing $sweep_yaml or $sweep_base"
    echo "Run: PYTHONPATH=. python3 scripts/cv4fold/generate_configs.py --phase tune_sweep_base --all-mice --models ${model}"
    exit 1
  fi

  if [ "$model" = "chmmgmvae" ]; then
    queue=gpua100
    mem=12GB
    walltime=12:00
  else
    queue=gpuv100
    mem=8GB
    walltime=8:00
  fi

  export SWEEP_MEM="$mem" SWEEP_SLOTS="$SWEEP_SLOTS" QUEUE="$queue"
  echo "Submitting ${model} all-mice sweep (${SWEEP_TAG}, ${NUM_AGENTS} agents, ${mem}/slot, ${queue})..."
  bash hpc/submit/launch_wandb_sweep.sh "$sweep_yaml" "$NUM_AGENTS" "$TRIALS_PER_AGENT" "$walltime" "$queue"
}

if [ -n "${MODEL:-}" ]; then
  case "$MODEL" in
    cgmvae|chmmgmvae) submit_one_model "$MODEL" ;;
    *)
      echo "MODEL must be cgmvae or chmmgmvae (got: $MODEL)"
      exit 1
      ;;
  esac
else
  submit_one_model cgmvae
  submit_one_model chmmgmvae
fi
