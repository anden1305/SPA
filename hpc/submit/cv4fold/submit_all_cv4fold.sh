#!/bin/bash
# Phase 2: full cv4fold grid (all folds/scopes). After smoke80 + all-mice tune + manifest hyperparams.

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "$SPA_ROOT"
source hpc/submit/cv4fold/cv4fold_queue.sh
source hpc/submit/cv4fold/cv4fold_bsub.sh

shopt -s nullglob

declare -A QUEUE_COUNT=()

submit_cfg() {
  local cfg="$1"
  local kind="$2"
  local QUEUE
  QUEUE=$(cv4fold_pick_queue "$cfg")
  if [[ "$kind" == vae ]]; then
    cv4fold_bsub_vae "$cfg" "$QUEUE" "16:00"
  else
    cv4fold_bsub_hmm_raw "$cfg" "$QUEUE" "24:00"
  fi
  QUEUE_COUNT["$QUEUE"]=$(( ${QUEUE_COUNT[$QUEUE]:-0} + 1 ))
  echo "  $(basename "$cfg") -> ${QUEUE}"
}

echo "VAE configs..."
for cfg in src/config/run/cvaeprior/cv4fold/joint/fold_*/{cgmvae,hmmgmvae,chmmgmvae}.yaml \
           src/config/run/cvaeprior/cv4fold/per_lab/lab_*/fold_*/{cgmvae,hmmgmvae,chmmgmvae}.yaml; do
  submit_cfg "$cfg" vae
done

echo "HMM raw configs..."
for cfg in src/config/run/hmm/cv4fold/joint/fold_*/hmm_raw.yaml \
           src/config/run/hmm/cv4fold/per_lab/lab_*/fold_*/hmm_raw.yaml; do
  submit_cfg "$cfg" hmm
done

echo "Submitted full cv4fold grid:"
for q in gpuv100 gpua100 gpua40 gpul40s; do
  echo "  ${q}: ${QUEUE_COUNT[$q]:-0}"
done
