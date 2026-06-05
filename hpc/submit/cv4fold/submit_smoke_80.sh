#!/bin/bash
# Phase 0 smoke: joint fold 4, 80 epochs, all four model families on gpuv100.
# Next: submit_tune_sweep.sh (all mice) → submit_all_cv4fold.sh (full grid).
#
# Prereq:
#   PYTHONPATH=. python3 scripts/cv4fold/generate_configs.py --phase smoke_80

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "$SPA_ROOT"
source hpc/submit/cv4fold/cv4fold_bsub.sh

QUEUE="gpuv100"

for cfg in \
  src/config/run/cvaeprior/cv4fold/smoke80/cgmvae_joint_fold_4.yaml \
  src/config/run/cvaeprior/cv4fold/smoke80/hmmgmvae_joint_fold_4.yaml \
  src/config/run/cvaeprior/cv4fold/smoke80/chmmgmvae_joint_fold_4.yaml \
  src/config/run/hmm/cv4fold/smoke80/hmm_raw_joint_fold_4.yaml
do
  if [[ "$cfg" == *hmm/cv4fold/smoke80* ]]; then
    cv4fold_bsub_hmm_raw "$cfg" "$QUEUE" "8:00"
  else
    cv4fold_bsub_vae "$cfg" "$QUEUE" "8:00"
  fi
  echo "Submitted smoke80: $(basename "$cfg") -> ${QUEUE}"
done

echo "Done (4 jobs @ 80 epochs, 8h wall). Monitor: bjobs"
