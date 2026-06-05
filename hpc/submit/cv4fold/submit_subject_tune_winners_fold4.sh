#!/bin/bash
# Ablation: tune-winner hyperparams + subject-only conditioning, joint fold 4 only (2 jobs).
#
# Prereq:
#   PYTHONPATH=. python3 scripts/cv4fold/generate_configs.py --phase subject_tune_winners --folds 4
#
# Submit:
#   bash hpc/submit/cv4fold/submit_subject_tune_winners_fold4.sh

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "$SPA_ROOT"
source hpc/submit/cv4fold/cv4fold_queue.sh
source hpc/submit/cv4fold/cv4fold_bsub.sh

for cfg in src/config/run/cvaeprior/cv4fold/subject_tune_winners/joint/fold_4/{cgmvae,chmmgmvae}.yaml; do
  [[ -f "$cfg" ]] || { echo "Missing $cfg — run generate_configs first"; exit 1; }
  QUEUE=$(cv4fold_pick_queue "$cfg")
  echo "Submitting $cfg -> $QUEUE"
  cv4fold_bsub_vae "$cfg" "$QUEUE" "4:00"
done

echo "Logs: hpc/output/cv4fold/vae/%J.out"
echo "Results: results/cv4fold/subject_tune_winners/{cgmvae,chmmgmvae}/joint/fold_4/"
