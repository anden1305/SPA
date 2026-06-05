#!/bin/bash
# subject_lab + tune_winner hyperparams: joint 4-fold, cgmvae + chmmgmvae (8 bsub jobs).
#
# Prereq (login node):
#   cd /work3/s204070/SPA
#   PYTHONPATH=. python3 scripts/cv4fold/generate_configs.py --phase subject_lab_tune_winners
#
# Submit:
#   bash hpc/submit/cv4fold/submit_subject_lab_tune_winners.sh
#
# After jobs finish (per fold/model, replace <timestamp_dir>):
#   PYTHONPATH=. python3 scripts/cv4fold/postprocess_fold.py \
#     --result-root results/cv4fold/subject_lab_tune_winners/chmmgmvae/joint/fold_4/<timestamp_dir>
#   PYTHONPATH=. python3 scripts/cv4fold.select_best_vae_run \
#     --fold-dir results/cv4fold/subject_lab_tune_winners/chmmgmvae/joint/fold_4/<timestamp_dir>
#
# Fold-level summary (all completed folds):
#   PYTHONPATH=. python3 scripts/cv4fold/summarize_subject_lab_cv.py

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "$SPA_ROOT"
source hpc/submit/cv4fold/cv4fold_queue.sh
source hpc/submit/cv4fold/cv4fold_bsub.sh

shopt -s nullglob

declare -A QUEUE_COUNT=()

submit_cfg() {
  local cfg="$1"
  local QUEUE
  QUEUE=$(cv4fold_pick_queue "$cfg")
  local tag results_base run_name
  tag=$(basename "$cfg" .yaml)
  run_name=$(python3 -c "import yaml; print(yaml.safe_load(open('${cfg}'))['run_name'])")
  results_base=$(python3 -c "import yaml; print(yaml.safe_load(open('${cfg}'))['results_dir'])")
  echo "  CONFIG: ${cfg}"
  echo "    queue=${QUEUE} walltime=4:00"
  echo "    LSF logs: hpc/output/cv4fold/vae/%J.out (JOBID printed by bsub below)"
  echo "    results:  ${results_base}/${run_name}_<timestamp>/"
  cv4fold_bsub_vae "$cfg" "$QUEUE" "4:00"
  QUEUE_COUNT["$QUEUE"]=$(( ${QUEUE_COUNT[$QUEUE]:-0} + 1 ))
  echo "  $(basename "$cfg") -> ${QUEUE}"
}

echo "subject_lab tune_winners VAE configs..."
for cfg in src/config/run/cvaeprior/cv4fold/subject_lab_tune_winners/joint/fold_*/{cgmvae,chmmgmvae}.yaml; do
  submit_cfg "$cfg"
done

echo "Submitted subject_lab_tune_winners:"
for q in gpuv100 gpua100 gpua40 gpul40s; do
  echo "  ${q}: ${QUEUE_COUNT[$q]:-0}"
done
echo ""
echo "Monitor (queue can stay PEND a long time):"
echo "  bjobs -u \$USER"
echo "  bjobs -J cv4fold_vae_cgmvae   # or cv4fold_vae_chmmgmvae"
echo "  tail -f hpc/output/cv4fold/vae/<JOBID>.out"
echo ""
echo "After all jobs finish:"
echo "  PYTHONPATH=. python3 scripts/cv4fold/summarize_subject_lab_cv.py"
echo "  -> results/cv4fold/subject_lab_tune_winners/summary.csv"
