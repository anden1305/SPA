#!/bin/bash
# Fold-4 sanity checks (5 GPU jobs, 4h each):
#   - subject_tune_winners_seq1: tune winners, subject, sequence_length=1
#   - subject_lab_tune_winners_seq1: tune winners, subject_lab, sequence_length=1
#   - cgmvae_cvae_final: cvae_final.yaml recipe, subject, encoder+decoder conditioning
#
# Generate configs (login node OK):
#   cd /work3/s204070/SPA
#   PYTHONPATH=. python3 scripts/cv4fold/generate_configs.py --phase subject_tune_winners_seq1 --folds 4
#   PYTHONPATH=. python3 scripts/cv4fold/generate_configs.py --phase subject_lab_tune_winners_seq1 --folds 4
#   PYTHONPATH=. python3 scripts/cv4fold/generate_configs.py --phase cgmvae_cvae_final --folds 4
#
# Submit:
#   bash hpc/submit/cv4fold/submit_fold4_sanity_checks.sh

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "$SPA_ROOT"
source hpc/submit/cv4fold/cv4fold_queue.sh
source hpc/submit/cv4fold/cv4fold_bsub.sh

WALLTIME="4:00"

PYTHONPATH=. python3 scripts/cv4fold/generate_configs.py --phase subject_tune_winners_seq1 --folds 4
PYTHONPATH=. python3 scripts/cv4fold/generate_configs.py --phase subject_lab_tune_winners_seq1 --folds 4
PYTHONPATH=. python3 scripts/cv4fold/generate_configs.py --phase cgmvae_cvae_final --folds 4

CONFIGS=(
  src/config/run/cvaeprior/cv4fold/subject_tune_winners_seq1/joint/fold_4/cgmvae.yaml
  src/config/run/cvaeprior/cv4fold/subject_tune_winners_seq1/joint/fold_4/chmmgmvae.yaml
  src/config/run/cvaeprior/cv4fold/subject_lab_tune_winners_seq1/joint/fold_4/cgmvae.yaml
  src/config/run/cvaeprior/cv4fold/subject_lab_tune_winners_seq1/joint/fold_4/chmmgmvae.yaml
  src/config/run/cvaeprior/cv4fold/cgmvae_cvae_final/joint/fold_4/cgmvae.yaml
)

for cfg in "${CONFIGS[@]}"; do
  [[ -f "$cfg" ]] || { echo "Missing $cfg"; exit 1; }
  QUEUE=$(cv4fold_pick_queue "$cfg")
  echo "Submitting $cfg -> ${QUEUE} walltime=${WALLTIME}"
  cv4fold_bsub_vae "$cfg" "$QUEUE" "$WALLTIME"
done

echo ""
echo "Logs: hpc/output/cv4fold/vae/%J.out"
echo "Results:"
echo "  results/cv4fold/subject_tune_winners_seq1/{cgmvae,chmmgmvae}/joint/fold_4/"
echo "  results/cv4fold/subject_lab_tune_winners_seq1/{cgmvae,chmmgmvae}/joint/fold_4/"
echo "  results/cv4fold/cgmvae_cvae_final/cgmvae/joint/fold_4/"
