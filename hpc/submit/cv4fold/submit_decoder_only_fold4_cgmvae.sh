#!/bin/bash
# Decoder-only cGMVAE fold 4: steps A/B/C (3 GPU jobs, cgmvae only).
#
#   A  subject_tune_winners_seq1   seq=1,  subject
#   B  subject_tune_winners        seq=64, subject
#   C  subject_lab_tune_winners     seq=64, subject_lab
#
# Submit:
#   bash hpc/submit/cv4fold/submit_decoder_only_fold4_cgmvae.sh

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "$SPA_ROOT"
source hpc/submit/cv4fold/cv4fold_queue.sh
source hpc/submit/cv4fold/cv4fold_bsub.sh

WALLTIME="4:00"

PYTHONPATH=. python3 scripts/cv4fold/generate_configs.py --phase subject_tune_winners_seq1 --folds 4
PYTHONPATH=. python3 scripts/cv4fold/generate_configs.py --phase subject_tune_winners --folds 4
PYTHONPATH=. python3 scripts/cv4fold/generate_configs.py --phase subject_lab_tune_winners --folds 4

CONFIGS=(
  src/config/run/cvaeprior/cv4fold/subject_tune_winners_seq1/joint/fold_4/cgmvae.yaml
  src/config/run/cvaeprior/cv4fold/subject_tune_winners/joint/fold_4/cgmvae.yaml
  src/config/run/cvaeprior/cv4fold/subject_lab_tune_winners/joint/fold_4/cgmvae.yaml
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
echo "  A: results/cv4fold/subject_tune_winners_seq1/cgmvae/joint/fold_4/"
echo "  B: results/cv4fold/subject_tune_winners/cgmvae/joint/fold_4/"
echo "  C: results/cv4fold/subject_lab_tune_winners/cgmvae/joint/fold_4/"
echo ""
echo "After finish:"
echo "  PYTHONPATH=. python3 scripts/cv4fold/summarize_decoder_only_fold4.py --postprocess"
