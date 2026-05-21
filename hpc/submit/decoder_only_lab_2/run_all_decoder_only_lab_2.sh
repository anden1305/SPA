#!/bin/bash
### Submit all decoder-only lab-2 cGMVAE experiments (subject conditioning).
### Submit from login node: bash hpc/submit/decoder_only_lab_2/run_all_decoder_only_lab_2.sh

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "$SPA_ROOT"

mkdir -p hpc/output/decoder_only_lab_2

echo "Submitting Decoder-Only Lab-2 Experiments..."
echo "============================================"

echo ""
echo "Subjectwise (6 subjects)..."
bash hpc/submit/decoder_only_lab_2/run_decoder_only_lab_2_subjectwise.sh

echo ""
echo "Generalization-subject..."
bash hpc/submit/decoder_only_lab_2/run_decoder_only_lab_2_generalization_subject.sh

echo ""
echo "Reliability..."
bsub < hpc/submit/decoder_only_lab_2/run_decoder_only_lab_2_reliability.sh

echo ""
echo "Generalization..."
bash hpc/submit/decoder_only_lab_2/run_decoder_only_lab_2_generalization.sh

echo ""
echo "Done. Logs: hpc/output/decoder_only_lab_2/"
