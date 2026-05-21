#!/bin/bash
### Submit all decoder-only lab-5 cGMVAE experiments (subject conditioning).
### Each config uses a per-subject model_checkpoint_path, so jobs within a
### pipeline can run in parallel without overwriting cvae_final_model.pth.

set -euo pipefail

mkdir -p hpc/output/decoder_only_lab_5

echo "Submitting Decoder-Only Lab-5 Experiments..."
echo "============================================"

echo ""
echo "Subjectwise Experiments:"
echo "------------------------"
bash hpc/submit/decoder_only_lab_5/run_decoder_only_lab_5_subjectwise.sh

echo ""
echo "Generalization-Subject Experiments:"
echo "-----------------------------------"
bash hpc/submit/decoder_only_lab_5/run_decoder_only_lab_5_generalization_subject.sh

echo ""
echo "Reliability Experiments:"
echo "------------------------"
bsub < hpc/submit/decoder_only_lab_5/run_decoder_only_lab_5_reliability.sh

echo ""
echo "Generalization Experiments:"
echo "---------------------------"
bash hpc/submit/decoder_only_lab_5/run_decoder_only_lab_5_generalization.sh

echo ""
echo "============================================"
echo "All decoder-only lab-5 jobs submitted!"
echo "Monitor with: bjobs"
echo "Check output in: hpc/output/decoder_only_lab_5/"
