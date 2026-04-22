#!/bin/bash
### Submit all decoder-only cGMVAE experiments.
### The decoder-only experiment scripts are submitter wrappers that
### launch the actual LSF jobs for each experiment group.

set -euo pipefail

mkdir -p hpc/output/decoder_only

echo "Submitting Decoder-Only Experiments..."
echo "======================================"

echo ""
echo "Subjectwise Experiments:"
echo "------------------------"
bash hpc/submit/decoder_only/run_decoder_only_subjectwise.sh

echo ""
echo "Generalization-Subject Experiments:"
echo "-----------------------------------"
bash hpc/submit/decoder_only/run_decoder_only_generalization_subject.sh

echo ""
echo "Reliability Experiments:"
echo "------------------------"
bsub < hpc/submit/decoder_only/run_decoder_only_reliability.sh

echo ""
echo "Generalization Experiments:"
echo "---------------------------"
bash hpc/submit/decoder_only/run_decoder_only_generalization.sh

echo ""
echo "======================================"
echo "All decoder-only jobs submitted!"
echo "Monitor with: bjobs"
echo "Check output in: hpc/output/decoder_only/"
