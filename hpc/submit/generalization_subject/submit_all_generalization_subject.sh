#!/bin/bash
### Submit all generalization_subject experiments (within-subject generalization)
### Tests how well models trained on runs 1,2 generalize to run 3 within the same subject

set -euo pipefail

# Create output directory
mkdir -p hpc/output/generalization_subject

echo "Submitting Generalization Subject Experiments..."
echo "================================================="

echo ""
echo "HMM Within-Subject Generalization Tests:"
echo "-----------------------------------------"

echo "1. HMM - sub-039 (train on runs 1,2, validate on run 3)"
bsub < hpc/submit/generalization_subject/run_gensubj_hmm_sub039.sh

echo "2. HMM - sub-041 (train on runs 1,2, validate on run 3)"
bsub < hpc/submit/generalization_subject/run_gensubj_hmm_sub041.sh

echo "3. HMM - sub-048 (train on runs 1,2, validate on run 3)"
bsub < hpc/submit/generalization_subject/run_gensubj_hmm_sub048.sh

echo ""
echo "MAR-HMM Within-Subject Generalization Tests:"
echo "---------------------------------------------"

echo "4. MAR-HMM - sub-039 (train on runs 1,2, validate on run 3)"
bsub < hpc/submit/generalization_subject/run_gensubj_marhmm_sub039.sh

echo "5. MAR-HMM - sub-041 (train on runs 1,2, validate on run 3)"
bsub < hpc/submit/generalization_subject/run_gensubj_marhmm_sub041.sh

echo "6. MAR-HMM - sub-048 (train on runs 1,2, validate on run 3)"
bsub < hpc/submit/generalization_subject/run_gensubj_marhmm_sub048.sh

echo ""
echo "================================================="
echo "All generalization_subject jobs submitted!"
echo "Monitor with: bjobs"
echo "Check output in: hpc/output/generalization_subject/"
echo ""
echo "Expected completion: ~2 hours"
echo "GPU distribution: gpua100(1), gpul40s(2), gpua10(1), gpuv100(3)"
echo "Expected metrics: Val NMI, Val likelihood, Train-Val gap"
echo "Analysis: Compare with generalization (between-subject) to assess"
echo "          within-subject vs between-subject generalization performance"
