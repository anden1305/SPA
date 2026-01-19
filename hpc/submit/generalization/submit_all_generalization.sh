#!/bin/bash
### Submit all generalization experiments (leave-one-mouse-out)
### Tests how well models trained on multiple mice generalize to unseen mice

set -euo pipefail

# Create output directory
mkdir -p hpc/output/generalization

echo "Submitting Generalization Experiments..."
echo "=========================================="

echo ""
echo "HMM Generalization Tests:"
echo "-------------------------"

echo "1. HMM - Test on sub-039 (train on all except sub-039)"
bsub < hpc/submit/generalization/run_gen_hmm_test_sub039.sh

echo "2. HMM - Test on sub-041 (train on all except sub-041)"
bsub < hpc/submit/generalization/run_gen_hmm_test_sub041.sh

echo "3. HMM - Test on sub-048 (train on all except sub-048)"
bsub < hpc/submit/generalization/run_gen_hmm_test_sub048.sh

echo ""
echo "MAR-HMM Generalization Tests:"
echo "-----------------------------"

echo "4. MAR-HMM - Test on sub-039 (train on all except sub-039)"
bsub < hpc/submit/generalization/run_gen_marhmm_test_sub039.sh

echo "5. MAR-HMM - Test on sub-041 (train on all except sub-041)"
bsub < hpc/submit/generalization/run_gen_marhmm_test_sub041.sh

echo "6. MAR-HMM - Test on sub-048 (train on all except sub-048)"
bsub < hpc/submit/generalization/run_gen_marhmm_test_sub048.sh

echo ""
echo "=========================================="
echo "All generalization jobs submitted!"
echo "Monitor with: bjobs"
echo "Check output in: hpc/output/generalization/"
echo ""
echo "Expected completion: ~12 hours"
echo "Expected metrics: Val NMI, Val likelihood, Train-Val gap"
echo "Analysis: Compare train vs val performance to assess overfitting"
