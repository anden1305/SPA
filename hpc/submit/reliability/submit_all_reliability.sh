#!/bin/bash
### Submit all reliability experiments
### Each experiment runs 10 times with different seeds to test model consistency

set -euo pipefail

# Create output directory
mkdir -p hpc/output/reliability

echo "Submitting Reliability Experiments..."
echo "======================================"

echo ""
echo "1. HMM on MSSV Features (10 runs, ~24 hours)"
bsub < hpc/submit/reliability/run_reliability_hmm_mssv_features.sh

echo ""
echo "2. HMM on Synthetic Features (10 runs, ~12 hours)"
bsub < hpc/submit/reliability/run_reliability_hmm_synth_features.sh

echo ""
echo "3. MAR-HMM on MSSV Features (10 runs, ~24 hours)"
bsub < hpc/submit/reliability/run_reliability_marhmm_mssv_features.sh

echo ""
echo "4. MAR-HMM on Synthetic Features (10 runs, ~12 hours)"
bsub < hpc/submit/reliability/run_reliability_marhmm_synth_features.sh

echo ""
echo "======================================"
echo "All reliability jobs submitted!"
echo "Monitor with: bjobs"
echo "Check output in: hpc/output/reliability/"
echo ""
echo "Expected completion: ~24 hours"
echo "Expected metrics: cross-NMI (consistency), NMI vs true states, loss distribution"
