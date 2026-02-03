#!/bin/bash
### Submit all substage analysis jobs to HPC

set -euo pipefail

echo "========================================="
echo "Submitting Substage Analysis Jobs"
echo "========================================="
echo ""

# Create output directory
mkdir -p hpc/output/substages_analysis

# Submit HMM jobs
echo "→ Submitting HMM jobs..."
for n in 5 6 7 8; do
    JOB_SCRIPT="hpc/submit/substages/analysis/run_hmm_n${n}.sh"
    echo "  Submitting: ${JOB_SCRIPT}"
    bsub < "${JOB_SCRIPT}"
done

echo ""
echo "→ Submitting MARHMM jobs..."
for n in 5 6 7 8; do
    JOB_SCRIPT="hpc/submit/substages/analysis/run_marhmm_n${n}.sh"
    echo "  Submitting: ${JOB_SCRIPT}"
    bsub < "${JOB_SCRIPT}"
done

echo ""
echo "========================================="
echo "✅ All jobs submitted!"
echo "========================================="
echo ""
echo "Monitor with: bjobs"
echo "Check output in: hpc/output/substages_analysis/"
echo ""
echo "Jobs:"
echo "  HMM: hmm_n5, hmm_n6, hmm_n7, hmm_n8"
echo "  MARHMM: marhmm_n5, marhmm_n6, marhmm_n7, marhmm_n8"
