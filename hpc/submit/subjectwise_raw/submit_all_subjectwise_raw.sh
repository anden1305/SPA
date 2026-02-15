#!/bin/bash
### Submit All Subject-wise Raw Experiments
### Launches both HMM and MARHMM raw experiments for all subjects

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "======================================"
echo "Submitting all subjectwise_raw experiments"
echo "======================================"
echo ""

# Submit HMM raw experiments
echo "1. Submitting HMM raw experiments..."
bash "${SCRIPT_DIR}/run_all_subjectwise_hmm_raw.sh"
echo ""

# Submit MARHMM raw experiments
echo "2. Submitting MARHMM raw experiments..."
bash "${SCRIPT_DIR}/run_all_subjectwise_marhmm_raw.sh"
echo ""

echo "======================================"
echo "All subjectwise_raw experiments submitted!"
echo "======================================"
echo ""
echo "Total jobs: 20 (10 HMM + 10 MARHMM)"
echo ""
echo "Monitor with: bjobs"
echo "Output logs: hpc/output/subjectwise_raw/{hmm,marhmm}/"
echo "Results: results/subjectwise_raw/{hmm,marhmm}/"

# Usage: bash ./hpc/submit/subjectwise_raw/submit_all_subjectwise_raw.sh
