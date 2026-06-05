#!/bin/bash
# Run after subject_lab_tune_winners jobs finish (login node: analysis only).
#
#   cd /work3/s204070/SPA
#   bash hpc/submit/cv4fold/run_after_subject_lab_batch.sh
#
# To also submit fold-4 subject-only ablation when cgmvae failed:
#   bash hpc/submit/cv4fold/run_after_subject_lab_batch.sh --submit-followup

set -euo pipefail
cd /work3/s204070/SPA
source .venv/bin/activate
export PYTHONPATH=.

ARGS=()
if [[ "${1:-}" == "--submit-followup" ]]; then
  ARGS+=(--submit-followup)
fi

python3 scripts/cv4fold/analyze_subject_lab_batch.py "${ARGS[@]}"

echo ""
echo "Report: results/cv4fold/subject_lab_tune_winners/analysis_report.md"
