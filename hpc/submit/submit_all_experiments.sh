#!/bin/bash
### Master script to submit ALL experiments (5, 6, 7)
### Use this to run the full experimental pipeline

set -euo pipefail

echo "╔════════════════════════════════════════════════════════╗"
echo "║  Submitting ALL Experiments (Reliability, Substages,  ║"
echo "║  Generalization)                                       ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

read -p "This will submit ~30+ jobs. Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "════════════════════════════════════════════════════════"
echo "EXPERIMENT 5: Reliability Analysis"
echo "════════════════════════════════════════════════════════"
bash hpc/submit/reliability/submit_all_reliability.sh

echo ""
echo "Waiting 10 seconds before next batch..."
sleep 10

echo ""
echo "════════════════════════════════════════════════════════"
echo "EXPERIMENT 6: Substages (State Count Sweep)"
echo "════════════════════════════════════════════════════════"
bash hpc/submit/substages/submit_all_substages.sh

echo ""
echo "Waiting 10 seconds before next batch..."
sleep 10

echo ""
echo "════════════════════════════════════════════════════════"
echo "EXPERIMENT 7: Generalization (Between-Mouse)"
echo "════════════════════════════════════════════════════════"
bash hpc/submit/generalization/submit_all_generalization.sh

echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║  ALL EXPERIMENTS SUBMITTED!                            ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""
echo "Monitor all jobs with: bjobs"
echo "Check outputs in: hpc/output/"
echo ""
echo "Estimated total time: ~24-36 hours"
echo ""
echo "Results will be in:"
echo "  - results/reliability/{hmm,marhmm}/"
echo "  - results/substages/{hmm,marhmm}/"
echo "  - results/generalization/{hmm,marhmm}/"
echo ""
echo "W&B Dashboard: https://wandb.ai/<your-entity>/SPA"
echo ""
