#!/bin/bash
### Submit all substages sweeps
### Each sweep tests n_states from 2 to 10 to find optimal model complexity

set -euo pipefail

# Note: Substages use W&B sweep launcher which creates hpc/output/sweep_<ID>/ automatically

echo "Submitting Substages Sweeps..."
echo "======================================"

echo ""
echo "1. HMM on MSSV Features (9 states x 1 trial = 9 jobs, ~12 hours total)"
bash hpc/submit/substages/launch_substages_hmm_mssv_features.sh

echo ""
echo "Waiting 5 seconds before next sweep..."
sleep 5

echo ""
echo "2. MAR-HMM on MSSV Features (9 states x 1 trial = 9 jobs, ~12 hours total)"
bash hpc/submit/substages/launch_substages_marhmm_mssv_features.sh

echo ""
echo "Waiting 5 seconds before next sweep..."
sleep 5

echo ""
echo "3. HMM on Synthetic Features (9 states x 1 trial = 9 jobs, ~8 hours total)"
bash hpc/submit/substages/launch_substages_hmm_synth_features.sh

echo ""
echo "Waiting 5 seconds before next sweep..."
sleep 5

echo ""
echo "4. MAR-HMM on Synthetic Features (9 states x 1 trial = 9 jobs, ~8 hours total)"
bash hpc/submit/substages/launch_substages_marhmm_synth_features.sh

echo ""
echo "======================================"
echo "All substages sweeps submitted!"
echo "Monitor with: bjobs"
echo "Check sweep outputs in: hpc/output/sweep_<SWEEP_ID>/"
echo ""
echo "Expected completion: ~12 hours"
echo "Expected metrics: NMI vs n_states, val_loss vs n_states"
echo ""
echo "AFTER COMPLETION - Generate analysis plots:"
echo "  1. Get sweep ID from wandb output or bjobs"
echo "  2. Run: python scripts/analyze_substages_sweep.py <SWEEP_ID>"
echo "  3. Plots will be saved to: results/substages_analysis/<SWEEP_ID>/"
echo ""
echo "Example:"
echo "  python scripts/analyze_substages_sweep.py abc123def456"
echo "  python scripts/analyze_substages_sweep.py abc123def456 --output_path results/my_analysis"
echo "======================================"
