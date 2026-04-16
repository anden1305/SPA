#!/bin/bash
#BSUB -J hmm_n5
#BSUB -o hpc/output/substages_analysis/hmm_n5_%J.out
#BSUB -e hpc/output/substages_analysis/hmm_n5_%J.err
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -R "rusage[mem=5GB]"
#BSUB -W 4:00
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"

# HMM Substage Analysis: n_states=5
# Validates on sub-048 runs 1-3

echo "========================================="
echo "HMM Substage Analysis: n_states=5"
echo "========================================="
echo "Job ID: $LSB_JOBID"
echo "Host: $(hostname)"
echo "Started: $(date)"
echo ""

CONFIG="src/config/run/hmm/substages/substage_analysis/substages_hmm_mssv_features_5.yaml"

# Load environment
source ~/.bashrc

# Train model
echo "→ Training model..."
uv run main.py --method train --config_path "$CONFIG"

# Find result directory
RESULT_DIR=$(ls -dt results/substages_analysis/hmm/substages_hmm_mssv_features_5_* 2>/dev/null | head -1)

if [ -z "$RESULT_DIR" ]; then
    echo "✗ ERROR: Could not find result directory"
    exit 1
fi

echo "✓ Training complete: $RESULT_DIR"

# Export to NPZ
echo ""
echo "→ Exporting predictions to NPZ..."
uv run python scripts/substage_analysis/export_predictions_to_npz.py "$RESULT_DIR"

echo ""
echo "========================================="
echo "✅ Job complete!"
echo "Finished: $(date)"
echo "========================================="
