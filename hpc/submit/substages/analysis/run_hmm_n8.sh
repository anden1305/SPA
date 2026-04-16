#!/bin/bash
#BSUB -J hmm_n8
#BSUB -o hpc/output/substages_analysis/hmm_n8_%J.out
#BSUB -e hpc/output/substages_analysis/hmm_n8_%J.err
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -R "rusage[mem=5GB]"
#BSUB -W 4:00
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"

# HMM Substage Analysis: n_states=8

echo "========================================="
echo "HMM Substage Analysis: n_states=8"
echo "========================================="
echo "Job ID: $LSB_JOBID"
echo "Host: $(hostname)"
echo "Started: $(date)"
echo ""

CONFIG="src/config/run/hmm/substages/substage_analysis/substages_hmm_mssv_features_8.yaml"

source ~/.bashrc

echo "→ Training model..."
uv run main.py --method train --config_path "$CONFIG"

RESULT_DIR=$(ls -dt results/substages_analysis/hmm/substages_hmm_mssv_features_8_* 2>/dev/null | head -1)

if [ -z "$RESULT_DIR" ]; then
    echo "✗ ERROR: Could not find result directory"
    exit 1
fi

echo "✓ Training complete: $RESULT_DIR"
echo ""
echo "→ Exporting predictions to NPZ..."
uv run python scripts/substage_analysis/export_predictions_to_npz.py "$RESULT_DIR"

echo ""
echo "========================================="
echo "✅ Job complete!"
echo "Finished: $(date)"
echo "========================================="
