#!/bin/bash
#BSUB -J marhmm_n6
#BSUB -o hpc/output/substages_analysis/marhmm_n6_%J.out
#BSUB -e hpc/output/substages_analysis/marhmm_n6_%J.err
#BSUB -q gpul40s
#BSUB -n 4
#BSUB -R "rusage[mem=5GB]"
#BSUB -W 4:00
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"

# MARHMM Substage Analysis: n_states=6

echo "========================================="
echo "MARHMM Substage Analysis: n_states=6"
echo "========================================="
echo "Job ID: $LSB_JOBID"
echo "Host: $(hostname)"
echo "Started: $(date)"
echo ""

CONFIG="src/config/run/marhmm/substages/substage_analysis/substages_marhmm_mssv_features_6.yaml"

source ~/.bashrc

echo "→ Training model..."
uv run main.py --method train --config_path "$CONFIG"

RESULT_DIR=$(ls -dt results/substages_analysis/marhmm/substages_marhmm_mssv_features_6_* 2>/dev/null | head -1)

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
