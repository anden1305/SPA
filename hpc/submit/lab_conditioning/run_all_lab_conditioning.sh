#!/bin/bash
### Submit all lab-conditioning cGMVAE experiments.

set -euo pipefail

mkdir -p hpc/output/lab_conditioning

echo "Submitting Lab-Conditioning Experiments..."
echo "=========================================="


echo ""
echo "Lab-Only LOLO Experiments:"
echo "--------------------------"
bash hpc/submit/lab_conditioning/run_lab_conditioning_lab_only_LOLO.sh

echo ""
echo "Lab+Subject LOLO Experiments:"
echo "-----------------------------"
bash hpc/submit/lab_conditioning/run_lab_conditioning_lab_and_subject_LOLO.sh

echo ""
echo "Lab+Subject LOLO Experiments:"
echo "-----------------------------"
bash hpc/submit/lab_conditioning/run_lab_conditioning_subject_only_LOLO.sh

echo ""
echo "=========================================="
echo "All lab-conditioning jobs submitted!"
echo "Monitor with: bjobs"
echo "Check output in: hpc/output/lab_conditioning/"
