#!/bin/bash
#BSUB -J dec_only_validate_all
#BSUB -q gpuv100
#BSUB -W 4:00
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=5GB]"
#BSUB -o hpc/output/decoder_only/validate_all/%J.out
#BSUB -e hpc/output/decoder_only/validate_all/%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail

module load cuda/12.8.1
source .venv/bin/activate

mkdir -p hpc/output/decoder_only/validate_all

python3 scripts/decoder_only/validate_decoder_only_all.py \
  --results_root results/decoder_only \
  --config_subdir validation_gmm \
  --include_shared
