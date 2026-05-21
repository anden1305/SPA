#!/bin/bash
#BSUB -J cnn_cgmvae_gmm_fix
#BSUB -q gpuv100
#BSUB -W 2:00
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -o hpc/output/cnn_cgmvae/gmm_fixed/%J.out
#BSUB -e hpc/output/cnn_cgmvae/gmm_fixed/%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "$SPA_ROOT"
mkdir -p hpc/output/cnn_cgmvae/gmm_fixed

module load cuda/12.8.1
source .venv/bin/activate

CONFIG="src/config/run/cvaeprior/cnn_cgmvae/sub039_cgmvae_gmm_fixed.yaml"
python3 main.py --method train_vae --config_path "${CONFIG}"

# Usage: bsub < hpc/submit/cnn_cgmvae/run_sub039_cgmvae_gmm_fixed.sh
