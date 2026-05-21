#!/bin/bash
#BSUB -J cnn_cgmvae_smoke_ae
#BSUB -q gpuv100
#BSUB -W 1:00
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=6GB]"
#BSUB -o hpc/output/cnn_cgmvae/smoke/%J.out
#BSUB -e hpc/output/cnn_cgmvae/smoke/%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

# CNN-front VAE smoke (raw_cnn, AE / standard prior, no KL).

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "$SPA_ROOT"
mkdir -p hpc/output/cnn_cgmvae/smoke

module load cuda/12.8.1
source .venv/bin/activate

CONFIG="src/config/run/cvaeprior/cnn_cgmvae/smoke_sub039_ae.yaml"

python3 main.py --method train_vae --config_path "${CONFIG}"
python3 main.py --method validate_cvae_gmm --config_path "${CONFIG}"

# Usage: bsub < hpc/submit/cnn_cgmvae/run_smoke_sub039_ae.sh
